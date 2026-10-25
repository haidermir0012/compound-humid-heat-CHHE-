#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-HH transition maps: what extremes tend to follow a multi-day HH event?

Conditioning event
  HH_run_end_k(t): HH(t)=1, HH(t+1)=0, and the HH run length ending at t is >= k.

Successor definition (hazard window)
  For successor category X, we compute:
    P( X occurs at least once in [t+1, ..., t+W] | HH_run_end_k(t) )

Successor categories (mutually exclusive at any single day)
  hot(t)   := tw(t)   > tw_thr(0.99)
  dry(t)   := vpd(t)  > vpd_thr(dry_quantile)  (high-tail VPD = atmospheric dryness)

  compound_again: hot & dry
  hot_only:       hot & ~dry
  dry_only:       ~hot & dry
  non_extreme: otherwise (i.e., not in the three extreme categories)

Implementation details
  - tw thresholds are computed from raw TW/VPD files per model+scenario+grid.
  - vpd thresholds are computed similarly for humidity extremes.
  - dry uses a high-tail VPD quantile (default 0.95).
  - Ensemble mean is computed across the 3 CMIP6 models used elsewhere.

Note on "dry":
  In atmospheric science, high VPD corresponds to atmospheric dryness (water-demand).
  Therefore "dry" is defined using high-tail VPD (vpd above its dry quantile).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from lag_cartopy_basemap import add_land_borders_no_disputed


BASE = Path("/intelnvme04/haider/project/compound_humid_heat")
RAW = Path("/intelnvme04/haider/raw_data")

OUT_DIR = BASE / "figures_postHH_trigger_FINAL"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CACHE_DIR = BASE / "cache_postHH_trigger_FINAL"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

proj = ccrs.PlateCarree()

extent = [-130, 165, -55, 55]
lon_ticks = np.array([-120, -60, 0, 60, 120])
lat_ticks = np.arange(-60, 61, 30)

# Match the "individual lag ensemble" styling tuned earlier
LABEL_PAD_X_EXTRA_PT = 0.1
AXIS_TICK_FONTSIZE = 22
CBAR_WIDTH_FRAC_OF_FIG = 0.022
CBAR_PAD_FRAC_OF_FIG = 0.043
CBAR_HEIGHT_FRAC = 0.995

SCENARIOS_DEFAULT = ["historical", "ssp126", "ssp245", "ssp585"]
MODELS_DEFAULT = ["EC_Earth3", "HadGEM3-GC31-LL", "MPI-ESM1-2-HR"]

SCEN_DIR = {
    "historical": "nexgddp_historical_1981_2014",
    "ssp126": "nexgddp_ssp126_2015_2100",
    "ssp245": "nexgddp_ssp245_2015_2100",
    "ssp585": "nexgddp_ssp585_2015_2100",
}


def format_axis(ax: plt.Axes) -> None:
    ax.set_extent(extent, crs=proj)
    ax.coastlines(linewidth=1.2, zorder=5)
    add_land_borders_no_disputed(ax)

    ax.set_xticks(lon_ticks, crs=proj)
    ax.set_yticks(lat_ticks, crs=proj)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.tick_params(direction="out", length=5, width=1)
    ax.tick_params(axis="x", pad=11.0 + LABEL_PAD_X_EXTRA_PT, labelsize=AXIS_TICK_FONTSIZE)
    ax.tick_params(axis="y", pad=9.5, labelsize=AXIS_TICK_FONTSIZE)


def model_variants(model: str) -> list[str]:
    # Raw files appear to use underscores; lag pipelines use hyphens in names.
    return list(dict.fromkeys([model, model.replace("-", "_"), model.replace("_", "-")]))


def find_tw_vpd_files(scen: str, model: str) -> list[Path]:
    base = RAW / SCEN_DIR[scen] / "TW_VPD_LANDONLY_SAFE"
    if not base.exists():
        return []
    variants = model_variants(model)
    files: list[Path] = []
    for fp in base.iterdir():
        if fp.suffix != ".nc":
            continue
        name = fp.name
        if any(v in name for v in variants):
            files.append(fp)
    return sorted(files)


def open_tw_vpd_stack(files: list[Path]) -> xr.Dataset:
    # Chunk in time to keep quantiles / sums memory-safe.
    return xr.open_mfdataset(files, combine="by_coords", chunks={"time": 365}, engine="netcdf4")


def ensure_tw_celsius(tw: xr.DataArray) -> xr.DataArray:
    # tw sometimes stored in Kelvin.
    tw_max = float(tw.max().compute())
    if tw_max > 200:
        return tw - 273.15
    return tw


def compute_thresholds(
    tw: xr.DataArray,
    vpd: xr.DataArray,
    *,
    tw_quantile: float,
    vpd_quantile: float,
    vpd_dry_quantile: float,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    tw = tw.chunk({"time": 365})
    vpd = vpd.chunk({"time": 365})

    # xarray's built-in quantile API here doesn't expose dask's allow_rechunk,
    # so we implement the reduction explicitly via apply_ufunc.
    def qtime(da: xr.DataArray, q: float) -> xr.DataArray:
        return xr.apply_ufunc(
            np.nanquantile,
            da,
            q,
            input_core_dims=[["time"], []],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"allow_rechunk": True},
        ).astype("float32")

    tw_thr = qtime(tw, tw_quantile)
    vpd_thr = qtime(vpd, vpd_quantile)
    vpd_dry_thr = qtime(vpd, vpd_dry_quantile)
    return tw_thr, vpd_thr, vpd_dry_thr


def _slug(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)


def cache_path(
    *,
    cache_dir: Path,
    scen: str,
    models: list[str],
    durations: list[int],
    window_days: int,
    categories: list[str],
    tw_quantile: float,
    vpd_quantile: float,
    vpd_dry_quantile: float,
    tw_threshold: float | None,
    vpd_threshold: float | None,
    vpd_dry_threshold: float | None,
) -> Path:
    # One cache per scenario + threshold spec + window + (durations, categories, models)
    # so style-only reruns can be plot-only.
    if tw_threshold is not None:
        tw_tag = f"twAbs{tw_threshold:g}"
    else:
        tw_tag = f"twQ{tw_quantile:g}"
    if vpd_threshold is not None:
        vpd_tag = f"vpdAbs{vpd_threshold:g}"
    else:
        vpd_tag = f"vpdQ{vpd_quantile:g}"
    if vpd_dry_threshold is not None:
        dry_tag = f"dryAbs{vpd_dry_threshold:g}"
    else:
        dry_tag = f"dryQ{vpd_dry_quantile:g}"

    dur_tag = "dur" + "-".join(map(str, durations))
    cat_tag = "cat" + "-".join(map(_slug, categories))
    mod_tag = "mods" + "-".join(map(_slug, models))

    name = f"postHH_probs_{scen}_win{window_days}_{dur_tag}_{cat_tag}_{mod_tag}_{tw_tag}_{vpd_tag}_{dry_tag}.nc"
    return cache_dir / _slug(name)


def write_cache(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    ds.to_netcdf(tmp, engine="netcdf4", format="NETCDF4")
    tmp.replace(path)


def load_cache(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="netcdf4")


def hh_run_end_mask(HH: xr.DataArray) -> xr.DataArray:
    """
    HH_end(t) = HH(t)=1 AND HH(t+1)=0
    """
    return (HH == 1) & (HH.shift(time=-1, fill_value=0) == 0)


def hh_run_end_duration_ge(HH: xr.DataArray, HH_end: xr.DataArray, k: int) -> xr.DataArray:
    """
    HH_end_k(t) means the HH run ending at t has duration >= k.
    At end day t, we require HH(t), HH(t-1), ..., HH(t-k+1) all equal 1.
    """
    mask = HH_end
    for j in range(1, k):
        mask = mask & (HH.shift(time=j, fill_value=0) == 1)
    return mask


def successor_any_in_window(cat: xr.DataArray, *, window_days: int) -> xr.DataArray:
    """
    cat_any(t) = OR_{d=1..window_days} cat(t+d)
    """
    shifted = [cat.shift(time=-d, fill_value=False) for d in range(1, window_days + 1)]
    stacked = xr.concat(shifted, dim="_lead")
    return stacked.any(dim="_lead")


def compute_successor_probabilities(
    tw: xr.DataArray,
    vpd: xr.DataArray,
    *,
    tw_thr: xr.DataArray,
    vpd_thr: xr.DataArray,
    vpd_dry_thr: xr.DataArray,
    durations: list[int],
    window_days: int,
    categories: list[str],
) -> dict[int, dict[str, xr.DataArray]]:
    # HH and end masks
    HH = ((tw > tw_thr) & (vpd > vpd_thr)).astype("int8")
    HH_end = hh_run_end_mask(HH).astype(bool)

    # Component extremes at each day
    hot = (tw > tw_thr)
    # Literature convention: high VPD corresponds to atmospheric dryness.
    dry = (vpd > vpd_dry_thr)

    cat_maps: dict[str, xr.DataArray] = {
        "compound_again": hot & dry,
        "hot_only": hot & (~dry),
        "dry_only": (~hot) & dry,
    }
    cat_maps["non_extreme"] = ~(cat_maps["compound_again"] | cat_maps["hot_only"] | cat_maps["dry_only"])

    # Precompute successor "any in next N days" booleans for each category
    cat_any: dict[str, xr.DataArray] = {}
    for cat in categories:
        cat_any[cat] = successor_any_in_window(cat_maps[cat], window_days=window_days)

    out: dict[int, dict[str, xr.DataArray]] = {}
    for k in durations:
        HH_end_k = hh_run_end_duration_ge(HH, HH_end, k)
        denom = HH_end_k.sum(dim="time")
        denom = denom.where(denom > 0)

        out[k] = {}
        for cat in categories:
            hits = (HH_end_k & cat_any[cat]).sum(dim="time")
            prob = (hits / denom).astype("float32")
            # Use NaN where denom==0 (avoid inf)
            prob = prob.where(np.isfinite(prob))
            prob.name = f"P_postHH_{cat}_dur{k}_win{window_days}"
            out[k][cat] = prob
    return out


def plot_probability_map(
    prob: xr.DataArray,
    *,
    title: str,
    outpath: Path,
    dpi: int,
    cbar_label: str,
) -> None:
    # prob dims: lat, lon
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": proj})
    fig.subplots_adjust(right=0.82)

    im = ax.pcolormesh(
        prob["lon"].values,
        prob["lat"].values,
        prob.values,
        transform=proj,
        cmap="inferno",
        vmin=0,
        vmax=1,
        shading="auto",
    )

    format_axis(ax)
    ax.set_title(title, pad=18)

    bbox = ax.get_position()
    h = bbox.height * CBAR_HEIGHT_FRAC
    y0 = bbox.y0 + (bbox.height - h) * 0.5

    cax = fig.add_axes([bbox.x1 + CBAR_PAD_FRAC_OF_FIG, y0, CBAR_WIDTH_FRAC_OF_FIG, h])
    cb = fig.colorbar(im, cax=cax)
    # Avoid label clipping by wrapping and slightly smaller font.
    wrapped = textwrap.fill(cbar_label, width=28)
    cb.set_label(wrapped, fontsize=13, labelpad=14, rotation=90, va="center")
    cb.ax.tick_params(labelsize=13)

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="*", default=SCENARIOS_DEFAULT)
    p.add_argument("--models", nargs="*", default=MODELS_DEFAULT)
    p.add_argument("--durations", nargs="*", type=int, default=[3, 7], help="HH end run length thresholds (days).")
    p.add_argument("--window-days", type=int, default=3, help="Successor hazard window: check next 1..W days (any-occurrence).")
    p.add_argument("--tw-quantile", type=float, default=0.99, help="Heat (tw) threshold quantile.")
    p.add_argument("--vpd-quantile", type=float, default=0.95, help="Humidity (vpd) threshold quantile.")
    p.add_argument(
        "--vpd-dry-quantile",
        type=float,
        default=0.95,
        help="Dry (high-tail vpd) threshold quantile. 'Dry' category uses vpd > this value.",
    )
    p.add_argument(
        "--categories",
        nargs="*",
        default=["compound_again", "hot_only", "dry_only", "non_extreme"],
        help="Subset of categories to output.",
    )
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Where to cache computed probability maps.")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Only plot from cached probability maps (no recomputation). Fails if cache is missing.",
    )
    p.add_argument("--force-recompute", action="store_true", help="Ignore cache and recompute probability maps.")
    p.add_argument(
        "--tw-threshold",
        type=float,
        default=None,
        help="Use an absolute TW threshold in Celsius (e.g., 32 or 35). Overrides --tw-quantile.",
    )
    p.add_argument(
        "--vpd-threshold",
        type=float,
        default=None,
        help="Use an absolute VPD threshold. Overrides --vpd-quantile.",
    )
    p.add_argument(
        "--vpd-dry-threshold",
        type=float,
        default=None,
        help="Use an absolute dry VPD threshold. Overrides --vpd-dry-quantile.",
    )
    p.add_argument("--dpi", type=int, default=350)
    args = p.parse_args()

    durations: list[int] = list(map(int, args.durations))
    window_days = int(args.window_days)
    categories: list[str] = list(args.categories)

    for scen in args.scenarios:
        print(f"\nScenario: {scen}")
        for k in durations:
            for cat in categories:
                print(f"  Will output: dur={k} cat={cat}")

        cache_fp = cache_path(
            cache_dir=args.cache_dir,
            scen=scen,
            models=list(args.models),
            durations=durations,
            window_days=window_days,
            categories=categories,
            tw_quantile=float(args.tw_quantile),
            vpd_quantile=float(args.vpd_quantile),
            vpd_dry_quantile=float(args.vpd_dry_quantile),
            tw_threshold=args.tw_threshold,
            vpd_threshold=args.vpd_threshold,
            vpd_dry_threshold=args.vpd_dry_threshold,
        )

        if cache_fp.exists() and not args.force_recompute:
            print(f"  Cache hit: {cache_fp}")
            ds_cached = load_cache(cache_fp)
        else:
            if args.plot_only:
                raise RuntimeError(f"--plot-only set but cache missing: {cache_fp}")

            print(f"  Cache miss: {cache_fp}")
            # Accumulate ensemble mean (sum then divide by n)
            ensemble_sum: dict[int, dict[str, np.ndarray]] = {k: {} for k in durations}
            n_used = 0

            for model in args.models:
                files = find_tw_vpd_files(scen, model)
                if not files:
                    print(f"  [skip] missing raw TW/VPD for {model} in {scen}")
                    continue

                print(f"  Model: {model} | files={len(files)}")
                ds = open_tw_vpd_stack(files)
                tw = ds["tw"]
                vpd = ds["vpd"]
                tw_c = ensure_tw_celsius(tw)

                if args.tw_threshold is not None:
                    tw_thr = xr.full_like(tw_c.isel(time=0, drop=True), float(args.tw_threshold), dtype="float32")
                if args.vpd_threshold is not None:
                    vpd_thr = xr.full_like(vpd.isel(time=0, drop=True), float(args.vpd_threshold), dtype="float32")
                if args.vpd_dry_threshold is not None:
                    vpd_dry_thr = xr.full_like(vpd.isel(time=0, drop=True), float(args.vpd_dry_threshold), dtype="float32")

                if (args.tw_threshold is None) or (args.vpd_threshold is None) or (args.vpd_dry_threshold is None):
                    tw_thr_q, vpd_thr_q, vpd_dry_thr_q = compute_thresholds(
                        tw_c,
                        vpd,
                        tw_quantile=args.tw_quantile,
                        vpd_quantile=args.vpd_quantile,
                        vpd_dry_quantile=args.vpd_dry_quantile,
                    )
                    if args.tw_threshold is None:
                        tw_thr = tw_thr_q
                    if args.vpd_threshold is None:
                        vpd_thr = vpd_thr_q
                    if args.vpd_dry_threshold is None:
                        vpd_dry_thr = vpd_dry_thr_q

                # Compute conditional successor probabilities
                probs = compute_successor_probabilities(
                    tw_c,
                    vpd,
                    tw_thr=tw_thr,
                    vpd_thr=vpd_thr,
                    vpd_dry_thr=vpd_dry_thr,
                    durations=durations,
                    window_days=window_days,
                    categories=categories,
                )

                # Compute and accumulate
                for k in durations:
                    for cat in categories:
                        da = probs[k][cat].compute().astype("float32")
                        if cat not in ensemble_sum[k]:
                            ensemble_sum[k][cat] = da.values
                        else:
                            ensemble_sum[k][cat] += da.values

                n_used += 1
                ds.close()

            if n_used == 0:
                print(f"  [ERROR] No models processed for scenario={scen}")
                continue

            print(f"  Ensemble mean computed from {n_used} models.")

            # Load coordinates once (avoid repeated raw reads inside the category loop)
            coord_files = find_tw_vpd_files(scen, args.models[0])
            if not coord_files:
                raise RuntimeError(f"Cannot load coordinates for scenario={scen}")
            ds0 = open_tw_vpd_stack(coord_files[:1])
            lat_coord = ds0["lat"]
            lon_coord = ds0["lon"]
            ds0.close()

            # Build cached dataset: prob(duration, category, lat, lon)
            data = np.stack(
                [np.stack([ensemble_sum[k][cat] / float(n_used) for cat in categories], axis=0) for k in durations],
                axis=0,
            ).astype("float32")
            da_prob = xr.DataArray(
                data,
                dims=("duration", "category", "lat", "lon"),
                coords={"duration": durations, "category": categories, "lat": lat_coord, "lon": lon_coord},
                name="prob",
            )
            ds_cached = xr.Dataset({"prob": da_prob})
            # NetCDF attrs cannot store None; only write optional threshold attrs when set.
            ds_cached.attrs.update(
                dict(
                    scenario=scen,
                    window_days=int(window_days),
                    models=",".join(args.models),
                    tw_quantile=float(args.tw_quantile),
                    vpd_quantile=float(args.vpd_quantile),
                    vpd_dry_quantile=float(args.vpd_dry_quantile),
                )
            )
            if args.tw_threshold is not None:
                ds_cached.attrs["tw_threshold"] = float(args.tw_threshold)
            if args.vpd_threshold is not None:
                ds_cached.attrs["vpd_threshold"] = float(args.vpd_threshold)
            if args.vpd_dry_threshold is not None:
                ds_cached.attrs["vpd_dry_threshold"] = float(args.vpd_dry_threshold)

            print(f"  Write cache: {cache_fp}")
            write_cache(ds_cached, cache_fp)

        # Plot from cached dataset (fast)
        for k in durations:
            for cat in categories:
                da_plot = ds_cached["prob"].sel(duration=k, category=cat).rename(f"P_postHH_{cat}")

                label_map = {
                    "compound_again": "P(compound again within W days | HH_end≥k)",
                    "hot_only": "P(hot-only within W days | HH_end≥k)",
                    "dry_only": "P(dry-only within W days | HH_end≥k)",
                    "non_extreme": "P(non-extreme within W days | HH_end≥k)",
                }
                cbar_label = label_map.get(cat, "Probability")
                title = f"{scen} | HH run ended (dur >= {k} days) | next {window_days} days: {cat}"

                outpath = OUT_DIR / f"postHH_transition_{scen}_dur{k}_win{window_days}_{cat}.png"
                print(f"  Save: {outpath}")
                plot_probability_map(da_plot, title=title, outpath=outpath, dpi=args.dpi, cbar_label=cbar_label)

    print("\nDONE")


if __name__ == "__main__":
    # This script is compute-heavy; warnings are muted but errors will still show.
    main()

