#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preconditioned post-HH successor probabilities (precipitation-based antecedent state).

Literature-aligned precondition choices (raw mm; no precip percentiles required)
  1) dry_spell (default): no "wet day" in the last L days ending at day t
     (wet day: daily precip >= wet_day_mm, default 1.0 mm d-1; WMO-style dry day).
  2) low_accum: L-day cumulative precip <= accum_mm (antecedent meteorological dryness).

Estimands (per grid cell, then ensemble-mean across models)
  P_succ = P( successor category within next W days | HH_run_end_k(t) )
  P_pre  = P( successor ... | HH_run_end_k(t) AND precondition(t) )

Also writes P_diff = P_pre - P_succ for maps. Population-weighted CSV is optional
(--population-summary); omit it until you integrate exposure in the main pipeline.

Data layout (see project docs / user paths)
  TW/VPD: RAW/<SCEN_DIR>/TW_VPD_LANDONLY_SAFE/*<model>*.nc
  pr:     RAW/<SCEN_DIR>/<ModelFolder>/pr_merged*.nc OR pr_day*.nc
  pop:    RAW/population/population_2015soc_30arcmin_annual_*_0p25deg.nc
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

OUT_DIR = BASE / "figures_postHH_precondition_FINAL"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CACHE_DIR = BASE / "cache_postHH_precondition_FINAL"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

POP_HIST = RAW / "population" / "population_2015soc_30arcmin_annual_1850_2014_0p25deg.nc"
POP_FUT = RAW / "population" / "population_2015soc_30arcmin_annual_2015_2100_0p25deg.nc"

proj = ccrs.PlateCarree()
extent = [-130, 165, -55, 55]
lon_ticks = np.array([-120, -60, 0, 60, 120])
lat_ticks = np.arange(-60, 61, 30)

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


def model_variants(model: str) -> list[str]:
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


def find_model_data_root(scen: str, model: str) -> Path | None:
    root = RAW / SCEN_DIR[scen]
    if not root.exists():
        return None
    variants = model_variants(model)
    skip = {"TW_VPD_LANDONLY", "TW_VPD_LANDONLY_SAFE", "__pycache__"}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in skip:
            continue
        if any(v in d.name for v in variants):
            return d
    return None


def open_tw_vpd_stack(files: list[Path]) -> xr.Dataset:
    return xr.open_mfdataset(files, combine="by_coords", chunks={"time": 365}, engine="netcdf4")


def open_pr_mmday(root: Path) -> tuple[xr.DataArray, xr.Dataset]:
    merged = sorted(root.glob("pr_merged*.nc"))
    if merged:
        ds = xr.open_mfdataset(merged, combine="by_coords", chunks={"time": 365}, engine="netcdf4")
    else:
        daily = sorted(root.glob("pr_day*.nc"))
        if not daily:
            raise FileNotFoundError(f"No pr_merged*.nc or pr_day*.nc under {root}")
        ds = xr.open_mfdataset(daily, combine="by_coords", chunks={"time": 365}, engine="netcdf4")
    pr = ds["pr"] * 86400.0
    pr = pr.astype("float32")
    pr.name = "pr_mmday"
    pr.attrs["units"] = "mm/day"
    return pr, ds


def ensure_tw_celsius(tw: xr.DataArray) -> xr.DataArray:
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


def hh_run_end_mask(HH: xr.DataArray) -> xr.DataArray:
    return (HH == 1) & (HH.shift(time=-1, fill_value=0) == 0)


def hh_run_end_duration_ge(HH: xr.DataArray, HH_end: xr.DataArray, k: int) -> xr.DataArray:
    mask = HH_end
    for j in range(1, k):
        mask = mask & (HH.shift(time=j, fill_value=0) == 1)
    return mask


def successor_any_in_window(cat: xr.DataArray, *, window_days: int) -> xr.DataArray:
    shifted = [cat.shift(time=-d, fill_value=False) for d in range(1, window_days + 1)]
    stacked = xr.concat(shifted, dim="_lead")
    return stacked.any(dim="_lead")


def precondition_mask(
    pr_mmday: xr.DataArray,
    *,
    kind: str,
    L: int,
    wet_day_mm: float,
    accum_mm: float,
) -> xr.DataArray:
    """
    Boolean mask at each (time, lat, lon). True where precondition holds.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    if kind == "dry_spell":
        wet = pr_mmday >= float(wet_day_mm)
        wsum = wet.astype("float32").rolling(time=L, min_periods=L).sum()
        out = wsum == 0
        return out.fillna(False)
    if kind == "low_accum":
        s = pr_mmday.rolling(time=L, min_periods=L).sum()
        out = s <= float(accum_mm)
        return out.fillna(False)
    raise ValueError(f"Unknown precondition kind: {kind}")


def compute_baseline_and_preconditioned_probs(
    tw: xr.DataArray,
    vpd: xr.DataArray,
    pre: xr.DataArray,
    *,
    tw_thr: xr.DataArray,
    vpd_thr: xr.DataArray,
    vpd_dry_thr: xr.DataArray,
    durations: list[int],
    window_days: int,
    categories: list[str],
) -> tuple[dict[int, dict[str, xr.DataArray]], dict[int, dict[str, xr.DataArray]]]:
    HH = ((tw > tw_thr) & (vpd > vpd_thr)).astype("int8")
    HH_end = hh_run_end_mask(HH).astype(bool)

    hot = tw > tw_thr
    dry = vpd > vpd_dry_thr
    cat_maps: dict[str, xr.DataArray] = {
        "compound_again": hot & dry,
        "hot_only": hot & (~dry),
        "dry_only": (~hot) & dry,
    }
    cat_maps["non_extreme"] = ~(cat_maps["compound_again"] | cat_maps["hot_only"] | cat_maps["dry_only"])

    cat_any: dict[str, xr.DataArray] = {}
    for cat in categories:
        cat_any[cat] = successor_any_in_window(cat_maps[cat], window_days=window_days)

    base: dict[int, dict[str, xr.DataArray]] = {}
    pre_d: dict[int, dict[str, xr.DataArray]] = {}

    for k in durations:
        HH_end_k = hh_run_end_duration_ge(HH, HH_end, k)
        denom_b = HH_end_k.sum(dim="time").where(HH_end_k.sum(dim="time") > 0)
        denom_p = (HH_end_k & pre).sum(dim="time").where((HH_end_k & pre).sum(dim="time") > 0)

        base[k] = {}
        pre_d[k] = {}
        for cat in categories:
            pb = (HH_end_k & cat_any[cat]).sum(dim="time") / denom_b
            pp = (HH_end_k & pre & cat_any[cat]).sum(dim="time") / denom_p
            base[k][cat] = pb.astype("float32").where(np.isfinite(pb))
            pre_d[k][cat] = pp.astype("float32").where(np.isfinite(pp))

    return base, pre_d


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
    pre_kind: str,
    pre_L: int,
    wet_day_mm: float,
    accum_mm: float,
    tw_quantile: float,
    vpd_quantile: float,
    vpd_dry_quantile: float,
) -> Path:
    dur_tag = "dur" + "-".join(map(str, durations))
    cat_tag = "cat" + "-".join(map(_slug, categories))
    mod_tag = "mods" + "-".join(map(_slug, models))
    pre_tag = f"pre{_slug(pre_kind)}_L{pre_L}_wet{wet_day_mm:g}_acc{accum_mm:g}"
    tw_tag = f"twQ{tw_quantile:g}"
    vpd_tag = f"vpdQ{vpd_quantile:g}"
    dry_tag = f"dryQ{vpd_dry_quantile:g}"
    name = f"postHH_precond_{scen}_win{window_days}_{dur_tag}_{cat_tag}_{mod_tag}_{pre_tag}_{tw_tag}_{vpd_tag}_{dry_tag}.nc"
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


def plot_probability_map(
    prob: xr.DataArray,
    *,
    title: str,
    outpath: Path,
    dpi: int,
    cbar_label: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "inferno",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": proj})
    fig.subplots_adjust(right=0.82)
    im = ax.pcolormesh(
        prob["lon"].values,
        prob["lat"].values,
        prob.values,
        transform=proj,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    format_axis(ax)
    if title:
        ax.set_title(title, pad=18)
    bbox = ax.get_position()
    h = bbox.height * CBAR_HEIGHT_FRAC
    y0 = bbox.y0 + (bbox.height - h) * 0.5
    cax = fig.add_axes([bbox.x1 + CBAR_PAD_FRAC_OF_FIG, y0, CBAR_WIDTH_FRAC_OF_FIG, h])
    cb = fig.colorbar(im, cax=cax)
    # Extra label padding prevents label/tick overlap in mosaics.
    cb.set_label(textwrap.fill(cbar_label, width=22), fontsize=15, labelpad=24, rotation=90, va="center")
    cb.ax.tick_params(labelsize=12)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def lon_to_0360(da: xr.DataArray) -> xr.DataArray:
    lon = (da.lon % 360).astype("float32")
    return da.assign_coords(lon=lon).sortby("lon")


def population_field_for_year(pop_path: Path, year: int, var: str) -> xr.DataArray:
    ds = xr.open_dataset(pop_path, chunks={"lat": 256, "lon": 256})
    if var not in ds:
        ds.close()
        raise KeyError(f"{var} not in {list(ds.data_vars)}")
    # Annual stamps YYYY-01-01 12:00 (proleptic_gregorian)
    tsel = ds.sel(time=np.datetime64(f"{year}-01-01"), method="nearest", drop=True)
    pop = tsel[var].squeeze(drop=True).astype("float32").load()
    ds.close()
    return lon_to_0360(pop)


def population_weighted_mean(prob: xr.DataArray, pop_on_climate: xr.DataArray) -> float:
    """Mean probability weighted by population (masked where prob is not finite)."""
    w = pop_on_climate.where(np.isfinite(prob))
    num = float((prob * w).sum(skipna=True).values)
    den = float(w.sum(skipna=True).values)
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num / den)


def regrid_pop_to_prob(pop: xr.DataArray, prob: xr.DataArray) -> xr.DataArray:
    return pop.interp(lat=prob["lat"], lon=prob["lon"], method="linear")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="*", default=["historical"])
    p.add_argument("--models", nargs="*", default=MODELS_DEFAULT)
    p.add_argument("--durations", nargs="*", type=int, default=[3, 7])
    p.add_argument("--window-days", type=int, default=7)
    p.add_argument("--tw-quantile", type=float, default=0.99)
    p.add_argument("--vpd-quantile", type=float, default=0.95)
    p.add_argument("--vpd-dry-quantile", type=float, default=0.95)
    p.add_argument(
        "--categories",
        nargs="*",
        default=["compound_again", "hot_only", "dry_only", "non_extreme"],
    )
    p.add_argument(
        "--precondition-kind",
        choices=("dry_spell", "low_accum"),
        default="dry_spell",
        help="dry_spell: no wet day in last L days; low_accum: L-day sum <= accum-mm.",
    )
    p.add_argument("--pre-L", type=int, default=14, help="Antecedent window length (days) ending at trigger day t.")
    p.add_argument("--wet-day-mm", type=float, default=1.0, help="Precip threshold for a wet day (mm/day).")
    p.add_argument("--accum-mm", type=float, default=5.0, help="For low_accum: max L-day total precip (mm).")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--plot-diff", action="store_true", help="Also save P(pre)-P(base) maps (RdBu_r, symmetric scale).")
    p.add_argument("--dpi", type=int, default=350)
    p.add_argument(
        "--style",
        choices=("default", "ncc"),
        default="default",
        help="Figure style preset. 'ncc' reduces clutter and uses journal-like typography.",
    )
    p.add_argument(
        "--no-titles",
        action="store_true",
        help="Suppress per-map titles (recommended for Nature-style multi-panel plates).",
    )
    p.add_argument(
        "--population-summary",
        action="store_true",
        help="Write CSV of population-weighted mean P (baseline vs pre) using fixed pop year.",
    )
    p.add_argument("--population-year", type=int, default=2010, help="Which annual population slice to use.")
    p.add_argument(
        "--population-var",
        default="total-population",
        help="Dataset variable name (e.g. total-population).",
    )
    args = p.parse_args()

    # Make progress visible even when stdout is buffered (common on HPC / wrappers)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if args.style == "ncc":
        import matplotlib as mpl

        mpl.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "font.size": 14,
                "axes.titlesize": 14,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
            }
        )

    durations = list(map(int, args.durations))
    window_days = int(args.window_days)
    categories = list(args.categories)
    pre_kind = str(args.precondition_kind)
    pre_L = int(args.pre_L)

    summary_rows: list[str] = []

    for scen in args.scenarios:
        print(f"\nScenario: {scen}")
        cache_fp = cache_path(
            cache_dir=args.cache_dir,
            scen=scen,
            models=list(args.models),
            durations=durations,
            window_days=window_days,
            categories=categories,
            pre_kind=pre_kind,
            pre_L=pre_L,
            wet_day_mm=float(args.wet_day_mm),
            accum_mm=float(args.accum_mm),
            tw_quantile=float(args.tw_quantile),
            vpd_quantile=float(args.vpd_quantile),
            vpd_dry_quantile=float(args.vpd_dry_quantile),
        )

        if cache_fp.exists() and not args.force_recompute:
            print(f"  Cache hit: {cache_fp}")
            ds_out = load_cache(cache_fp)
        else:
            if args.plot_only:
                raise RuntimeError(f"--plot-only but missing cache: {cache_fp}")
            print(f"  Cache miss -> compute: {cache_fp}")

            ens_pb: dict[int, dict[str, np.ndarray]] = {k: {c: None for c in categories} for k in durations}
            ens_pp: dict[int, dict[str, np.ndarray]] = {k: {c: None for c in categories} for k in durations}
            n_used = 0

            for model in args.models:
                tw_files = find_tw_vpd_files(scen, model)
                root = find_model_data_root(scen, model)
                if not tw_files or root is None:
                    print(f"  [skip] {model}: tw={bool(tw_files)} pr_root={root is not None}")
                    continue
                try:
                    pr_raw, pr_ds = open_pr_mmday(root)
                except FileNotFoundError as e:
                    print(f"  [skip] {model}: {e}")
                    continue

                print(f"  Model: {model}")
                ds = open_tw_vpd_stack(tw_files)
                tw = ds["tw"]
                vpd = ds["vpd"]
                tw_c = ensure_tw_celsius(tw)

                tw_a, pr_a = xr.align(tw_c, pr_raw, join="inner")
                vpd_a = vpd.sel(time=tw_a.time)
                pre = precondition_mask(
                    pr_a,
                    kind=pre_kind,
                    L=pre_L,
                    wet_day_mm=float(args.wet_day_mm),
                    accum_mm=float(args.accum_mm),
                )

                tw_thr, vpd_thr, vpd_dry_thr = compute_thresholds(
                    tw_a,
                    vpd_a,
                    tw_quantile=args.tw_quantile,
                    vpd_quantile=args.vpd_quantile,
                    vpd_dry_quantile=args.vpd_dry_quantile,
                )

                pb, pp = compute_baseline_and_preconditioned_probs(
                    tw_a,
                    vpd_a,
                    pre,
                    tw_thr=tw_thr,
                    vpd_thr=vpd_thr,
                    vpd_dry_thr=vpd_dry_thr,
                    durations=durations,
                    window_days=window_days,
                    categories=categories,
                )

                for k in durations:
                    for cat in categories:
                        b = pb[k][cat].compute().values.astype("float32")
                        prm = pp[k][cat].compute().values.astype("float32")
                        if ens_pb[k][cat] is None:
                            ens_pb[k][cat] = b
                            ens_pp[k][cat] = prm
                        else:
                            ens_pb[k][cat] += b
                            ens_pp[k][cat] += prm

                n_used += 1
                ds.close()
                pr_ds.close()

            if n_used == 0:
                print(f"  [ERROR] No models for {scen}")
                continue

            coord_files = find_tw_vpd_files(scen, args.models[0])
            ds0 = open_tw_vpd_stack(coord_files[:1])
            lat_coord = ds0["lat"]
            lon_coord = ds0["lon"]
            ds0.close()

            def stack_probs(ens: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
                return np.stack(
                    [np.stack([ens[k][c] / float(n_used) for c in categories], axis=0) for k in durations],
                    axis=0,
                ).astype("float32")

            data_b = stack_probs(ens_pb)
            data_p = stack_probs(ens_pp)
            data_d = data_p - data_b

            ds_out = xr.Dataset(
                {
                    "P_baseline": (
                        ("duration", "category", "lat", "lon"),
                        data_b,
                    ),
                    "P_preconditioned": (("duration", "category", "lat", "lon"), data_p),
                    "P_diff": (("duration", "category", "lat", "lon"), data_d),
                },
                coords={
                    "duration": np.array(durations, dtype=np.int32),
                    "category": categories,
                    "lat": lat_coord,
                    "lon": lon_coord,
                },
            )
            ds_out.attrs.update(
                scenario=scen,
                window_days=window_days,
                precondition_kind=pre_kind,
                precondition_L=pre_L,
                wet_day_mm=float(args.wet_day_mm),
                accum_mm=float(args.accum_mm),
                models=",".join(args.models),
                n_models=int(n_used),
                tw_quantile=float(args.tw_quantile),
                vpd_quantile=float(args.vpd_quantile),
                vpd_dry_quantile=float(args.vpd_dry_quantile),
            )
            print(f"  Write cache: {cache_fp}")
            write_cache(ds_out, cache_fp)
            ds_out = load_cache(cache_fp)

        pre_tag = f"{pre_kind}_L{pre_L}"

        for k in durations:
            for cat in categories:
                da_b = ds_out["P_baseline"].sel(duration=k, category=cat)
                da_p = ds_out["P_preconditioned"].sel(duration=k, category=cat)
                da_d = ds_out["P_diff"].sel(duration=k, category=cat)

                # Full cbar labels (previous settings)
                lab_b = f"P({cat} within W days | HH_end?{k})"
                lab_p = f"P({cat} within W days | HH_end?{k}, antecedent={pre_tag})"
                title_b = "" if args.no_titles else f"{scen} | baseline | HH_end?{k} | W={window_days} | {cat}"
                title_p = "" if args.no_titles else f"{scen} | preconditioned | HH_end?{k} | W={window_days} | {cat}"

                out_b = OUT_DIR / f"postHH_pre_{scen}_dur{k}_win{window_days}_{cat}_{pre_tag}_baseline.png"
                out_p = OUT_DIR / f"postHH_pre_{scen}_dur{k}_win{window_days}_{cat}_{pre_tag}_precond.png"
                print(f"  Save: {out_b}")
                plot_probability_map(da_b, title=title_b, outpath=out_b, dpi=args.dpi, cbar_label=lab_b)
                print(f"  Save: {out_p}")
                plot_probability_map(da_p, title=title_p, outpath=out_p, dpi=args.dpi, cbar_label=lab_p)

                if args.plot_diff:
                    out_d = OUT_DIR / f"postHH_pre_{scen}_dur{k}_win{window_days}_{cat}_{pre_tag}_diff.png"
                    vmax = 0.35
                    plot_probability_map(
                        da_d,
                        title="" if args.no_titles else f"{scen} | ?P (pre ? base) | HH_end?{k} | W={window_days} | {cat} | {pre_tag}",
                        outpath=out_d,
                        dpi=args.dpi,
                        cbar_label="?P = P(preconditioned) ? P(baseline)",
                        vmin=-vmax,
                        vmax=vmax,
                        cmap="RdBu_r",
                    )

                if args.population_summary:
                    pop_path = POP_HIST if scen == "historical" else POP_FUT
                    y = int(args.population_year)
                    pop = population_field_for_year(pop_path, y, args.population_var)
                    pop_i = regrid_pop_to_prob(pop, da_b)
                    mb = population_weighted_mean(da_b, pop_i)
                    mp = population_weighted_mean(da_p, pop_i)
                    summary_rows.append(
                        f"{scen},{cat},dur{k},win{window_days},{pre_tag},{y},{mb:.6f},{mp:.6f},{mp - mb:.6f}"
                    )

        ds_out.close()

    if args.population_summary and summary_rows:
        out_csv = OUT_DIR / "population_weighted_mean_probabilities.csv"
        hdr = "scenario,category,duration,window,precond,pop_year,P_baseline_weighted,P_pre_weighted,delta\n"
        out_csv.write_text(hdr + "\n".join(summary_rows) + "\n")
        print(f"\nWrote {out_csv}")

    print("\nDONE")


if __name__ == "__main__":
    main()
