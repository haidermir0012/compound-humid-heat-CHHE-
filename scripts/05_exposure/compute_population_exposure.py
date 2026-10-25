#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Population exposure to local-quantile compound humid-heat (HH) extremes (person-days).

Manuscript Fig. 5 core metric (climate-driven exposure; fixed 2015soc population):
  E_y(x) = Pop_2015(x) * D_y(x)
where D_y(x) is the annual count of local-quantile HH days
(TW > cell Q0.99 and VPD > cell Q0.95 from percentile_HH_FINAL).

Method notes (final publication alignment)
  - Population counts are aligned to matching 0.25° HH cell centres (nearest;
    no bilinear interpolation of the extensive population field).
  - Future population is held fixed at the 2015soc 2015 slice across SSPs.
  - Incomplete model-years are excluded (not interpolated):
      ssp126 / EC_Earth3: 2100
      ssp585 / HadGEM3-GC31-LL: 2030

Inputs
  percentile_HH_FINAL/<scenario>/<model>/hh_percentile.nc  (var: HH)
  raw_data/population/population_2015soc_*_0p25deg.nc

Outputs
  cache_population_exposure_HH/HH_exposure_<scen>_<model>_years*_total-population.nc

Path configuration: set CHHE_PROJECT_ROOT and CHHE_RAW_DATA (see utils/path_config.py).
"""
from __future__ import annotations

import argparse
import sys
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

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS / "utils") not in sys.path:
    sys.path.insert(0, str(_SCRIPTS / "utils"))
from path_config import PROJECT_ROOT, RAW_DATA, LAND_MASK  # noqa: E402

HH_DIR = PROJECT_ROOT / "percentile_HH_FINAL"
OUT_CACHE = PROJECT_ROOT / "cache_population_exposure_HH"
OUT_CACHE.mkdir(parents=True, exist_ok=True)
OUT_FIG = PROJECT_ROOT / "figures_population_exposure_HH"
OUT_FIG.mkdir(parents=True, exist_ok=True)

POP_HIST = RAW_DATA / "population" / "population_2015soc_30arcmin_annual_1850_2014_0p25deg.nc"
POP_FUT = RAW_DATA / "population" / "population_2015soc_30arcmin_annual_2015_2100_0p25deg.nc"

MODELS_DEFAULT = ["EC_Earth3", "HadGEM3-GC31-LL", "MPI-ESM1-2-HR"]
EXCLUDED_MODEL_YEARS = {
    ("ssp126", "EC_Earth3"): {2100},
    ("ssp585", "HadGEM3-GC31-LL"): {2030},
}

proj = ccrs.PlateCarree()
extent = [-130, 165, -55, 55]
lon_ticks = np.array([-120, -60, 0, 60, 120])
lat_ticks = np.arange(-60, 61, 30)


def lon_to_0360(da: xr.DataArray) -> xr.DataArray:
    lon = (da["lon"] % 360).astype("float32")
    return da.assign_coords(lon=lon).sortby("lon")


def open_population_for_year(*, scenario: str, year: int, var: str) -> xr.DataArray:
    pop_path = POP_HIST if scenario == "historical" else POP_FUT
    with xr.open_dataset(pop_path, chunks={"lat": 256, "lon": 256}) as ds:
        if var not in ds:
            raise KeyError(f"{var} not in {list(ds.data_vars)}")
        tsel = ds.sel(time=np.datetime64(f"{year}-01-01"), method="nearest", drop=True)
        pop = tsel[var].squeeze(drop=True).astype("float32").load()
    return lon_to_0360(pop)


def align_pop_to_hh(pop: xr.DataArray, hh: xr.DataArray) -> xr.DataArray:
    """Align population counts to matching HH cell centres without interpolation."""
    pop = pop.sortby("lat").sortby("lon")
    aligned = pop.reindex(
        lat=hh["lat"],
        lon=hh["lon"],
        method="nearest",
        tolerance=1.0e-6,
    )
    if bool(aligned.isnull().all()):
        raise RuntimeError("Population and HH grids do not share matching cell centres")
    source_total = float(pop.sum(skipna=True))
    aligned_total = float(aligned.sum(skipna=True))
    if source_total > 0 and abs(aligned_total - source_total) / source_total > 1.0e-6:
        raise RuntimeError(
            f"Population count not preserved: source={source_total}, aligned={aligned_total}"
        )
    return aligned


def format_axis(ax: plt.Axes) -> None:
    ax.set_extent(extent, crs=proj)
    ax.coastlines(linewidth=1.0, zorder=5)
    ax.set_xticks(lon_ticks, crs=proj)
    ax.set_yticks(lat_ticks, crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=12)


def plot_map(
    da: xr.DataArray,
    *,
    title: str,
    outpath: Path,
    cbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 350,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": proj})
    fig.subplots_adjust(right=0.83)
    im = ax.pcolormesh(
        da["lon"].values,
        da["lat"].values,
        da.values,
        transform=proj,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    format_axis(ax)
    ax.set_title(title, pad=12, fontsize=14)
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x1 + 0.02, bbox.y0 + 0.02, 0.02, bbox.height * 0.90])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=12)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def find_hh_file(scenario: str, model: str) -> Path | None:
    for m in [model, model.replace("-", "_"), model.replace("_", "-")]:
        p = HH_DIR / scenario / m / "hh_percentile.nc"
        if p.exists():
            return p
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="historical", choices=("historical", "ssp126", "ssp245", "ssp585"))
    p.add_argument("--models", nargs="*", default=MODELS_DEFAULT)
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument("--pop-var", default="total-population")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dpi", type=int, default=350)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    scen = str(args.scenario)
    if args.start_year is None or args.end_year is None:
        y0, y1 = (1981, 2014) if scen == "historical" else (2015, 2100)
    else:
        y0, y1 = int(args.start_year), int(args.end_year)

    mask = None
    if LAND_MASK.exists():
        mask = xr.open_dataarray(LAND_MASK).load()

    for model in args.models:
        in_nc = find_hh_file(scen, model)
        if in_nc is None:
            print(f"[skip] missing HH file for {scen}/{model}")
            continue

        tag = f"{scen}_{model}_years{y0}-{y1}_{args.pop_var}".replace("/", "_")
        out_nc = OUT_CACHE / f"HH_exposure_{tag}.nc"
        if out_nc.exists() and not args.force:
            print(f"Cache hit: {out_nc}")
            continue

        print(f"Compute: {scen} | {model} | years {y0}-{y1}")
        ds = xr.open_dataset(in_nc, chunks={"time": 365}, engine="netcdf4")
        HH = ds["HH"].astype("int8").sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
        HH_days_all = HH.groupby("time.year").sum("time").astype("float32")
        HH_days_all = HH_days_all.sel(year=slice(y0, y1))
        if mask is not None:
            HH_days_all = HH_days_all.where(mask)

        excluded = EXCLUDED_MODEL_YEARS.get((scen, model), set())
        if excluded:
            HH_days_all = HH_days_all.sel(year=~HH_days_all["year"].isin(sorted(excluded)))

        pop_year = 2015 if scen != "historical" else y0
        pop = open_population_for_year(scenario=scen, year=pop_year, var=args.pop_var)
        pop_hh = align_pop_to_hh(pop, HH_days_all.isel(year=0, drop=True))
        if mask is not None:
            pop_hh = pop_hh.where(mask)

        exposure = (HH_days_all * pop_hh).astype("float32")
        exposure.name = "exposure_persondays"
        HH_days_all.name = "HH_days"

        out = xr.Dataset(
            {
                "HH_days": HH_days_all,
                "exposure_persondays": exposure,
                "HH_days_mean": HH_days_all.mean("year"),
                "exposure_persondays_mean": exposure.mean("year"),
            }
        )
        out.attrs.update(
            scenario=scen,
            model=model,
            start_year=y0,
            end_year=y1,
            pop_var=args.pop_var,
            exposure_units="population_units * days",
            population_reference_year=int(pop_year),
            population_alignment=(
                "Direct nearest-coordinate alignment between matching 0.25-degree "
                "cell centres; no interpolation of population counts"
            ),
            population_total_check="Preserved within 1e-6 relative tolerance",
            excluded_years=str(sorted(excluded)) if excluded else "none",
        )
        out.to_netcdf(out_nc, engine="netcdf4")
        ds.close()
        print(f"Wrote cache: {out_nc}")

        if not args.no_plots:
            HH_days_mean = out["HH_days_mean"]
            EXP_mean = out["exposure_persondays_mean"]
            plot_map(
                HH_days_mean,
                title=f"{scen} | {model} | mean HH days ({y0}-{y1})",
                outpath=OUT_FIG / f"mean_HH_days_{scen}_{model}_{y0}-{y1}.png",
                cbar_label="days/year",
                cmap="magma",
                vmin=0,
                vmax=float(np.nanpercentile(HH_days_mean.values, 99)),
                dpi=args.dpi,
            )
            plot_map(
                np.log10(EXP_mean + 1.0),
                title=f"{scen} | {model} | log10(1 + mean exposure)",
                outpath=OUT_FIG / f"mean_exposure_persondays_log10p1_{scen}_{model}_{y0}-{y1}.png",
                cbar_label="log10(1 + person-days)",
                cmap="viridis",
                vmin=0,
                vmax=float(np.nanpercentile(np.log10(EXP_mean.values + 1.0), 99)),
                dpi=args.dpi,
            )

    print("DONE")


if __name__ == "__main__":
    main()
