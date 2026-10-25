#!/usr/bin/env python3
"""
Fig. 5 population exposure panels.

Reads:
  cache_population_exposure_HH/HH_exposure_<scenario>_<model>_years*_total-population.nc
  variables: exposure_persondays, HH_days (year, lat, lon)

Writes under Manuscript final plots/figures/Fig5_population_exposure/:
  Fig5_exposure_row_SSP*.png, Fig5_exposure_3x3.png,
  Fig5_exposure_timeseries_row.png, Fig5_zonal_HHdays_Sen_slope.png,
  Fig5_composite_compact.png, Fig5_exposure_colorbar_horizontal.png

Usage:
  export CHHE_PROJECT_ROOT=/path/to/compound_humid_heat
  conda activate mir
  python plot_fig5_population_exposure.py
  python plot_fig5_population_exposure.py --composite-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import theilslopes

_SCRIPTS = Path(__file__).resolve().parents[1]
for extra in (_SCRIPTS / "utils", _SCRIPTS / "helpers"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
from path_config import PROJECT_ROOT, LAND_MASK  # noqa: E402

try:
    from manuscript_basemap import format_ax_light
except Exception:  # pragma: no cover
    def format_ax_light(ax, show_x=True, show_y=True):
        ax.coastlines(linewidth=0.6)
        ax.set_extent([-130, 155, -55, 55], crs=ccrs.PlateCarree())

CACHE_DIR = PROJECT_ROOT / "cache_population_exposure_HH"
OUTDIR = PROJECT_ROOT / "Manuscript final plots" / "figures" / "Fig5_population_exposure"
PANEL_DIR = OUTDIR / "panels"
INSET_DIR = OUTDIR / "insets"
OUTDIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR.mkdir(parents=True, exist_ok=True)
INSET_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["EC_Earth3", "MPI-ESM1-2-HR", "HadGEM3-GC31-LL"]
POP_VAR = "total-population"
SCENARIOS = [
    ("SSP126", "ssp126", ["a", "b", "c"]),
    ("SSP245", "ssp245", ["d", "e", "f"]),
    ("SSP585", "ssp585", ["g", "h", "i"]),
]
EPOCHS = [("NF", 2021, 2040), ("MC", 2041, 2060), ("EC", 2081, 2100)]
REGIONS = {
    "Amz": {"lon": (-80, -45), "lat": (-15, 5)},
    "CAf": {"lon": (10, 40), "lat": (0, 15)},
    "SEA": {"lon": (90, 130), "lat": (-10, 25)},
    "SAs": {"lon": (60, 100), "lat": (5, 35)},
}
REGION_LABELS_SHORT = ["Amz", "CAf", "SEA", "SAs"]
SSP_COLORS = {"ssp126": "#3288bd", "ssp245": "#fdae61", "ssp585": "#d7191c"}
CMAP_EXPOSURE = LinearSegmentedColormap.from_list(
    "exposure_wbgyr", ["#ffffff", "#3288bd", "#66bd63", "#fee08b", "#c51b1f"]
)

proj = ccrs.PlateCarree()
extent = [-130, 155, -55, 55]
lon_ticks = [-70, -10, 50, 110]
lat_ticks = [35, 5, -25, -55]
PNG_DPI = 400

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

land_mask = None
if LAND_MASK.exists():
    land_mask = xr.open_dataarray(LAND_MASK).load()


def _cache_path(scenario: str, model: str) -> Path:
    if scenario == "historical":
        return CACHE_DIR / f"HH_exposure_historical_{model}_years1981-2014_{POP_VAR}.nc"
    return CACHE_DIR / f"HH_exposure_{scenario}_{model}_years2015-2100_{POP_VAR}.nc"


def _open_model_caches(scenario: str) -> list[xr.Dataset]:
    paths = [_cache_path(scenario, m) for m in MODELS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing caches for {scenario}: {missing}")
    return [xr.open_dataset(p) for p in paths]


def _common_years(datasets: list[xr.Dataset]) -> np.ndarray:
    years = datasets[0]["year"].values
    for ds in datasets[1:]:
        years = np.intersect1d(years, ds["year"].values)
    return years


def _ensemble_epoch_field(scenario: str, var: str, y0: int, y1: int) -> xr.DataArray:
    dsets = _open_model_caches(scenario)
    years = _common_years(dsets)
    years = years[(years >= y0) & (years <= y1)]
    fields = []
    for ds in dsets:
        da = ds[var].sel(year=years).mean("year")
        if land_mask is not None:
            da = da.where(land_mask)
        fields.append(da)
        ds.close()
    out = xr.concat(fields, dim="model").mean("model")
    out.name = var
    return out


def _annual_global_exposure(scenario: str):
    dsets = _open_model_caches(scenario)
    years = _common_years(dsets)
    # area-weight not required for person-days already on equal-area-ish 0.25 grid counts
    series = []
    for ds in dsets:
        e = ds["exposure_persondays"].sel(year=years)
        if land_mask is not None:
            e = e.where(land_mask)
        series.append(e.sum(dim=("lat", "lon")))
        ds.close()
    stack = xr.concat(series, dim="model")
    return years, stack.mean("model").values, stack.min("model").values, stack.max("model").values


def _zonal_mean_hh_per_year(scenario: str):
    dsets = _open_model_caches(scenario)
    years = _common_years(dsets)
    lat = dsets[0]["lat"]
    w = np.cos(np.deg2rad(lat)).clip(min=0)
    fields = []
    for ds in dsets:
        hh = ds["HH_days"].sel(year=years)
        if land_mask is not None:
            hh = hh.where(land_mask)
        z = (hh * w).sum("lon") / (w * (~hh.isnull()).sum("lon")).where(lambda x: x > 0)
        fields.append(z)
        ds.close()
    return xr.concat(fields, dim="model").mean("model")


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTDIR / f"{stem}.png", dpi=PNG_DPI)
    fig.savefig(OUTDIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"Wrote {stem}")


def _plot_map_row(scen_label, scen_tag, letters, data, cmap, vmin, vmax, stem_prefix, draw_regions=False):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2), subplot_kw={"projection": proj})
    for ax, (ep_label, y0, y1), letter in zip(axes, EPOCHS, letters):
        da = data[(scen_tag, ep_label)]
        # `data` is expected to already be log10(1+E) for exposure rows
        im = ax.pcolormesh(da["lon"], da["lat"], da.values, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        format_ax_light(ax)
        ax.set_title(f"{letter}  {scen_label} {ep_label}", fontsize=10)
        if draw_regions and ep_label == "EC":
            for name, box in REGIONS.items():
                ax.plot(
                    [box["lon"][0], box["lon"][1], box["lon"][1], box["lon"][0], box["lon"][0]],
                    [box["lat"][0], box["lat"][0], box["lat"][1], box["lat"][1], box["lat"][0]],
                    color="k",
                    lw=0.8,
                    transform=proj,
                )
                ax.text(np.mean(box["lon"]), box["lat"][1] + 2, name, fontsize=7, ha="center", transform=proj)
    fig.subplots_adjust(wspace=0.05, right=0.90)
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}(1+E)$ (person-days)")
    _save(fig, f"{stem_prefix}_{scen_label}")


def plot_timeseries_epochs():
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
    for ax, (ep_label, y0, y1), letter in zip(axes, EPOCHS, ["j", "k", "l"]):
        for ssp_label, ssp_tag, _ in SCENARIOS:
            years, mean, lo, hi = _annual_global_exposure(ssp_tag)
            m = (years >= y0) & (years <= y1)
            ax.fill_between(years[m], lo[m] / 1e9, hi[m] / 1e9, color=SSP_COLORS[ssp_tag], alpha=0.2, linewidth=0)
            ax.plot(years[m], mean[m] / 1e9, color=SSP_COLORS[ssp_tag], lw=1.5, label=ssp_label)
        ax.set_title(f"{letter}  {ep_label}")
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3, lw=0.4)
    axes[0].set_ylabel(r"Exposure ($10^{9}$ pers.-d yr$^{-1}$)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, "Fig5_exposure_timeseries_row")


def plot_zonal_sen_slope():
    fig, ax = plt.subplots(figsize=(4.2, 5.0))
    for ssp_label, ssp_tag, _ in SCENARIOS:
        z = _zonal_mean_hh_per_year(ssp_tag)
        slopes = []
        lats = z["lat"].values
        years = z["year"].values.astype(float)
        for i in range(len(lats)):
            y = z.isel(lat=i).values.astype(float)
            ok = np.isfinite(y)
            if ok.sum() < 10:
                slopes.append(np.nan)
                continue
            slope, *_ = theilslopes(y[ok], years[ok])
            slopes.append(slope)
        ax.plot(slopes, lats, color=SSP_COLORS[ssp_tag], lw=1.4, label=ssp_label)
    ax.axhspan(-20, 20, color="0.85", zorder=0)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(r"Sen slope (d yr$^{-2}$)")
    ax.set_ylabel("Latitude")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(-60, 60)
    fig.tight_layout()
    _save(fig, "Fig5_zonal_HHdays_Sen_slope")


def plot_exposure_3x3(log_exp, vmax):
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.5), subplot_kw={"projection": proj})
    im = None
    for r, (scen_label, scen_tag, letters) in enumerate(SCENARIOS):
        for c, (ep_label, y0, y1) in enumerate(EPOCHS):
            ax = axes[r, c]
            da = log_exp[(scen_tag, ep_label)]
            im = ax.pcolormesh(da["lon"], da["lat"], da.values, transform=proj, cmap=CMAP_EXPOSURE, vmin=0, vmax=vmax, shading="auto")
            format_ax_light(ax)
            ax.set_title(f"{letters[c]}  {scen_label} {ep_label}", fontsize=9)
    fig.subplots_adjust(hspace=0.15, wspace=0.05, right=0.90)
    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}(1+E)$ (person-days)")
    _save(fig, "Fig5_exposure_3x3")


def plot_fig5_composite_compact(log_exp, vmax):
    fig = plt.figure(figsize=(12.8, 10.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.85], hspace=0.22, wspace=0.05)
    im = None
    for r, (scen_label, scen_tag, letters) in enumerate(SCENARIOS):
        for c, (ep_label, y0, y1) in enumerate(EPOCHS):
            ax = fig.add_subplot(gs[r, c], projection=proj)
            da = log_exp[(scen_tag, ep_label)]
            im = ax.pcolormesh(da["lon"], da["lat"], da.values, transform=proj, cmap=CMAP_EXPOSURE, vmin=0, vmax=vmax, shading="auto")
            format_ax_light(ax)
            ax.set_title(f"{letters[c]}  {scen_label} {ep_label}", fontsize=9)
    for c, (ep_label, y0, y1) in enumerate(EPOCHS):
        ax = fig.add_subplot(gs[3, c])
        letter = ["j", "k", "l"][c]
        for ssp_label, ssp_tag, _ in SCENARIOS:
            years, mean, lo, hi = _annual_global_exposure(ssp_tag)
            m = (years >= y0) & (years <= y1)
            ax.fill_between(years[m], lo[m] / 1e9, hi[m] / 1e9, color=SSP_COLORS[ssp_tag], alpha=0.18, linewidth=0)
            ax.plot(years[m], mean[m] / 1e9, color=SSP_COLORS[ssp_tag], lw=1.3, label=ssp_label)
        ax.set_title(f"{letter}  {ep_label}")
        ax.grid(True, alpha=0.3, lw=0.4)
        if c == 0:
            ax.set_ylabel(r"Exposure ($10^{9}$ pers.-d yr$^{-1}$)")
            ax.legend(fontsize=7, frameon=False)
    cax = fig.add_axes([0.92, 0.38, 0.015, 0.45])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}(1+E)$ (person-days)")
    _save(fig, "Fig5_composite_compact")


def plot_colorbar_exposure(vmax):
    fig, ax = plt.subplots(figsize=(6, 0.45))
    sm = plt.cm.ScalarMappable(cmap=CMAP_EXPOSURE, norm=mpl.colors.Normalize(vmin=0, vmax=vmax))
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label(r"$\log_{10}(1+E)$ (person-days)")
    _save(fig, "Fig5_exposure_colorbar_horizontal")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig. 5 population exposure panels")
    ap.add_argument("--composite-only", action="store_true", help="Build only Fig5_composite_compact")
    ap.add_argument("--ssp126-row-only", action="store_true", help="Rebuild only SSP126 exposure row")
    args = ap.parse_args()

    print("Loading epoch ensemble fields...")
    log_exp: dict[tuple[str, str], xr.DataArray] = {}
    all_vals = []
    for _, scen_tag, _ in SCENARIOS:
        for ep_label, y0, y1 in EPOCHS:
            e = _ensemble_epoch_field(scen_tag, "exposure_persondays", y0, y1)
            log = np.log10(np.maximum(e, 0.0) + 1.0)
            log_exp[(scen_tag, ep_label)] = log
            all_vals.append(log.values[np.isfinite(log.values)])
    vmax = float(np.nanpercentile(np.concatenate(all_vals), 99.5))
    print(f"Exposure scale (p99.5) = {vmax:.3f}")

    if args.ssp126_row_only:
        _plot_map_row("SSP126", "ssp126", ["a", "b", "c"], log_exp, CMAP_EXPOSURE, 0, vmax, "Fig5_exposure_row", draw_regions=True)
        return

    if not args.composite_only:
        for scen_label, scen_tag, letters in SCENARIOS:
            _plot_map_row(scen_label, scen_tag, letters, log_exp, CMAP_EXPOSURE, 0, vmax, "Fig5_exposure_row", draw_regions=(scen_tag == "ssp126"))
        plot_exposure_3x3(log_exp, vmax)
        plot_timeseries_epochs()
        plot_zonal_sen_slope()
        plot_colorbar_exposure(vmax)

    plot_fig5_composite_compact(log_exp, vmax)
    print("DONE (reconstructed Fig. 5 plotter)")


if __name__ == "__main__":
    main()
