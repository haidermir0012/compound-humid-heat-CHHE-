#!/usr/bin/env python3
"""
Figure 3 (FINAL): Conditional persistence of compound HH after event onset.

P(HH at t+τ | HH_start at t) — ensemble mean across 3 CMIP6 models.

Reads:
  lag_analysis_FINAL/lag_HH_<scenario>_<model>.nc

Writes (PNG + PDF):
  Manuscript final plots/figures/Fig3_persistence/
    Fig3_persistence_3x3.png/.pdf          — full figure (SSP rows × τ columns)
    Fig3_persistence_row_SSP126.png/.pdf   — manual assembly rows (τ=1,3,7)
    Fig3_persistence_row_SSP245.png/.pdf
    Fig3_persistence_row_SSP585.png/.pdf   — includes horizontal colorbar
    Fig3_persistence_row_historical.png/.pdf  — optional baseline row
    Fig3_persistence_colorbar.png/.pdf
    panels/Fig3_persist_{letter}_{ssp}_tau{tau}.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

_HELPERS = Path(__file__).resolve().parents[1] / "helpers"
if str(_HELPERS) not in sys.path:
    sys.path.insert(0, str(_HELPERS))
from manuscript_basemap import format_ax_light

PROJECT_DIR = Path("/intelnvme04/haider/project/compound_humid_heat")
DATA_DIR = PROJECT_DIR / "lag_analysis_FINAL"
OUTDIR = PROJECT_DIR / "Manuscript final plots" / "figures" / "Fig3_persistence"
PANEL_DIR = OUTDIR / "panels"
OUTDIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["EC_Earth3", "HadGEM3-GC31-LL", "MPI-ESM1-2-HR"]
SCENARIOS = [
    ("SSP126", "ssp126"),
    ("SSP245", "ssp245"),
    ("SSP585", "ssp585"),
]
HISTORICAL = ("Historical", "historical")
LAGS = [1, 3, 7]
# Supplementary Fig. S5 (historical row): own panel letters
HIST_LETTERS = ["a", "b", "c"]

LETTERS_3X3 = [
    ["a", "b", "c"],
    ["d", "e", "f"],
    ["g", "h", "i"],
]

proj = ccrs.PlateCarree()
extent = [-130, 155, -55, 55]
lon_ticks = [-70, -10, 50, 110]
lat_ticks = [35, 5, -25, -55]

PNG_DPI = 600

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

land_mask = xr.open_dataarray(PROJECT_DIR / "land_mask.nc").load()
CMAP = plt.cm.inferno
VMIN, VMAX = 0.0, 1.0


def _load_model(scen_tag: str, model: str) -> xr.Dataset | None:
    p = DATA_DIR / f"lag_HH_{scen_tag}_{model}.nc"
    if not p.exists():
        return None
    return xr.open_dataset(p)


def build_ensemble_field(scen_tag: str, tau: int) -> xr.DataArray:
    parts = []
    for m in MODELS:
        ds = _load_model(scen_tag, m)
        if ds is None:
            continue
        var = "P_HH_given_HHstart" if "P_HH_given_HHstart" in ds else list(ds.data_vars)[0]
        da = ds[var].sel(lag=tau).load()
        ds.close()
        parts.append(da)

    if not parts:
        raise FileNotFoundError(f"No lag files for scenario={scen_tag}")

    mean = xr.concat(parts, dim="model").mean("model")
    return mean.where(land_mask == 1)


def format_ax(ax, *, show_y: bool, show_x: bool) -> None:
    format_ax_light(
        ax,
        proj=proj,
        extent=extent,
        lon_ticks=lon_ticks,
        lat_ticks=lat_ticks,
        show_y=show_y,
        show_x=show_x,
        show_ipcc=False,
    )


def _plot_field(ax, da: xr.DataArray) -> mpl.cm.ScalarMappable:
    return ax.pcolormesh(
        da["lon"].values,
        da["lat"].values,
        da.values,
        transform=proj,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        shading="auto",
        rasterized=True,
        zorder=2,
    )


def _panel_letter(row: int, col: int) -> str:
    return LETTERS_3X3[row][col]


def add_panel_letter(ax, letter: str) -> None:
    ax.text(
        0.0,
        1.02,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="0.10",
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")


def plot_grid_3x3() -> None:
    """Full Fig. 3: 3 SSP rows × 3 τ columns; panel letters only (no titles)."""
    fig = plt.figure(figsize=(10.2, 6.25))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 0.055], hspace=0.02, wspace=0.03)
    im = None

    for ri, (_, scen_tag) in enumerate(SCENARIOS):
        for ci, tau in enumerate(LAGS):
            ax = fig.add_subplot(gs[ri, ci], projection=proj)
            da = build_ensemble_field(scen_tag, tau)
            im = _plot_field(ax, da)
            format_ax(ax, show_y=(ci == 0), show_x=(ri == 2))
            add_panel_letter(ax, _panel_letter(ri, ci))

    cax = fig.add_subplot(gs[3, :])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max", extendrect=False)
    cb.set_label(
        r"$P(\mathrm{HH}_{t+\tau}\,|\,\mathrm{HH\_start}_t)$",
        fontsize=10,
        labelpad=2,
    )
    cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    _save(fig, OUTDIR / "Fig3_persistence_3x3")
    plt.close(fig)
    print("Wrote Fig3_persistence_3x3.png/.pdf")


def plot_row(
    *,
    ssp_label: str,
    scen_tag: str,
    row_index: int,
    attach_cbar: bool,
) -> None:
    if attach_cbar:
        fig = plt.figure(figsize=(10.2, 2.38))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.055], hspace=0.02, wspace=0.03)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax = fig.add_subplot(gs[1, :])
    else:
        fig = plt.figure(figsize=(10.2, 2.05))
        gs = fig.add_gridspec(1, 3, wspace=0.03)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax = None

    im = None
    for ci, tau in enumerate(LAGS):
        da = build_ensemble_field(scen_tag, tau)
        im = _plot_field(axes[ci], da)
        format_ax(axes[ci], show_y=(ci == 0), show_x=True)
        add_panel_letter(axes[ci], _panel_letter(row_index, ci))

    stem = OUTDIR / f"Fig3_persistence_row_{ssp_label}"
    if attach_cbar and cax is not None and im is not None:
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max", extendrect=False)
        cb.set_label(
            r"$P(\mathrm{HH}_{t+\tau}\,|\,\mathrm{HH\_start}_t)$",
            fontsize=10,
            labelpad=2,
        )
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)
        stem = OUTDIR / f"Fig3_persistence_row_{ssp_label}_cbar"

    _save(fig, stem)
    plt.close(fig)
    print(f"Wrote {stem.name}.png/.pdf")


def plot_historical_row() -> None:
    """Supplementary Fig. S5: historical baseline row (τ = 1, 3, 7), panels a–c."""
    fig = plt.figure(figsize=(10.2, 2.05))
    gs = fig.add_gridspec(1, 3, wspace=0.03)
    axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
    for ci, (tau, letter) in enumerate(zip(LAGS, HIST_LETTERS)):
        da = build_ensemble_field(HISTORICAL[1], tau)
        _plot_field(axes[ci], da)
        format_ax(axes[ci], show_y=(ci == 0), show_x=True)
        add_panel_letter(axes[ci], letter)

    _save(fig, OUTDIR / "FigS5_persistence_historical")
    plt.close(fig)
    print("Wrote FigS5_persistence_historical.png/.pdf")


def plot_standalone_panels() -> None:
    for ri, (ssp_label, scen_tag) in enumerate(SCENARIOS):
        for ci, tau in enumerate(LAGS):
            letter = _panel_letter(ri, ci)
            da = build_ensemble_field(scen_tag, tau)
            fig = plt.figure(figsize=(3.4, 1.85))
            ax = plt.axes(projection=proj)
            _plot_field(ax, da)
            format_ax(ax, show_y=True, show_x=True)
            add_panel_letter(ax, letter)
            stem = PANEL_DIR / f"Fig3_persist_{letter}_{ssp_label}_tau{tau}"
            fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
            plt.close(fig)
    print(f"Wrote 9 panels in {PANEL_DIR}")


def plot_colorbar_standalone() -> None:
    fig = plt.figure(figsize=(4.3, 0.55))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.07, 0.55, 0.90, 0.25])
    norm = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=CMAP, norm=norm, orientation="horizontal", extend="max")
    cb.set_label(
        r"$P(\mathrm{HH}_{t+\tau}\,|\,\mathrm{HH\_start}_t)$",
        fontsize=10,
    )
    cb.ax.tick_params(labelsize=9)
    _save(fig, OUTDIR / "Fig3_persistence_colorbar")
    plt.close(fig)
    print("Wrote Fig3_persistence_colorbar.png/.pdf")


def main() -> None:
    print("Building Fig. 3 persistence maps from lag_analysis_FINAL ...")
    plot_grid_3x3()
    for ri, (ssp_label, scen_tag) in enumerate(SCENARIOS):
        plot_row(
            ssp_label=ssp_label,
            scen_tag=scen_tag,
            row_index=ri,
            attach_cbar=(ssp_label == "SSP585"),
        )
    plot_historical_row()
    plot_standalone_panels()
    plot_colorbar_standalone()
    print("Done:", OUTDIR)


if __name__ == "__main__":
    main()
