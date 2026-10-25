#!/usr/bin/env python3
"""
Fig. 2 RD rows (FINAL style): 1×3 panels (NF/MC/EC) per SSP for manual assembly.

Goal: match the clean multi-row style example (light-grey land, minimal clutter).

Writes:
  Manuscript final plots/figures/Fig2_RD_rows_style/
    - Fig2_RDrow_SSP126.png/.pdf
    - Fig2_RDrow_SSP245.png/.pdf
    - Fig2_RDrow_SSP585.png/.pdf
    - Fig2_RD_colorbar_horizontal.png/.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

_HELPERS = Path(__file__).resolve().parents[1] / "helpers"
if str(_HELPERS) not in sys.path:
    sys.path.insert(0, str(_HELPERS))
from manuscript_basemap import format_ax_light
from rd_colormap import RD_NEUTRAL_HI, RD_NEUTRAL_LO, RD_VLIM, rd_hh_cmap_norm

PROJECT_DIR = Path("/intelnvme04/haider/project/compound_humid_heat")
OUTDIR = PROJECT_DIR / "Manuscript final plots" / "figures" / "Fig2_RD_rows_style"
OUTDIR.mkdir(parents=True, exist_ok=True)

SSP_FOLDERS = {
    "SSP126": PROJECT_DIR / "computed_ssp126",
    "SSP245": PROJECT_DIR / "computed_ssp245",
    "SSP585": PROJECT_DIR / "computed_ssp585",
}

PERIODS = [("NF", "nf"), ("MC", "mc"), ("EC", "ec")]

proj = ccrs.PlateCarree()
extent = [-130, 155, -55, 55]
lon_ticks = [-70, -10, 50, 110]
lat_ticks = [35, 5, -25, -55]

# Fig1-like baseline typography; keep light and consistent.
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


def _load(folder: Path, scen: str, tag: str) -> xr.DataArray:
    p = folder / f"{scen}_{tag}_HH_mean.nc"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return xr.open_dataarray(p)


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


def plot_row(
    *,
    scen: str,
    folder: Path,
    vlim: float,
    letters: list[str],
    show_epoch_titles: bool,
    attach_cbar: bool,
    cbar_orientation: str = "vertical",
    conical_cbar: bool = False,
) -> None:
    hist = _load(folder, scen.lower(), "historical").load()
    rd_maps = []
    for _, tag in PERIODS:
        fut = _load(folder, scen.lower(), tag).load()
        rd_maps.append((fut - hist))

    if attach_cbar and cbar_orientation not in {"vertical", "horizontal"}:
        raise ValueError("cbar_orientation must be 'vertical' or 'horizontal'")

    # Horizontal colorbar beneath panels; conical ends (extend="both") for SSP585 continuity.
    if attach_cbar and cbar_orientation == "horizontal":
        fig_h = 2.32 if conical_cbar else 2.45
        cbar_ratio = 0.055 if conical_cbar else 0.10
        cbar_hspace = 0.02 if conical_cbar else 0.10
        fig = plt.figure(figsize=(10.2, fig_h))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, cbar_ratio], hspace=cbar_hspace, wspace=0.03)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax = fig.add_subplot(gs[1, :])
    else:
        fig_w = 10.6 if attach_cbar else 10.2
        fig = plt.figure(figsize=(fig_w, 2.05))
        if attach_cbar:
            gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.03)
            axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
            cax = fig.add_subplot(gs[0, 3])
        else:
            gs = fig.add_gridspec(1, 3, wspace=0.03)
            axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
            cax = None

    cmap, norm = rd_hh_cmap_norm(vlim=vlim)
    im = None
    for i, (ax, (p_short, _), rd, letter) in enumerate(zip(axes, PERIODS, rd_maps, letters)):
        im = ax.pcolormesh(
            rd["lon"].values,
            rd["lat"].values,
            rd.values,
            transform=proj,
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=True,
            zorder=2,
        )
        format_ax(ax, show_y=(i == 0), show_x=True)

        # Panel letter (a..i) in the top-left, outside the map content
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

        # Optional epoch titles (disabled per current request)
        if show_epoch_titles:
            ax.set_title(p_short, fontsize=11, fontweight="bold", pad=6)

    if attach_cbar:
        cb = fig.colorbar(
            im,
            cax=cax,
            orientation=cbar_orientation,
            extend="both" if conical_cbar else "neither",
            extendrect=False,
        )
        cb.set_label("ΔHH (days yr$^{-1}$)", fontsize=10, labelpad=2 if conical_cbar else None)
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2 if conical_cbar else None)

    stem = OUTDIR / f"Fig2_RDrow_{scen}"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)


def plot_colorbar(*, vlim: float) -> None:
    fig = plt.figure(figsize=(4.3, 0.55))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.07, 0.55, 0.90, 0.25])
    cmap, norm = rd_hh_cmap_norm(vlim=vlim)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_label("ΔHH (days yr$^{-1}$)", fontsize=10)
    cb.ax.tick_params(labelsize=9, width=0.6, length=2.5)
    stem = OUTDIR / "Fig2_RD_colorbar_horizontal"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)


def main() -> None:
    vlim = RD_VLIM
    print(
        f"RD color scale (fixed symmetric): ±{vlim:.1f} days/yr; "
        f"grey neutral band [{RD_NEUTRAL_LO:.0f}, {RD_NEUTRAL_HI:.0f}]"
    )

    plot_row(
        scen="SSP126",
        folder=SSP_FOLDERS["SSP126"],
        vlim=vlim,
        letters=["a", "b", "c"],
        show_epoch_titles=False,
        attach_cbar=False,
    )
    print("Wrote row: SSP126")

    plot_row(
        scen="SSP245",
        folder=SSP_FOLDERS["SSP245"],
        vlim=vlim,
        letters=["d", "e", "f"],
        show_epoch_titles=False,
        attach_cbar=False,
    )
    print("Wrote row: SSP245")

    plot_row(
        scen="SSP585",
        folder=SSP_FOLDERS["SSP585"],
        vlim=vlim,
        letters=["g", "h", "i"],
        show_epoch_titles=False,
        attach_cbar=True,
        cbar_orientation="horizontal",
        conical_cbar=True,
    )
    print("Wrote row: SSP585 (conical colorbar)")

    # No standalone colorbar (requested)
    print("Done.")


if __name__ == "__main__":
    main()

