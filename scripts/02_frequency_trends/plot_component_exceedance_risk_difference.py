#!/usr/bin/env python3
"""
Plot mechanism-lite supplement: Δdays(TW>35) and Δdays(VPD>25) as RD rows (NF/MC/EC) per SSP.

Matches Fig. 2 RD row conventions (see plot_fig2_risk_difference.py):
  - SSP126 panels a–c, SSP245 d–f, SSP585 g–i (no NF/MC/EC titles on panels)
  - SSP126/SSP245 rows: maps only (no colorbar) for manual merge
  - SSP585 row: thin horizontal conical-ended colorbar close under lon ticks
  - Separate standalone colorbars per metric for assembly

Reads:
  Manuscript final plots/computed_component_exceedance/<ssp>/

Writes:
  Manuscript final plots/figures/Fig2_components_RD_rows_style/
    FigS_TW35_RDrow_SSP126.png/.pdf  (etc.)
    FigS_VPD25_RDrow_SSP585_conicalCbar.png/.pdf
    FigS_TW35_RD_colorbar_horizontal.png/.pdf
    FigS_VPD25_RD_colorbar_horizontal.png/.pdf
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

PROJECT_DIR = Path("/intelnvme04/haider/project/compound_humid_heat")
IN_BASE = PROJECT_DIR / "Manuscript final plots" / "computed_component_exceedance"
OUTDIR = PROJECT_DIR / "Manuscript final plots" / "figures" / "Fig2_components_RD_rows_style"
OUTDIR.mkdir(parents=True, exist_ok=True)

SSPS = ["ssp126", "ssp245", "ssp585"]
PERIOD_TAGS = ["nf", "mc", "ec"]

# Panel letters across the full supplement figure (same as Fig. 2 HH RD)
LETTERS_BY_SSP = {
    "ssp126": ["a", "b", "c"],
    "ssp245": ["d", "e", "f"],
    "ssp585": ["g", "h", "i"],
}

COMPONENT_LABEL = {
    "tw35": r"$\Delta$days (TW $>$ 35 $^\circ$C) (days yr$^{-1}$)",
    "vpd25": r"$\Delta$days (VPD $>$ 25 hPa) (days yr$^{-1}$)",
}

proj = ccrs.PlateCarree()
extent = [-130, 155, -55, 55]
lon_ticks = [-70, -10, 50, 110]
lat_ticks = [35, 5, -25, -55]

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


def _load(ssp: str, epoch: str, component: str) -> xr.DataArray:
    p = IN_BASE / ssp / f"{ssp}_{epoch}_{component}_days_mean.nc"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return xr.open_dataarray(p)


def format_ax(ax, *, show_y: bool) -> None:
    format_ax_light(
        ax,
        proj=proj,
        extent=extent,
        lon_ticks=lon_ticks,
        lat_ticks=lat_ticks,
        show_y=show_y,
        show_x=True,
    )


def rd_scale_98(component: str) -> float:
    vals = []
    for ssp in SSPS:
        hist = _load(ssp, "historical", component)
        for tag in PERIOD_TAGS:
            fut = _load(ssp, tag, component)
            vals.append((fut - hist).values.ravel())
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    return float(np.nanpercentile(np.abs(v), 98))


def plot_row(
    *,
    ssp: str,
    component: str,
    vlim: float,
    letters: list[str],
    attach_cbar: bool,
    conical_cbar: bool,
) -> Path:
    comp_tag = component.upper()
    scen_tag = ssp.upper()
    hist = _load(ssp, "historical", component).load()

    if attach_cbar and conical_cbar:
        fig = plt.figure(figsize=(10.2, 2.32))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.055], hspace=0.02, wspace=0.03)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax = fig.add_subplot(gs[1, :])
    else:
        fig = plt.figure(figsize=(10.2, 2.05))
        gs = fig.add_gridspec(1, 3, wspace=0.03)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax = None

    im = None
    for i, (tag, letter) in enumerate(zip(PERIOD_TAGS, letters)):
        fut = _load(ssp, tag, component).load()
        rd = fut - hist
        ax = axes[i]
        im = ax.pcolormesh(
            rd["lon"].values,
            rd["lat"].values,
            rd.values,
            transform=proj,
            cmap=plt.cm.RdYlBu_r,
            vmin=-vlim,
            vmax=vlim,
            shading="auto",
            rasterized=True,
            zorder=2,
        )
        format_ax(ax, show_y=(i == 0))
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

    if attach_cbar and cax is not None and im is not None:
        cb = fig.colorbar(
            im,
            cax=cax,
            orientation="horizontal",
            extend="both" if conical_cbar else "neither",
            extendrect=False,
        )
        cb.set_label(COMPONENT_LABEL[component], fontsize=10, labelpad=2)
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)
        stem = OUTDIR / f"FigS_{comp_tag}_RDrow_{scen_tag}_conicalCbar"
    else:
        stem = OUTDIR / f"FigS_{comp_tag}_RDrow_{scen_tag}"

    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    return stem


def plot_colorbar(*, component: str, vlim: float) -> None:
    fig = plt.figure(figsize=(4.3, 0.55))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.07, 0.55, 0.90, 0.25])
    norm = mpl.colors.Normalize(vmin=-vlim, vmax=vlim)
    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=plt.cm.RdYlBu_r,
        norm=norm,
        orientation="horizontal",
        extend="both",
        extendrect=False,
    )
    cb.set_label(COMPONENT_LABEL[component], fontsize=10)
    cb.ax.tick_params(labelsize=9, width=0.6, length=2.5)
    stem = OUTDIR / f"FigS_{component.upper()}_RD_colorbar_horizontal"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)


def main() -> None:
    scales = {comp: rd_scale_98(comp) for comp in ("tw35", "vpd25")}
    for comp, vlim in scales.items():
        print(f"{comp.upper()} RD scale (98th pct |Δ|, pooled): ±{vlim:.3f} days/yr")

    for component in ("tw35", "vpd25"):
        vlim = scales[component]
        for ssp in SSPS:
            letters = LETTERS_BY_SSP[ssp]
            attach = ssp == "ssp585"
            stem = plot_row(
                ssp=ssp,
                component=component,
                vlim=vlim,
                letters=letters,
                attach_cbar=attach,
                conical_cbar=attach,
            )
            print(f"Wrote {stem.name}.png/.pdf")

        plot_colorbar(component=component, vlim=vlim)
        print(f"Wrote FigS_{component.upper()}_RD_colorbar_horizontal.png/.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
