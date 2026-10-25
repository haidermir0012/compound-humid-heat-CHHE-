#!/usr/bin/env python3
"""
Supplement for Fig. 2: Model agreement on ΔHH (RD > 0).

For each SSP (126/245/585) and epoch (NF/MC/EC), compute RD per model:
  RD_m(x) = HH_m,future(x) - HH_m,historical(x)
where HH is mean annual HH-day frequency (days/yr) for TW>35°C & VPD>25 hPa.

Agreement fraction:
  A(x) = (1/M) * sum_m 1[ RD_m(x) > 0 ], M=3 models
Values are {0, 1/3, 2/3, 1}.

Outputs:
  Manuscript final plots/figures/Fig2_model_agreement_RD/
    - FigS_modelAgreement_RD_SSP126.png
    - FigS_modelAgreement_RD_SSP245.png
    - FigS_modelAgreement_RD_SSP585.png  (row includes colorbar)
    - panels/FigS_agree_{letter}_{SSP}_{epoch}.png  (9 panels per SSP, transparent optional)

Cache (skip recompute if present):
  Manuscript final plots/cache_model_agreement/agree_{ssp}_{period}.nc

Note: Reads raw TW/VPD daily files per model (slow first run).
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
from tqdm import tqdm

_HELPERS = Path(__file__).resolve().parents[1] / "helpers"
if str(_HELPERS) not in sys.path:
    sys.path.insert(0, str(_HELPERS))
from manuscript_basemap import format_ax_light

PROJECT_DIR = Path("/intelnvme04/haider/project/compound_humid_heat")
RAW = Path("/intelnvme04/haider/raw_data")

HIST_DIR = RAW / "nexgddp_historical_1981_2014" / "TW_VPD_LANDONLY_SAFE"
SSP_DIRS = {
    "SSP126": RAW / "nexgddp_ssp126_2015_2100" / "TW_VPD_LANDONLY_SAFE",
    "SSP245": RAW / "nexgddp_ssp245_2015_2100" / "TW_VPD_LANDONLY_SAFE",
    "SSP585": RAW / "nexgddp_ssp585_2015_2100" / "TW_VPD_LANDONLY_SAFE",
}

OUTDIR = PROJECT_DIR / "Manuscript final plots" / "figures" / "Fig2_model_agreement_RD"
PANEL_DIR = OUTDIR / "panels"
CACHE_DIR = PROJECT_DIR / "Manuscript final plots" / "cache_model_agreement"
OUTDIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PNG_ONLY = True
PNG_DPI = 600

LETTERS_BY_SSP = {
    "SSP126": ["a", "b", "c"],
    "SSP245": ["d", "e", "f"],
    "SSP585": ["g", "h", "i"],
}

MODELS = ["EC_Earth3", "HadGEM3_GC31_LL", "MPI-ESM1-2-HR"]
# NEX filenames vary by scenario (e.g. SSP245 uses EC-Earth3 with a hyphen).
MODEL_FILE_ALIASES: dict[str, list[str]] = {
    "EC_Earth3": ["EC_Earth3", "EC-Earth3"],
    "HadGEM3_GC31_LL": ["HadGEM3_GC31_LL"],
    "MPI-ESM1-2-HR": ["MPI-ESM1-2-HR"],
}
THRESH_TW = 35.0
THRESH_VPD = 25.0

PERIODS = [("NF", 2021, 2040, "nf"), ("MC", 2041, 2060, "mc"), ("EC", 2081, 2100, "ec")]
HIST = (1981, 2014)

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

land_mask = xr.open_dataarray(PROJECT_DIR / "land_mask.nc").load()


def format_ax(ax, *, show_y: bool, show_x: bool) -> None:
    format_ax_light(
        ax,
        proj=proj,
        extent=extent,
        lon_ticks=lon_ticks,
        lat_ticks=lat_ticks,
        show_y=show_y,
        show_x=show_x,
    )


def _open_first_match(base: Path, patterns: list[str]) -> xr.Dataset | None:
    for pattern in patterns:
        files = list(base.glob(pattern))
        if files:
            return xr.open_dataset(files[0])
    return None


def _index_files_by_year(base_dir: Path, model: str, tag: str) -> dict[int, Path]:
    """
    Build a year->file index once (much faster than globbing each year).
    Handles mixed-case tags for historical/SSPs.
    """
    if tag.lower() == "historical":
        tag_variants = ["Historical", "historical"]
    elif tag.lower().startswith("ssp"):
        tag_variants = [tag.lower(), tag.upper()]
    else:
        tag_variants = [tag]

    model_variants = MODEL_FILE_ALIASES.get(model, [model])
    files: list[Path] = []
    for mv in model_variants:
        for tv in tag_variants:
            files.extend(base_dir.glob(f"*{mv}*{tv}*TW_VPD_*.nc"))
            files.extend(base_dir.glob(f"*{mv}*{tv}*_{tv}_*TW_VPD_*.nc"))
            files.extend(base_dir.glob(f"*{mv}*{tv}*_{tv}_*_*TW_VPD_*.nc"))
            files.extend(base_dir.glob(f"*{mv}*{tv}*_*TW_VPD_*.nc"))
            # SSP245 EC-Earth3: EC-Earth3_ssp245_2015_2100_TW_VPD_YYYY.nc
            files.extend(base_dir.glob(f"*{mv}*{tv}*_*TW_VPD_*.nc"))

    out: dict[int, Path] = {}
    for f in files:
        # year is typically the last 4 digits before .nc
        try:
            y = int(f.stem.split("_")[-1])
        except Exception:
            continue
        if y not in out:
            out[y] = f
    return out


def epoch_mean_hh(model: str, *, start: int, end: int, base_dir: Path, tag: str) -> xr.DataArray:
    """
    Mean annual HH days/yr for one model over [start, end].
    We average the annual HH-day counts across years (days/yr).
    """
    accum = None
    n = 0
    idx = _index_files_by_year(base_dir, model, tag)
    for yr in tqdm(range(start, end + 1), desc=f"{model} {tag} {start}-{end}", leave=False):
        f = idx.get(yr)
        if f is None:
            continue
        ds = xr.open_dataset(f)
        try:
            tw = ds["tw"]
            if float(tw.max()) > 200:
                tw = tw - 273.15
            hh = (tw > THRESH_TW) & (ds["vpd"] > THRESH_VPD)
            days = hh.sum("time").astype("float32")
            accum = days if accum is None else (accum + days)
            n += 1
        finally:
            ds.close()
    if n == 0:
        raise RuntimeError(f"No files processed for model={model} tag={tag} {start}-{end}")
    mean = (accum / n).where(land_mask).astype("float32")
    return mean


def agreement_fraction(ssp: str, *, use_cache: bool = True) -> dict[str, xr.DataArray]:
    """Return {period: agreement_fraction_map} for NF/MC/EC for one SSP."""
    ssp_dir = SSP_DIRS[ssp]
    out: dict[str, xr.DataArray] = {}

    hist_by_model: dict[str, xr.DataArray] = {}
    for m in MODELS:
        hist_cache = CACHE_DIR / f"hh_mean_hist_{m}.nc"
        if use_cache and hist_cache.exists():
            hist_by_model[m] = xr.open_dataarray(hist_cache).load()
            print(f"  historical: {m} (cache)")
        else:
            print(f"  historical: {m} {HIST[0]}-{HIST[1]}")
            hist_by_model[m] = epoch_mean_hh(m, start=HIST[0], end=HIST[1], base_dir=HIST_DIR, tag="historical")
            hist_by_model[m].to_netcdf(hist_cache)

    for period, y0, y1, tag in PERIODS:
        cache_path = CACHE_DIR / f"agree_{ssp}_{period}.nc"
        if use_cache and cache_path.exists():
            print(f"  {period}: load cache {cache_path.name}")
            out[period] = xr.open_dataarray(cache_path).load()
            continue

        print(f"  {period}: {y0}-{y1} (compute)")
        votes = []
        for m in MODELS:
            print(f"    future: {m} ({ssp.lower()})")
            fut = epoch_mean_hh(m, start=y0, end=y1, base_dir=ssp_dir, tag=ssp.lower())
            rd = fut - hist_by_model[m]
            votes.append((rd > 0).astype("int8"))
        vote_sum = xr.concat(votes, dim="model").sum("model").astype("float32")
        frac = (vote_sum / float(len(MODELS))).where(land_mask)
        frac.name = f"agree_RDpos_{ssp}_{period}"
        frac.to_netcdf(cache_path)
        print(f"    cached: {cache_path}")
        out[period] = frac
    return out


def plot_row(ssp: str, agree: dict[str, xr.DataArray], *, letters: list[str], attach_cbar: bool) -> None:
    cmap, norm, levels = _agreement_cmap()

    if attach_cbar:
        fig = plt.figure(figsize=(10.2, 2.45))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.10], hspace=0.10, wspace=0.03)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax = fig.add_subplot(gs[1, :])
    else:
        fig = plt.figure(figsize=(10.2, 2.05))
        gs = fig.add_gridspec(1, 3, wspace=0.03)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax = None

    im = None
    for i, (ax, period, letter) in enumerate(zip(axes, ["NF", "MC", "EC"], letters)):
        da = agree[period]
        im = ax.pcolormesh(
            da["lon"].values,
            da["lat"].values,
            da.values,
            transform=proj,
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=True,
            zorder=2,
        )
        format_ax(ax, show_y=(i == 0), show_x=True)
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

    if attach_cbar:
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Model agreement on ΔHH > 0 (fraction of 3 models)", fontsize=10)
        cb.set_ticks(levels)
        cb.set_ticklabels(["0", "1/3", "2/3", "1"])
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    stem = OUTDIR / f"FigS_modelAgreement_RD_{ssp}"
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
    if not SAVE_PNG_ONLY:
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _agreement_cmap():
    levels = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype="float32")
    cmap = mpl.colors.ListedColormap(["#f7f7f7", "#c6dbef", "#6baed6", "#08519c"])
    norm = mpl.colors.BoundaryNorm(boundaries=[-0.01, 1 / 6, 0.5, 5 / 6, 1.01], ncolors=cmap.N)
    return cmap, norm, levels


def plot_individual_panels(
    ssp: str, agree: dict[str, xr.DataArray], *, letters: list[str], force_replot: bool = False
) -> None:
    cmap, norm, _ = _agreement_cmap()
    for (period, *_rest), letter in zip(PERIODS, letters):
        da = agree[period]
        fig = plt.figure(figsize=(3.4, 1.85))
        ax = plt.axes(projection=proj)
        fig.patch.set_facecolor("white")
        ax.pcolormesh(
            da["lon"].values,
            da["lat"].values,
            da.values,
            transform=proj,
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=True,
            zorder=2,
        )
        format_ax(ax, show_y=True, show_x=True)
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
        stem = PANEL_DIR / f"FigS_agree_{letter}_{ssp}_{period}"
        if not force_replot and stem.with_suffix(".png").exists():
            print(f"    panel exists, skip: {stem.name}.png")
            plt.close(fig)
            continue
        fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
        plt.close(fig)
        print(f"    panel: {stem.name}.png")


def main() -> None:
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--force-replot", action="store_true", help="Overwrite existing row/panel PNGs.")
    args = ap.parse_args()

    print("Computing model agreement on ΔHH > 0 (all SSPs)...")
    cmap, norm, levels = _agreement_cmap()

    ssps = ["SSP126", "SSP245", "SSP585"]
    start_from = os.environ.get("RESUME_FROM_SSP", "").strip()
    if start_from:
        if start_from not in ssps:
            raise ValueError(f"RESUME_FROM_SSP must be one of {ssps}, got {start_from!r}")
        ssps = ssps[ssps.index(start_from) :]
        print(f"Resume from {start_from} (skip earlier SSPs)")

    for ssp in ssps:
        print(f"\n=== {ssp} ===")
        agree = agreement_fraction(ssp, use_cache=True)
        attach = ssp == "SSP585"
        row_png = OUTDIR / f"FigS_modelAgreement_RD_{ssp}.png"
        if args.force_replot or not row_png.exists():
            plot_row(ssp, agree, letters=LETTERS_BY_SSP[ssp], attach_cbar=attach)
            print(f"  row: {row_png.name}")
        else:
            print(f"  row exists, skip: {row_png.name}")
        plot_individual_panels(
            ssp, agree, letters=LETTERS_BY_SSP[ssp], force_replot=args.force_replot
        )

    # Standalone colorbar
    fig = plt.figure(figsize=(4.3, 0.55))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.07, 0.55, 0.90, 0.25])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label("Model agreement on ΔHH > 0 (fraction of 3 models)", fontsize=10)
    cb.set_ticks(levels)
    cb.set_ticklabels(["0", "1/3", "2/3", "1"])
    cb.ax.tick_params(labelsize=9)
    stem = OUTDIR / "FigS_modelAgreement_RD_colorbar"
    if stem.with_suffix(".png").exists():
        print(f"\nColorbar exists, skip: {stem.name}.png")
    else:
        fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
        print(f"\nWrote colorbar: {stem.name}.png")
    plt.close(fig)
    print("Done. Outputs:", OUTDIR)


if __name__ == "__main__":
    main()

