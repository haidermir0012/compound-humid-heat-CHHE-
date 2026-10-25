#!/usr/bin/env python3
"""
Figure 4 (FINAL): Post-HH successor probabilities with dry-spell precondition.

Reads ensemble cache from script 16:
  cache_postHH_precondition_FINAL/postHH_precond_{scenario}_win{W}_dur{k}_*.nc

Writes (PNG 600 dpi + PDF unless noted):
  Manuscript final plots/figures/Fig4_postHH/
    Fig4_postHH_{scenario}_4x3.png/.pdf     — full plate (4 categories × 3 map types)
    Fig4_postHH_{scenario}_row_{category}.png/.pdf     — 1×3 rows for manual tweaks
    Fig4_postHH_{scenario}_row_dry_only_cbar.png/.pdf  — last row with dual colorbars
    Fig4_postHH_{scenario}_prob_colorbar.png/.pdf
    Fig4_postHH_{scenario}_diff_colorbar.png/.pdf
    panels/{scenario}/Fig4_{letter}_{category}_{kind}.png

Use --main-only to skip per-row exports and standalone colorbars (keeps 4×3 + panels/).
Use --rows-only to wipe prior PNG/PDF outputs and write per-category 1×3 row panels only
(ensemble mean; last row includes dual colorbars). Optionally includes 4×3 when layout is compact.
"""

from __future__ import annotations

import argparse
import string
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
from rd_colormap import rd_hh_cmap_norm

PROJECT_DIR = Path("/intelnvme04/haider/project/compound_humid_heat")
CACHE_DIR = PROJECT_DIR / "cache_postHH_precondition_FINAL"
OUTDIR = PROJECT_DIR / "Manuscript final plots" / "figures" / "Fig4_postHH"
PANEL_DIR = OUTDIR / "panels"

# Match Fig. 2 / Fig. 3 manuscript layout
SCENARIO = "historical"
WINDOW_DAYS = 7
DURATION = 3
PRE_TAG = "dry_spell_L14"

CATEGORIES = ["compound_again", "non_extreme", "hot_only", "dry_only"]
MAIN_TEXT_CATEGORIES = ["compound_again", "hot_only", "dry_only"]
ROW_LABELS = {
    "compound_again": "Compound again",
    "hot_only": "Hot only",
    "dry_only": "Dry only",
}
COL_KINDS = ["baseline", "precond", "diff"]  # maps to dataset vars
PRECURSOR_COL_KINDS = ["baseline", "precond"]
PRECURSOR_COL_TITLES = ["Baseline", "Preconditioned"]
PUBLICATION_PRECURSORS_SUBDIR = "PUBLICATION_Fig4a_precursors_3x2"
PUBLICATION_SSP_COMPARE_SUBDIR = "PUBLICATION_Fig4b_ssp_compare_DELTA_3x3"
# Verified epoch-stratified publication bundle (NF/MC/EC plates from epoch caches).
PUBLICATION_EPOCH_ROOT = "PUBLICATION_Fig4_EPOCH_STRATIFIED"
PUBLICATION_EPOCH_PRECURSORS_SUBDIR = f"{PUBLICATION_EPOCH_ROOT}/Fig4a_precursors_3x2"
PUBLICATION_EPOCH_SSP_COMPARE_SUBDIR = f"{PUBLICATION_EPOCH_ROOT}/Fig4b_ssp_compare_DELTA_3x3"
# Legacy export (baseline | precond | ΔP for one SSP); use --publication-precursors instead.
PUBLICATION_SUBDIR = "PUBLICATION_ssp585_MAIN_3x3"
SSP_COMPARE = [
    ("SSP126", "ssp126"),
    ("SSP245", "ssp245"),
    ("SSP585", "ssp585"),
]
# Match compute_post_hh_epoch_ensemble.py / Fig. 5 epoch windows
EPOCHS: list[tuple[str, int, int]] = [
    ("NF", 2021, 2040),
    ("MC", 2041, 2060),
    ("EC", 2081, 2100),
]
EPOCH_LABELS = {
    "NF": "NF (2021–2040)",
    "MC": "MC (2041–2060)",
    "EC": "EC (2081–2100)",
}
EPOCH_CACHE_TAG = "epochsNF-MC-EC"
EPOCH_FILE_TAGS = ("_NF_", "_MC_", "_EC_")

proj = ccrs.PlateCarree()
extent = [-130, 155, -55, 55]
lon_ticks = [-70, -10, 50, 110]
lat_ticks = [35, 5, -25, -55]

PNG_DPI = 600
PROB_VMIN, PROB_VMAX = 0.0, 1.0
DIFF_VMAX = 0.35
# ΔP neutral band (probability units): suppress weak ±3 pp noise like Fig. 2 ±3 d yr⁻¹.
DIFF_NEUTRAL_LO, DIFF_NEUTRAL_HI = -0.03, 0.03
# One shared sequential cmap for cols 1–2 (all categories); matches Fig. 3 / single prob colorbar.
PROB_CMAP = plt.cm.inferno

# Fig. 3 row geometry (10.2 in wide; 2.05 in per labelled map row; 2.38 in with colorbar).
ROW_FIG_W = 10.2
ROW_FIG_H = 2.05
ROW_FIG_H_CBAR = 2.38
CBAR_ROW_RATIO = 0.055
MAP_ROW_RATIO = 1.0
GS_HSPACE_ROW_CBAR = 0.02  # map ↔ colorbar gap (Fig. 3 style)
GS_HSPACE_4X3 = 0.0  # inter-row gap for stacked 3×3 / 4×3 plates
GS_WSPACE = 0.03
MAP_ROW_GAP_FRAC = 0.032  # uniform vertical gap between stacked map rows
CBAR_GAP_FRAC = 0.030
CBAR_HEIGHT_FRAC = 0.025
PANEL_LETTER_TOP_PAD = 0.011  # above top row when no row above
# Compact 4×3 enabled in --rows-only after spacing aligned with Fig. 3.
ROWS_ONLY_INCLUDE_4X3 = True

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


def find_cache(
    scenario: str,
    *,
    window_days: int,
    duration: int,
    pre_tag: str,
    min_models: int = 3,
    epoch_cache: bool = False,
) -> Path:
    if epoch_cache:
        patt = (
            f"postHH_precond_{scenario}_win{window_days}_{EPOCH_CACHE_TAG}_"
            f"dur{duration}_*pre{pre_tag}_*.nc"
        )
    else:
        patt = f"postHH_precond_{scenario}_win{window_days}_dur{duration}_*pre{pre_tag}_*.nc"
    cands = sorted(CACHE_DIR.glob(patt), key=lambda p: p.stat().st_mtime, reverse=True)
    if not epoch_cache:
        cands = [p for p in cands if EPOCH_CACHE_TAG not in p.name]
    if not cands:
        raise FileNotFoundError(f"No cache matching {CACHE_DIR}/{patt}")

    def n_models_in(path: Path) -> int:
        try:
            with xr.open_dataset(path) as ds:
                return int(ds.attrs.get("n_models", 0))
        except Exception:
            return 0

    # Prefer latest cache with full ensemble (core figure requires 3 models).
    for fp in cands:
        if n_models_in(fp) >= min_models:
            return fp

    best = max(cands, key=n_models_in)
    nm = n_models_in(best)
    if nm < min_models:
        raise FileNotFoundError(
            f"No {min_models}-model cache found (best has n_models={nm}: {best.name}). "
            "Run compute_post_hh_ssp_ensemble.py with all three models first."
        )
    return best


def load_fields(cache_fp: Path) -> xr.Dataset:
    ds = xr.open_dataset(cache_fp, engine="netcdf4")
    for v in ("P_baseline", "P_preconditioned", "P_diff"):
        if v not in ds:
            ds.close()
            raise KeyError(f"{v} missing in {cache_fp}")
    return ds


def get_da(
    ds: xr.Dataset,
    *,
    kind: str,
    category: str,
    duration: int,
    epoch: str | None = None,
) -> xr.DataArray:
    if kind == "baseline":
        da = ds["P_baseline"]
    elif kind == "precond":
        da = ds["P_preconditioned"]
    else:
        da = ds["P_diff"]
    has_epoch = "epoch" in da.dims
    if epoch is not None and not has_epoch:
        raise ValueError(
            f"epoch={epoch!r} requested but cache lacks epoch dimension "
            f"(full-period cache loaded? source={getattr(ds, 'encoding', {}).get('source', '?')})"
        )
    if has_epoch:
        if epoch is None:
            raise ValueError("epoch required when loading from epoch-stratified cache")
        out = da.sel(epoch=epoch, duration=duration, category=category).load()
    else:
        if epoch is not None:
            raise ValueError(f"epoch={epoch!r} requested but variable has no epoch dimension")
        out = da.sel(duration=duration, category=category).load()
    out = out.squeeze(drop=True)
    if set(out.dims) != {"lat", "lon"}:
        raise ValueError(f"expected 2D lat/lon field, got dims={out.dims}")
    return out.where(land_mask == 1)


def _land_mean(da: xr.DataArray) -> float:
    return float(da.where(land_mask == 1).mean(skipna=True).values)


def assert_epoch_stratified_datasets(ds_by_ssp: dict[str, xr.Dataset], *, epochs: list[str]) -> None:
    """Fail fast when epoch_cache=True but loaded files are not epoch-stratified."""
    for ssp_tag, ds in ds_by_ssp.items():
        da = ds["P_diff"]
        if "epoch" not in da.dims:
            raise RuntimeError(
                f"{ssp_tag}: epoch-stratified cache required but P_diff lacks epoch dimension "
                f"(attrs={dict(ds.attrs)})"
            )
        found = {str(v) for v in da.coords["epoch"].values}
        missing = [ep for ep in epochs if ep not in found]
        if missing:
            raise RuntimeError(f"{ssp_tag}: cache missing epoch coordinate(s) {missing} (have {sorted(found)})")


def print_epoch_sanity_check(
    ds_by_ssp: dict[str, xr.Dataset],
    *,
    epochs: list[str],
    full_period_ds_by_ssp: dict[str, xr.Dataset] | None = None,
) -> None:
    """Log global mean ΔP(dry_only) per epoch per SSP; verify NF≠MC≠EC."""
    print("\n=== Epoch-stratification sanity check (global land-mean ΔP, dry_only) ===")
    for _label, ssp_tag in SSP_COMPARE:
        ds = ds_by_ssp[ssp_tag]
        means: dict[str, float] = {}
        for ep in epochs:
            da = get_da(ds, kind="diff", category="dry_only", duration=DURATION, epoch=ep)
            means[ep] = _land_mean(da)
            print(f"  {ssp_tag} {ep}: {means[ep]:+.4f}")
        if len(set(round(v, 6) for v in means.values())) < len(epochs):
            raise RuntimeError(
                f"{ssp_tag}: epoch panels would be identical — "
                f"dry_only ΔP land-means {means} (expected distinct NF/MC/EC values)"
            )
        if full_period_ds_by_ssp is not None:
            da_full = get_da(
                full_period_ds_by_ssp[ssp_tag],
                kind="diff",
                category="dry_only",
                duration=DURATION,
                epoch=None,
            )
            full_mean = _land_mean(da_full)
            print(f"  {ssp_tag} FULL (reference): {full_mean:+.4f}")
            for ep in epochs:
                if abs(means[ep] - full_mean) < 1e-6:
                    raise RuntimeError(
                        f"{ssp_tag} {ep}: land-mean ΔP matches full-period ({full_mean:+.4f}) — "
                        "epoch selection is not taking effect"
                    )
    print("=== Sanity check passed: epochs differ and differ from full-period ===\n")


def publication_subdirs(*, epoch_cache: bool) -> tuple[str, str]:
    """Return (precursors_subdir, ssp_compare_subdir) under OUTDIR."""
    if epoch_cache:
        return PUBLICATION_EPOCH_PRECURSORS_SUBDIR, PUBLICATION_EPOCH_SSP_COMPARE_SUBDIR
    return PUBLICATION_PRECURSORS_SUBDIR, PUBLICATION_SSP_COMPARE_SUBDIR


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
    # Strip reserved x-tick band on internal rows (major source of inter-row white gap).
    if not show_x:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False, pad=0)
        ax.xaxis.set_visible(False)


def diff_cmap_norm(diff_vmax: float) -> tuple[mpl.colors.LinearSegmentedColormap, mpl.colors.Normalize]:
    """Blue (decrease) | light grey neutral | red (increase) for ΔP panels."""
    return rd_hh_cmap_norm(
        vlim=diff_vmax,
        neutral_lo=DIFF_NEUTRAL_LO,
        neutral_hi=DIFF_NEUTRAL_HI,
    )


def plot_on_ax(
    ax,
    da: xr.DataArray,
    *,
    kind: str,
    category: str,
    diff_vmax: float,
) -> mpl.cm.ScalarMappable:
    if kind == "diff":
        cmap, norm = diff_cmap_norm(diff_vmax)
        return ax.pcolormesh(
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
    return ax.pcolormesh(
        da["lon"].values,
        da["lat"].values,
        da.values,
        transform=proj,
        cmap=PROB_CMAP,
        vmin=PROB_VMIN,
        vmax=PROB_VMAX,
        shading="auto",
        rasterized=True,
        zorder=2,
    )


def add_panel_letter(ax, letter: str) -> None:
    """Single-row / standalone panels: letter just above axis top."""
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


def _place_panel_letters_in_row_gaps(
    fig: plt.Figure,
    *,
    n_map_rows: int,
    ncols: int,
    letters: list[str],
) -> None:
    """Place panel letters in whitespace above each map row (after row stacking)."""
    fig.canvas.draw()
    map_axes = fig.axes[: n_map_rows * ncols]

    def row_y_bounds(ri: int) -> tuple[float, float]:
        ps = [map_axes[ri * ncols + c].get_position() for c in range(ncols)]
        return min(p.y0 for p in ps), max(p.y1 for p in ps)

    for idx, ax in enumerate(map_axes):
        ri = idx // ncols
        x_fig = ax.get_position().x0
        if ri == 0:
            _, y1_top = row_y_bounds(0)
            y_fig = y1_top + PANEL_LETTER_TOP_PAD
            valign = "bottom"
        else:
            y0_above, _ = row_y_bounds(ri - 1)
            _, y1_curr = row_y_bounds(ri)
            y_fig = 0.5 * (y1_curr + y0_above)
            valign = "center"
        fig.text(
            x_fig,
            y_fig,
            letters[idx],
            ha="left",
            va=valign,
            fontsize=11,
            fontweight="bold",
            color="0.10",
        )


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")


def cleanup_fig4_outputs() -> list[str]:
    """Delete PNG/PDF under Fig4_postHH (root + panels/); keep README.md."""
    deleted: list[str] = []
    for pat in ("*.png", "*.pdf"):
        for fp in OUTDIR.glob(pat):
            fp.unlink()
            deleted.append(str(fp))
    if PANEL_DIR.exists():
        for pat in ("*.png", "*.pdf"):
            for fp in PANEL_DIR.rglob(pat):
                fp.unlink()
                deleted.append(str(fp))
    return deleted


def _is_epoch_publication_file(path: Path) -> bool:
    """True when filename marks NF, MC, or EC epoch stratification."""
    return any(tag in path.name for tag in EPOCH_FILE_TAGS)


def cleanup_legacy_publication_outputs(pub_dir: Path) -> list[str]:
    """
    Remove full-period publication exports (no _NF_/_MC_/_EC_ in name).

    Keeps README.md and epoch-stratified PNG/PDF/snippet files.
    """
    deleted: list[str] = []
    if not pub_dir.exists():
        return deleted
    for fp in pub_dir.iterdir():
        if fp.name == "README.md":
            continue
        if fp.suffix.lower() not in {".png", ".pdf", ".md"}:
            continue
        if _is_epoch_publication_file(fp):
            continue
        fp.unlink()
        deleted.append(str(fp))
    return deleted


def epoch_caches_available(
    *,
    window_days: int,
    duration: int,
    pre_tag: str,
    min_models: int,
) -> bool:
    """True when all three SSP epoch-stratified ensemble caches exist."""
    for _label, tag in SSP_COMPARE:
        try:
            find_cache(
                tag,
                window_days=window_days,
                duration=duration,
                pre_tag=pre_tag,
                min_models=min_models,
                epoch_cache=True,
            )
        except FileNotFoundError:
            return False
    return True


def _grid_height_ratios(n_map_rows: int) -> list[float]:
    """Equal map-row slots; internal rows hide x-axis instead of shrinking height_ratios."""
    if n_map_rows <= 1:
        return [MAP_ROW_RATIO, CBAR_ROW_RATIO]
    return [MAP_ROW_RATIO] * n_map_rows + [CBAR_ROW_RATIO]


def _composite_fig_height(n_map_rows: int) -> float:
    """Scale composite height from Fig. 3 row height (2.05 in for one labelled row)."""
    ratios = _grid_height_ratios(n_map_rows)
    return ROW_FIG_H * sum(ratios) / MAP_ROW_RATIO


def _precursor_fig_width() -> float:
    """Match per-column width to the 3-column publication plates."""
    return ROW_FIG_W * len(PRECURSOR_COL_KINDS) / len(COL_KINDS)


# Left of first-column maps; keep clear of lat tick labels (e.g. 35°N).
ROW_LABEL_X_AXES = -0.14
ROW_LABEL_FIG_LEFT = 0.06


def _add_row_label(ax, label: str) -> None:
    ax.text(
        ROW_LABEL_X_AXES,
        0.5,
        label,
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=9,
        color="0.15",
        clip_on=False,
    )


def _uniform_stack_map_rows(
    fig: plt.Figure,
    *,
    n_map_rows: int,
    ncols: int,
    gap_frac: float = MAP_ROW_GAP_FRAC,
) -> None:
    """Stack map rows with equal height and uniform vertical gaps (anchor top row)."""
    if n_map_rows < 2 or len(fig.axes) < n_map_rows * ncols:
        return
    fig.canvas.draw()
    axes = fig.axes[: n_map_rows * ncols]
    top_ps = [axes[c].get_position() for c in range(ncols)]
    y_top = max(p.y1 for p in top_ps)
    row_h = top_ps[0].height
    cbar_reserve = CBAR_GAP_FRAC + CBAR_HEIGHT_FRAC + 0.015
    available = y_top - cbar_reserve
    needed = n_map_rows * row_h + (n_map_rows - 1) * gap_frac
    if needed > available:
        row_h = (available - (n_map_rows - 1) * gap_frac) / n_map_rows
    for ri in range(n_map_rows):
        y1 = y_top - ri * (row_h + gap_frac)
        y0 = y1 - row_h
        for ci in range(ncols):
            ax = axes[ri * ncols + ci]
            p = ax.get_position()
            ax.set_position([p.x0, y0, p.width, row_h])


def _finalize_map_stack_layout(fig: plt.Figure, *, n_map_rows: int, ncols: int) -> None:
    _uniform_stack_map_rows(fig, n_map_rows=n_map_rows, ncols=ncols)


def _cbar_axes_below_bottom_row(
    fig: plt.Figure,
    *,
    n_map_rows: int,
    ncols: int,
    dual: bool,
) -> tuple[plt.Axes, plt.Axes | None]:
    """Place colorbar axes directly under the bottom map row (after manual row layout)."""
    fig.canvas.draw()
    bottom = [fig.axes[(n_map_rows - 1) * ncols + c].get_position() for c in range(ncols)]
    y_map_bottom = min(p.y0 for p in bottom)
    y_cbar = y_map_bottom - CBAR_GAP_FRAC - CBAR_HEIGHT_FRAC
    if dual:
        cax_l = fig.add_axes(
            [bottom[0].x0, y_cbar, bottom[1].x1 - bottom[0].x0, CBAR_HEIGHT_FRAC]
        )
        cax_r = fig.add_axes([bottom[2].x0, y_cbar, bottom[2].width, CBAR_HEIGHT_FRAC])
        return cax_l, cax_r
    cax = fig.add_axes([bottom[0].x0, y_cbar, bottom[-1].x1 - bottom[0].x0, CBAR_HEIGHT_FRAC])
    return cax, None


def plot_grid_ncx3(
    ds: xr.Dataset,
    *,
    categories: list[str],
    scenario: str,
    diff_vmax: float,
    out_stem: Path,
    row_labels: bool = False,
    epoch: str | None = None,
) -> None:
    """N categories x 3 map types; sequential panel letters; dual colorbars."""
    nrows, ncols = len(categories), len(COL_KINDS)
    letters = list(string.ascii_lowercase[: nrows * ncols])
    diff_cmap, diff_norm = diff_cmap_norm(diff_vmax)

    fig = plt.figure(figsize=(ROW_FIG_W, _composite_fig_height(nrows)))
    gs = fig.add_gridspec(
        nrows + 1,
        ncols,
        height_ratios=_grid_height_ratios(nrows),
        hspace=GS_HSPACE_4X3,
        wspace=GS_WSPACE,
    )
    if row_labels:
        fig.subplots_adjust(left=ROW_LABEL_FIG_LEFT)

    im_prob = None
    im_diff = None
    idx = 0
    for ri, cat in enumerate(categories):
        for ci, kind in enumerate(COL_KINDS):
            ax = fig.add_subplot(gs[ri, ci], projection=proj)
            da = get_da(ds, kind=kind, category=cat, duration=DURATION, epoch=epoch)
            im = plot_on_ax(ax, da, kind=kind, category=cat, diff_vmax=diff_vmax)
            if kind == "diff":
                im_diff = im
            else:
                im_prob = im
            format_ax(ax, show_y=(ci == 0), show_x=(ri == nrows - 1))
            if row_labels and ci == 0:
                _add_row_label(ax, ROW_LABELS.get(cat, cat))
            idx += 1

    _finalize_map_stack_layout(fig, n_map_rows=nrows, ncols=ncols)
    _place_panel_letters_in_row_gaps(fig, n_map_rows=nrows, ncols=ncols, letters=letters)

    cax_l, cax_r = _cbar_axes_below_bottom_row(fig, n_map_rows=nrows, ncols=ncols, dual=True)

    if im_prob is not None:
        cb1 = fig.colorbar(im_prob, cax=cax_l, orientation="horizontal", extend="max", extendrect=False)
        cb1.set_label(
            r"$P(\mathrm{successor}\;\mathrm{in}\;7\,\mathrm{d}\,|\,\mathrm{HH\;end}\geq 3\,\mathrm{d})$",
            fontsize=9,
            labelpad=2,
        )
        cb1.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    if im_diff is not None:
        cb2 = fig.colorbar(
            mpl.cm.ScalarMappable(norm=diff_norm, cmap=diff_cmap),
            cax=cax_r,
            orientation="horizontal",
            extend="both",
            extendrect=False,
        )
        cb2.set_label(
            r"$\Delta P = P(\mathrm{pre}) - P(\mathrm{base})$",
            fontsize=9,
            labelpad=2,
        )
        cb2.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    _save(fig, out_stem)
    plt.close(fig)
    print(f"Wrote {out_stem.name}.png/.pdf")


def plot_grid_precursors(
    ds: xr.Dataset,
    *,
    categories: list[str],
    scenario: str,
    diff_vmax: float,
    out_stem: Path,
    row_labels: bool = False,
    col_titles: bool = True,
    epoch: str | None = None,
) -> None:
    """N categories x 2 precursor maps (baseline | precond); one probability colorbar."""
    nrows, ncols = len(categories), len(PRECURSOR_COL_KINDS)
    letters = list(string.ascii_lowercase[: nrows * ncols])

    fig = plt.figure(figsize=(_precursor_fig_width(), _composite_fig_height(nrows)))
    gs = fig.add_gridspec(
        nrows + 1,
        ncols,
        height_ratios=_grid_height_ratios(nrows),
        hspace=GS_HSPACE_4X3,
        wspace=GS_WSPACE,
    )
    if row_labels:
        fig.subplots_adjust(left=ROW_LABEL_FIG_LEFT)

    im_prob = None
    idx = 0
    for ri, cat in enumerate(categories):
        for ci, kind in enumerate(PRECURSOR_COL_KINDS):
            ax = fig.add_subplot(gs[ri, ci], projection=proj)
            da = get_da(ds, kind=kind, category=cat, duration=DURATION, epoch=epoch)
            im_prob = plot_on_ax(ax, da, kind=kind, category=cat, diff_vmax=diff_vmax)
            format_ax(ax, show_y=(ci == 0), show_x=(ri == nrows - 1))
            if row_labels and ci == 0:
                _add_row_label(ax, ROW_LABELS.get(cat, cat))
            if col_titles and ri == 0:
                ax.set_title(PRECURSOR_COL_TITLES[ci], fontsize=9, pad=6)
            idx += 1

    _finalize_map_stack_layout(fig, n_map_rows=nrows, ncols=ncols)
    _place_panel_letters_in_row_gaps(fig, n_map_rows=nrows, ncols=ncols, letters=letters)

    cax, _ = _cbar_axes_below_bottom_row(fig, n_map_rows=nrows, ncols=ncols, dual=False)
    if im_prob is not None:
        cb = fig.colorbar(im_prob, cax=cax, orientation="horizontal", extend="max", extendrect=False)
        cb.set_label(
            r"$P(\mathrm{successor}\;\mathrm{in}\;7\,\mathrm{d}\,|\,\mathrm{HH\;end}\geq 3\,\mathrm{d})$",
            fontsize=9,
            labelpad=2,
        )
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    if epoch:
        fig.suptitle(EPOCH_LABELS[epoch], fontsize=10, y=0.995, color="0.15")

    _save(fig, out_stem)
    plt.close(fig)
    print(f"Wrote {out_stem.name}.png/.pdf")


def plot_grid_4x3(
    ds: xr.Dataset,
    *,
    scenario: str,
    diff_vmax: float,
    epoch: str | None = None,
) -> None:
    """4 categories x 3 map types; panel letters a-l; dual colorbars."""
    plot_grid_ncx3(
        ds,
        categories=CATEGORIES,
        scenario=scenario,
        diff_vmax=diff_vmax,
        out_stem=OUTDIR / f"Fig4_postHH_{scenario}_4x3",
        row_labels=False,
        epoch=epoch,
    )


def plot_category_row(
    ds: xr.Dataset,
    *,
    scenario: str,
    category: str,
    letter_offset: int,
    diff_vmax: float,
    attach_cbar: bool = False,
    out_stem: Path | None = None,
    letters: list[str] | None = None,
    row_label: str | None = None,
    epoch: str | None = None,
) -> None:
    """1x3 row per category (for manual assembly); optional dual colorbars on last row."""
    if letters is None:
        letters = list(string.ascii_lowercase)[letter_offset : letter_offset + 3]
    diff_cmap, diff_norm = diff_cmap_norm(diff_vmax)

    if attach_cbar:
        fig = plt.figure(figsize=(ROW_FIG_W, ROW_FIG_H_CBAR))
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[MAP_ROW_RATIO, CBAR_ROW_RATIO],
            hspace=GS_HSPACE_ROW_CBAR,
            wspace=GS_WSPACE,
        )
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax_l = fig.add_subplot(gs[1, :2])
        cax_r = fig.add_subplot(gs[1, 2])
        if row_label:
            fig.subplots_adjust(left=ROW_LABEL_FIG_LEFT)
    else:
        fig = plt.figure(figsize=(ROW_FIG_W, ROW_FIG_H))
        gs = fig.add_gridspec(1, 3, wspace=GS_WSPACE)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax_l = cax_r = None
        if row_label:
            fig.subplots_adjust(left=ROW_LABEL_FIG_LEFT)

    im_prob = None
    im_diff = None
    for ci, kind in enumerate(COL_KINDS):
        ax = axes[ci]
        da = get_da(ds, kind=kind, category=category, duration=DURATION, epoch=epoch)
        im = plot_on_ax(ax, da, kind=kind, category=category, diff_vmax=diff_vmax)
        if kind == "diff":
            im_diff = im
        else:
            im_prob = im
        format_ax(ax, show_y=(ci == 0), show_x=True)
        add_panel_letter(ax, letters[ci])
        if row_label and ci == 0:
            _add_row_label(ax, row_label)

    if attach_cbar and cax_l is not None and im_prob is not None:
        cb1 = fig.colorbar(im_prob, cax=cax_l, orientation="horizontal", extend="max", extendrect=False)
        cb1.set_label(
            r"$P(\mathrm{successor}\;\mathrm{in}\;7\,\mathrm{d}\,|\,\mathrm{HH\;end}\geq 3\,\mathrm{d})$",
            fontsize=9,
            labelpad=2,
        )
        cb1.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)
    if attach_cbar and cax_r is not None and im_diff is not None:
        cb2 = fig.colorbar(
            mpl.cm.ScalarMappable(norm=diff_norm, cmap=diff_cmap),
            cax=cax_r,
            orientation="horizontal",
            extend="both",
            extendrect=False,
        )
        cb2.set_label(
            r"$\Delta P = P(\mathrm{pre}) - P(\mathrm{base})$",
            fontsize=9,
            labelpad=2,
        )
        cb2.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    stem = out_stem if out_stem is not None else OUTDIR / f"Fig4_postHH_{scenario}_row_{category}"
    _save(fig, stem)
    plt.close(fig)
    print(f"Wrote {stem.name}.png/.pdf")


def plot_category_row_precursors(
    ds: xr.Dataset,
    *,
    scenario: str,
    category: str,
    diff_vmax: float,
    attach_cbar: bool = False,
    out_stem: Path | None = None,
    letters: list[str] | None = None,
    row_label: str | None = None,
    col_titles: bool = True,
    epoch: str | None = None,
) -> None:
    """1x2 row per category: baseline | precond; optional probability colorbar."""
    if letters is None:
        letters = ["a", "b"]

    if attach_cbar:
        fig = plt.figure(figsize=(_precursor_fig_width(), ROW_FIG_H_CBAR))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[MAP_ROW_RATIO, CBAR_ROW_RATIO],
            hspace=GS_HSPACE_ROW_CBAR,
            wspace=GS_WSPACE,
        )
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(2)]
        cax = fig.add_subplot(gs[1, :])
        if row_label:
            fig.subplots_adjust(left=ROW_LABEL_FIG_LEFT)
    else:
        fig = plt.figure(figsize=(_precursor_fig_width(), ROW_FIG_H))
        gs = fig.add_gridspec(1, 2, wspace=GS_WSPACE)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(2)]
        cax = None
        if row_label:
            fig.subplots_adjust(left=ROW_LABEL_FIG_LEFT)

    im_prob = None
    for ci, kind in enumerate(PRECURSOR_COL_KINDS):
        ax = axes[ci]
        da = get_da(ds, kind=kind, category=category, duration=DURATION, epoch=epoch)
        im_prob = plot_on_ax(ax, da, kind=kind, category=category, diff_vmax=diff_vmax)
        format_ax(ax, show_y=(ci == 0), show_x=True)
        add_panel_letter(ax, letters[ci])
        if col_titles:
            ax.set_title(PRECURSOR_COL_TITLES[ci], fontsize=9, pad=6)
        if row_label and ci == 0:
            _add_row_label(ax, row_label)

    if attach_cbar and cax is not None and im_prob is not None:
        cb = fig.colorbar(im_prob, cax=cax, orientation="horizontal", extend="max", extendrect=False)
        cb.set_label(
            r"$P(\mathrm{successor}\;\mathrm{in}\;7\,\mathrm{d}\,|\,\mathrm{HH\;end}\geq 3\,\mathrm{d})$",
            fontsize=9,
            labelpad=2,
        )
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    stem = out_stem if out_stem is not None else OUTDIR / f"Fig4_postHH_{scenario}_row_{category}_precursors"
    _save(fig, stem)
    plt.close(fig)
    print(f"Wrote {stem.name}.png/.pdf")


def plot_publication_precursors_one_ssp(
    ds: xr.Dataset,
    *,
    scenario: str,
    ssp_label: str,
    diff_vmax: float,
    pub_dir: Path,
    epoch: str | None = None,
) -> None:
    """Fig. 4a bundle for one SSP: 3 row strips + full 3x2 precursor plate."""
    pub_dir.mkdir(parents=True, exist_ok=True)
    epoch_tag = f"_{epoch}" if epoch else ""
    tag = f"Fig4a_{scenario}_ENSEMBLE3{epoch_tag}"

    row_specs = [
        (1, "compound_again", ["a", "b"], f"{tag}_ROW01_compound_again_baseline_precond", False),
        (2, "hot_only", ["c", "d"], f"{tag}_ROW02_hot_only_baseline_precond", False),
        (3, "dry_only", ["e", "f"], f"{tag}_ROW03_dry_only_baseline_precond_WITH_COLORBAR", True),
    ]
    for _row_num, cat, letters, fname, with_cbar in row_specs:
        plot_category_row_precursors(
            ds,
            scenario=scenario,
            category=cat,
            diff_vmax=diff_vmax,
            attach_cbar=with_cbar,
            out_stem=pub_dir / fname,
            letters=letters,
            row_label=ROW_LABELS.get(cat),
            epoch=epoch,
        )

    plot_grid_precursors(
        ds,
        categories=MAIN_TEXT_CATEGORIES,
        scenario=scenario,
        diff_vmax=diff_vmax,
        out_stem=pub_dir / f"{tag}_FULL_3x2_panels_a-f_WITH_COLORBAR",
        row_labels=True,
        col_titles=True,
        epoch=epoch,
    )

    epoch_line = ""
    if epoch:
        epoch_line = f"- Epoch: **{EPOCH_LABELS[epoch]}**\n"

    (pub_dir / f"{tag}_README_snippet.md").write_text(
        f"""## {ssp_label} — Fig. 4a precursors{f' ({epoch})' if epoch else ''}

{epoch_line}- `{tag}_FULL_3x2_panels_a-f_WITH_COLORBAR.png` — panels **a–f**
- Columns: **Baseline** | **Preconditioned** (14-d dry spell, no wet day ≥1 mm)
- Rows: compound again | hot only | dry only
""",
        encoding="utf-8",
    )


def plot_publication_precursors_all_ssp(
    ds_by_ssp: dict[str, xr.Dataset],
    *,
    diff_vmax: float,
    pub_dir: Path,
    epochs: list[str] | None = None,
) -> None:
    """Fig. 4a: baseline | precond for SSP126, SSP245, and SSP585 (no Delta P)."""
    pub_dir.mkdir(parents=True, exist_ok=True)
    snippets: list[str] = []
    epoch_list = epochs if epochs is not None else [None]
    for ep in epoch_list:
        for ssp_label, ssp_tag in SSP_COMPARE:
            plot_publication_precursors_one_ssp(
                ds_by_ssp[ssp_tag],
                scenario=ssp_tag,
                ssp_label=ssp_label,
                diff_vmax=diff_vmax,
                pub_dir=pub_dir,
                epoch=ep,
            )
            tag = f"Fig4a_{ssp_tag}_ENSEMBLE3{('_' + ep) if ep else ''}"
            snippets.append((pub_dir / f"{tag}_README_snippet.md").read_text(encoding="utf-8"))

    epoch_note = ""
    if epochs:
        epoch_note = (
            f"\n**Epoch windows:** {', '.join(EPOCH_LABELS[e] for e in epochs)} "
            "(spell-end triggers counted within epoch; thresholds from full SSP 2015–2100).\n"
        )

    readme = pub_dir / "README.md"
    readme.write_text(
        f"""# Fig. 4a — Precursor probabilities (3-model ensemble)

Generated by `plot_fig4_successors.py --publication-precursors`
{' --epoch-cache' if epochs else ''}.

**One plate per SSP** (baseline | preconditioned only — **no ΔP**; scenario contrasts are in Fig. 4b).
{epoch_note}
{"".join(snippets)}

## Why three SSP plates?

Each SSP has its own baseline and preconditioned successor probability fields. Fig. 4b then shows
**ΔP = P(pre) − P(base)** across SSP126 | SSP245 | SSP585 without repeating these absolute maps.

See `../{Path(pub_dir.name).parent.name}/{PUBLICATION_SSP_COMPARE_SUBDIR.split('/')[-1]}/` for Fig. 4b.
""",
        encoding="utf-8",
    )
    print(f"Wrote Fig. 4a precursor bundle in {pub_dir}")


def plot_publication_main(
    ds: xr.Dataset,
    *,
    scenario: str,
    diff_vmax: float,
    pub_dir: Path,
) -> None:
    """Main-text bundle: 3 row strips + one full 3x3 plate, distinct filenames."""
    pub_dir.mkdir(parents=True, exist_ok=True)
    tag = f"Fig4_PUBLICATION_{scenario}_ENSEMBLE3"

    row_specs = [
        (
            1,
            "compound_again",
            ["a", "b", "c"],
            f"{tag}_ROW01_compound_again_1x3_baseline_precond_deltaP",
            False,
        ),
        (
            2,
            "hot_only",
            ["d", "e", "f"],
            f"{tag}_ROW02_hot_only_1x3_baseline_precond_deltaP",
            False,
        ),
        (
            3,
            "dry_only",
            ["g", "h", "i"],
            f"{tag}_ROW03_dry_only_1x3_baseline_precond_deltaP_WITH_COLORBARS",
            True,
        ),
    ]
    for _row_num, cat, letters, fname, with_cbar in row_specs:
        plot_category_row(
            ds,
            scenario=scenario,
            category=cat,
            letter_offset=0,
            diff_vmax=diff_vmax,
            attach_cbar=with_cbar,
            out_stem=pub_dir / fname,
            letters=letters,
            row_label=ROW_LABELS.get(cat),
        )

    plot_grid_ncx3(
        ds,
        categories=MAIN_TEXT_CATEGORIES,
        scenario=scenario,
        diff_vmax=diff_vmax,
        out_stem=pub_dir / f"{tag}_FULL_3x3_panels_a-i_WITH_COLORBARS",
        row_labels=True,
    )

    readme = pub_dir / "README.md"
    readme.write_text(
        f"""# Fig. 4 publication export ({scenario}, 3-model ensemble)

Generated by `plot_fig4_successors.py --publication-main`.

## Row strips (manual merge, top to bottom)

1. `{tag}_ROW01_compound_again_1x3_baseline_precond_deltaP.png` — panels **a, b, c**
2. `{tag}_ROW02_hot_only_1x3_baseline_precond_deltaP.png` — panels **d, e, f**
3. `{tag}_ROW03_dry_only_1x3_baseline_precond_deltaP_WITH_COLORBARS.png` — panels **g, h, i** + colorbars

## Full plate (ready-made)

- `{tag}_FULL_3x3_panels_a-i_WITH_COLORBARS.png` / `.pdf`

Columns: Baseline | Preconditioned (14-d dry spell) | ΔP
""",
        encoding="utf-8",
    )
    print(f"Wrote publication bundle in {pub_dir}")


def load_ssp_datasets(
    *,
    window_days: int,
    duration: int,
    pre_tag: str,
    min_models: int,
    epoch_cache: bool = False,
) -> dict[str, xr.Dataset]:
    """Load ensemble caches for SSP126, SSP245, SSP585."""
    out: dict[str, xr.Dataset] = {}
    for _label, tag in SSP_COMPARE:
        fp = find_cache(
            tag,
            window_days=window_days,
            duration=duration,
            pre_tag=pre_tag,
            min_models=min_models,
            epoch_cache=epoch_cache,
        )
        ds = load_fields(fp)
        n_mod = int(ds.attrs.get("n_models", 0))
        if n_mod < min_models:
            ds.close()
            raise RuntimeError(f"{tag}: expected >={min_models} models, got {n_mod}")
        print(f"  loaded {tag}: {fp.name} (n_models={n_mod})")
        out[tag] = ds
    return out


def plot_category_row_by_ssp(
    ds_by_ssp: dict[str, xr.Dataset],
    *,
    category: str,
    kind: str,
    diff_vmax: float,
    letters: list[str],
    out_stem: Path,
    row_label: str | None = None,
    attach_cbar: bool = False,
    col_titles: bool = True,
    epoch: str | None = None,
) -> None:
    """1x3 row: SSP126 | SSP245 | SSP585 for one category and map kind (e.g. diff)."""
    diff_cmap, diff_norm = diff_cmap_norm(diff_vmax)

    if attach_cbar and kind == "diff":
        fig = plt.figure(figsize=(ROW_FIG_W, ROW_FIG_H_CBAR))
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[MAP_ROW_RATIO, CBAR_ROW_RATIO],
            hspace=GS_HSPACE_ROW_CBAR,
            wspace=GS_WSPACE,
        )
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax = fig.add_subplot(gs[1, :])
    else:
        fig = plt.figure(figsize=(ROW_FIG_W, ROW_FIG_H))
        gs = fig.add_gridspec(1, 3, wspace=GS_WSPACE)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
        cax = None

    if row_label:
        fig.subplots_adjust(left=ROW_LABEL_FIG_LEFT)

    im_last = None
    for ci, (ssp_label, ssp_tag) in enumerate(SSP_COMPARE):
        ax = axes[ci]
        da = get_da(ds_by_ssp[ssp_tag], kind=kind, category=category, duration=DURATION, epoch=epoch)
        im_last = plot_on_ax(ax, da, kind=kind, category=category, diff_vmax=diff_vmax)
        format_ax(ax, show_y=(ci == 0), show_x=True)
        add_panel_letter(ax, letters[ci])
        if row_label and ci == 0:
            _add_row_label(ax, row_label)
        if col_titles:
            ax.set_title(ssp_label, fontsize=9, pad=6)

    if attach_cbar and cax is not None and im_last is not None and kind == "diff":
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=diff_norm, cmap=diff_cmap),
            cax=cax,
            orientation="horizontal",
            extend="both",
            extendrect=False,
        )
        cb.set_label(
            r"$\Delta P = P(\mathrm{pre}) - P(\mathrm{base})$",
            fontsize=9,
            labelpad=2,
        )
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    _save(fig, out_stem)
    plt.close(fig)
    print(f"Wrote {out_stem.name}.png/.pdf")


def plot_grid_categories_by_ssp(
    ds_by_ssp: dict[str, xr.Dataset],
    *,
    categories: list[str],
    kind: str,
    diff_vmax: float,
    out_stem: Path,
    row_labels: bool = True,
    col_titles: bool = True,
    epoch: str | None = None,
) -> None:
    """N categories x 3 SSPs; one map kind (diff); single shared colorbar."""
    nrows, ncols = len(categories), len(SSP_COMPARE)
    letters = list(string.ascii_lowercase[: nrows * ncols])
    diff_cmap, diff_norm = diff_cmap_norm(diff_vmax)

    fig = plt.figure(figsize=(ROW_FIG_W, _composite_fig_height(nrows)))
    gs = fig.add_gridspec(
        nrows + 1,
        ncols,
        height_ratios=_grid_height_ratios(nrows),
        hspace=GS_HSPACE_4X3,
        wspace=GS_WSPACE,
    )
    if row_labels:
        fig.subplots_adjust(left=ROW_LABEL_FIG_LEFT)

    im_last = None
    idx = 0
    for ri, cat in enumerate(categories):
        for ci, (ssp_label, ssp_tag) in enumerate(SSP_COMPARE):
            ax = fig.add_subplot(gs[ri, ci], projection=proj)
            da = get_da(ds_by_ssp[ssp_tag], kind=kind, category=cat, duration=DURATION, epoch=epoch)
            im_last = plot_on_ax(ax, da, kind=kind, category=cat, diff_vmax=diff_vmax)
            format_ax(ax, show_y=(ci == 0), show_x=(ri == nrows - 1))
            if row_labels and ci == 0:
                _add_row_label(ax, ROW_LABELS.get(cat, cat))
            if col_titles and ri == 0:
                ax.set_title(ssp_label, fontsize=9, pad=6)
            idx += 1

    _finalize_map_stack_layout(fig, n_map_rows=nrows, ncols=ncols)
    _place_panel_letters_in_row_gaps(fig, n_map_rows=nrows, ncols=ncols, letters=letters)

    cax, _ = _cbar_axes_below_bottom_row(fig, n_map_rows=nrows, ncols=ncols, dual=False)
    if im_last is not None and kind == "diff":
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=diff_norm, cmap=diff_cmap),
            cax=cax,
            orientation="horizontal",
            extend="both",
            extendrect=False,
        )
        cb.set_label(
            r"$\Delta P = P(\mathrm{pre}) - P(\mathrm{base})$",
            fontsize=9,
            labelpad=2,
        )
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    if epoch:
        fig.suptitle(EPOCH_LABELS[epoch], fontsize=10, y=0.995, color="0.15")

    _save(fig, out_stem)
    plt.close(fig)
    print(f"Wrote {out_stem.name}.png/.pdf")


def plot_publication_ssp_compare(
    ds_by_ssp: dict[str, xr.Dataset],
    *,
    diff_vmax: float,
    pub_dir: Path,
    epochs: list[str] | None = None,
) -> None:
    """Fig. 4b: rows = categories, cols = SSP126/245/585, maps = Delta P only."""
    pub_dir.mkdir(parents=True, exist_ok=True)
    epoch_list = epochs if epochs is not None else [None]
    for ep in epoch_list:
        epoch_tag = f"_{ep}" if ep else ""
        tag = f"Fig4b_ENSEMBLE3{epoch_tag}"

        row_specs = [
            (1, "compound_again", ["a", "b", "c"]),
            (2, "hot_only", ["d", "e", "f"]),
            (3, "dry_only", ["g", "h", "i"]),
        ]
        for _n, cat, letters in row_specs:
            plot_category_row_by_ssp(
                ds_by_ssp,
                category=cat,
                kind="diff",
                diff_vmax=diff_vmax,
                letters=letters,
                out_stem=pub_dir / f"{tag}_ROW0{_n}_{cat}_DeltaP_ssp126_ssp245_ssp585",
                row_label=ROW_LABELS.get(cat),
                attach_cbar=(cat == "dry_only"),
                epoch=ep,
            )

        plot_grid_categories_by_ssp(
            ds_by_ssp,
            categories=MAIN_TEXT_CATEGORIES,
            kind="diff",
            diff_vmax=diff_vmax,
            out_stem=pub_dir / f"{tag}_FULL_3x3_DeltaP_ssp126_ssp245_ssp585_WITH_COLORBAR",
            row_labels=True,
            col_titles=True,
            epoch=ep,
        )

    epoch_note = ""
    if epochs:
        epoch_note = (
            f"\n**Epoch windows:** one 3×3 plate per epoch "
            f"({', '.join(EPOCH_LABELS[e] for e in epochs)}). "
            "Spell-end triggers counted within epoch.\n"
        )

    readme = pub_dir / "README.md"
    readme.write_text(
        f"""# Fig. 4b — SSP comparison of ΔP (3-model ensemble)

Generated by `plot_fig4_successors.py --publication-ssp-compare`
{' --epoch-cache' if epochs else ''}.
{epoch_note}
## Layout (per epoch plate)

|  | SSP126 | SSP245 | SSP585 |
|--|--------|--------|--------|
| **Compound again** | a | b | c |
| **Hot only** | d | e | f |
| **Dry only** | g | h | i |

All maps: **ΔP** = P(14-d dry-spell precondition) − P(baseline), after HH run ≥3 d ends.

Absolute baseline and preconditioned probabilities are in Fig. 4a (`../{Path(pub_dir.name).parent.name}/{PUBLICATION_PRECURSORS_SUBDIR.split('/')[-1]}/`).

## Files

- `Fig4b_ENSEMBLE3{{_EPOCH}}_FULL_3x3_DeltaP_ssp126_ssp245_ssp585_WITH_COLORBAR.png` — full plate
- `Fig4b_ENSEMBLE3{{_EPOCH}}_ROW01_compound_again_DeltaP_ssp126_ssp245_ssp585.png` — row strip
- `Fig4b_ENSEMBLE3{{_EPOCH}}_ROW02_hot_only_DeltaP_ssp126_ssp245_ssp585.png`
- `Fig4b_ENSEMBLE3{{_EPOCH}}_ROW03_dry_only_DeltaP_ssp126_ssp245_ssp585.png` — includes colorbar
""",
        encoding="utf-8",
    )
    print(f"Wrote Fig. 4b SSP-compare bundle in {pub_dir}")


def plot_standalone_panels(
    ds: xr.Dataset,
    *,
    scenario: str,
    diff_vmax: float,
    epoch: str | None = None,
) -> None:
    panel_dir = PANEL_DIR / scenario
    panel_dir.mkdir(parents=True, exist_ok=True)
    letters = list(string.ascii_lowercase[: len(CATEGORIES) * len(COL_KINDS)])
    idx = 0
    for cat in CATEGORIES:
        for kind in COL_KINDS:
            da = get_da(ds, kind=kind, category=cat, duration=DURATION, epoch=epoch)
            fig = plt.figure(figsize=(3.4, 1.80))
            ax = plt.axes(projection=proj)
            plot_on_ax(ax, da, kind=kind, category=cat, diff_vmax=diff_vmax)
            format_ax(ax, show_y=True, show_x=True)
            add_panel_letter(ax, letters[idx])
            stem = panel_dir / f"Fig4_{letters[idx]}_{cat}_{kind}"
            fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
            plt.close(fig)
            idx += 1
    print(f"Wrote {idx} panels in {panel_dir}")


def plot_colorbars(*, scenario: str, diff_vmax: float) -> None:
    diff_cmap, diff_norm = diff_cmap_norm(diff_vmax)

    fig = plt.figure(figsize=(4.3, 0.55))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.07, 0.55, 0.90, 0.25])
    norm = mpl.colors.Normalize(vmin=PROB_VMIN, vmax=PROB_VMAX)
    cb = mpl.colorbar.ColorbarBase(
        ax, cmap=PROB_CMAP, norm=norm, orientation="horizontal", extend="max"
    )
    cb.set_label(
        r"$P(\mathrm{successor}\;\mathrm{in}\;7\,\mathrm{d}\,|\,\mathrm{HH\;end}\geq 3\,\mathrm{d})$",
        fontsize=10,
    )
    cb.ax.tick_params(labelsize=9)
    _save(fig, OUTDIR / f"Fig4_postHH_{scenario}_prob_colorbar")
    plt.close(fig)

    fig = plt.figure(figsize=(4.3, 0.55))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.07, 0.55, 0.90, 0.25])
    cb = mpl.colorbar.ColorbarBase(ax, cmap=diff_cmap, norm=diff_norm, orientation="horizontal", extend="both")
    cb.set_label(r"$\Delta P = P(\mathrm{pre}) - P(\mathrm{base})$", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    _save(fig, OUTDIR / f"Fig4_postHH_{scenario}_diff_colorbar")
    plt.close(fig)
    print(f"Wrote Fig4_postHH_{scenario}_prob_colorbar and Fig4_postHH_{scenario}_diff_colorbar")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--duration", type=int, default=DURATION)
    parser.add_argument("--pre-tag", default=PRE_TAG)
    parser.add_argument("--diff-vmax", type=float, default=DIFF_VMAX)
    parser.add_argument("--min-models", type=int, default=3, help="Require ensemble cache with at least N models.")
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Only write 4×3 plate + panels/ (skip per-row exports and standalone colorbars).",
    )
    parser.add_argument(
        "--rows-only",
        action="store_true",
        help="Cleanup prior PNG/PDF; write per-category 1×3 row panels only (skip panels/, colorbars).",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="With --rows-only: skip deleting existing Fig4_postHH PNG/PDF (for batch replot after first scenario).",
    )
    parser.add_argument(
        "--with-4x3",
        action="store_true",
        help="With --rows-only: also write full 4×3 plate (default when compact layout is enabled).",
    )
    parser.add_argument(
        "--no-4x3",
        action="store_true",
        help="With --rows-only: skip full 4×3 plate even if compact layout is enabled.",
    )
    parser.add_argument(
        "--publication-main",
        action="store_true",
        help=(
            "Legacy: SSP585 baseline | precond | ΔP (3x3). "
            "Prefer --publication-precursors and --publication-ssp-compare."
        ),
    )
    parser.add_argument(
        "--publication-precursors",
        action="store_true",
        help=(
            "Fig. 4a: baseline | precond (3x2) for SSP126, SSP245, and SSP585 "
            f"under {PUBLICATION_PRECURSORS_SUBDIR}/."
        ),
    )
    parser.add_argument(
        "--publication-ssp-compare",
        action="store_true",
        help=(
            "Fig. 4b: ΔP comparison 3x3 (rows=categories, cols=SSP126/245/585) "
            f"under {PUBLICATION_SSP_COMPARE_SUBDIR}/."
        ),
    )
    parser.add_argument(
        "--publication-fig4",
        action="store_true",
        help="Write both Fig. 4a (precursors, all SSPs) and Fig. 4b (ΔP compare).",
    )
    parser.add_argument(
        "--epoch-cache",
        action="store_true",
        help=(
            f"Load {EPOCH_CACHE_TAG} ensemble caches (NF/MC/EC dimension) instead of "
            "full-period SSP caches. Auto-enabled for --publication-fig4 when epoch caches exist."
        ),
    )
    parser.add_argument(
        "--no-cleanup-legacy-publication",
        action="store_true",
        help=(
            "With --epoch-cache: keep legacy full-period publication PNG/PDF/snippet files "
            f"in {PUBLICATION_PRECURSORS_SUBDIR}/ and {PUBLICATION_SSP_COMPARE_SUBDIR}/."
        ),
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        choices=[ep[0] for ep in EPOCHS],
        default=None,
        help=(
            "With --epoch-cache: plot these epochs (default: NF MC EC). "
            "Omit for full-period caches."
        ),
    )
    parser.add_argument(
        "--skip-epoch-sanity-check",
        action="store_true",
        help="Skip global-mean ΔP(dry_only) epoch verification before plotting.",
    )
    parser.add_argument(
        "--publication-fig4-main-core",
        action="store_true",
        help=(
            "Fig. 4 MAIN CORE: P_base epoch×SSP 3×3 plates (continuous probability fields) "
            "under PUBLICATION_Fig4_MAIN_CORE/. Delegates to plot_fig4_baseline_probability.py."
        ),
    )
    args = parser.parse_args()

    if args.publication_fig4_main_core:
        from plot_fig4_baseline_probability import run_main_core

        run_main_core()
        return

    if args.publication_fig4:
        args.publication_precursors = True
        args.publication_ssp_compare = True

    pub_modes = sum(
        bool(x)
        for x in (
            args.publication_main,
            args.publication_precursors,
            args.publication_ssp_compare,
        )
    )
    if sum(bool(x) for x in (args.main_only, args.rows_only)) > 0 and pub_modes > 0:
        parser.error("--publication-* flags are mutually exclusive with --main-only and --rows-only.")
    if args.publication_main and (args.publication_precursors or args.publication_ssp_compare):
        parser.error("--publication-main cannot be combined with other --publication-* flags.")

    diff_vmax = float(args.diff_vmax)
    scenario = args.scenario
    epoch_cache = bool(args.epoch_cache)
    epochs: list[str] | None = None

    pub_requested = bool(args.publication_precursors or args.publication_ssp_compare)
    pub_requires_epochs = pub_requested and not args.publication_main
    if pub_requires_epochs and not epoch_cache:
        if epoch_caches_available(
            window_days=args.window_days,
            duration=args.duration,
            pre_tag=args.pre_tag,
            min_models=args.min_models,
        ):
            epoch_cache = True
            print(
                "Auto-enabled --epoch-cache: NF/MC/EC epoch caches found "
                "(Methods require epoch-stratified Fig. 4)."
            )
        else:
            parser.error(
                "Epoch-stratified caches are required for --publication-precursors / "
                "--publication-ssp-compare / --publication-fig4 but were not found for all SSPs. "
                "Run compute_post_hh_epoch_ensemble.py first."
            )

    if epoch_cache:
        epochs = args.epochs if args.epochs else [ep[0] for ep in EPOCHS]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    if pub_requested:
        if epoch_cache and not args.no_cleanup_legacy_publication:
            removed: list[str] = []
            for sub in (PUBLICATION_PRECURSORS_SUBDIR, PUBLICATION_SSP_COMPARE_SUBDIR):
                removed.extend(cleanup_legacy_publication_outputs(OUTDIR / sub))
            if removed:
                print(
                    f"Removed {len(removed)} legacy full-period publication file(s) "
                    "(kept epoch-stratified _NF_/_MC_/_EC_ outputs)."
                )
        cache_kind = "epoch-stratified" if epoch_cache else "full-period"
        print(f"Loading SSP126, SSP245, SSP585 {cache_kind} ensemble caches...")
        ds_by_ssp = load_ssp_datasets(
            window_days=args.window_days,
            duration=args.duration,
            pre_tag=args.pre_tag,
            min_models=args.min_models,
            epoch_cache=epoch_cache,
        )
        if epoch_cache:
            epochs = epochs or [ep[0] for ep in EPOCHS]
            print(f"  epochs: {', '.join(epochs or [])}")
            assert_epoch_stratified_datasets(ds_by_ssp, epochs=epochs)
            if not args.skip_epoch_sanity_check:
                full_by_ssp: dict[str, xr.Dataset] | None = None
                try:
                    full_by_ssp = load_ssp_datasets(
                        window_days=args.window_days,
                        duration=args.duration,
                        pre_tag=args.pre_tag,
                        min_models=args.min_models,
                        epoch_cache=False,
                    )
                except FileNotFoundError:
                    full_by_ssp = None
                print_epoch_sanity_check(
                    ds_by_ssp,
                    epochs=epochs,
                    full_period_ds_by_ssp=full_by_ssp,
                )
                if full_by_ssp is not None:
                    for ds in full_by_ssp.values():
                        ds.close()
        print(
            f"  ΔP scale: ±{diff_vmax:.2f}; grey neutral band "
            f"[{DIFF_NEUTRAL_LO:.2f}, {DIFF_NEUTRAL_HI:.2f}]"
        )
        precursors_subdir, ssp_compare_subdir = publication_subdirs(epoch_cache=epoch_cache)
        if args.publication_precursors:
            pub_dir = OUTDIR / precursors_subdir
            plot_publication_precursors_all_ssp(
                ds_by_ssp,
                diff_vmax=diff_vmax,
                pub_dir=pub_dir,
                epochs=epochs if epoch_cache else None,
            )
        if args.publication_ssp_compare:
            pub_dir = OUTDIR / ssp_compare_subdir
            plot_publication_ssp_compare(
                ds_by_ssp,
                diff_vmax=diff_vmax,
                pub_dir=pub_dir,
                epochs=epochs if epoch_cache else None,
            )
        for ds in ds_by_ssp.values():
            ds.close()
        print("Done:", OUTDIR)
        return

    cache_fp = find_cache(
        scenario,
        window_days=args.window_days,
        duration=args.duration,
        pre_tag=args.pre_tag,
        min_models=args.min_models,
        epoch_cache=epoch_cache,
    )
    print(f"Cache: {cache_fp.name}")
    ds = load_fields(cache_fp)
    n_mod = int(ds.attrs.get("n_models", 0))
    print(f"  n_models={n_mod}  models={ds.attrs.get('models', '?')}")
    print(
        f"  ΔP scale: ±{diff_vmax:.2f}; grey neutral band "
        f"[{DIFF_NEUTRAL_LO:.2f}, {DIFF_NEUTRAL_HI:.2f}]"
    )
    if n_mod < args.min_models:
        ds.close()
        raise RuntimeError(f"Expected >={args.min_models} models, got {n_mod}")

    if args.publication_main:
        pub_dir = OUTDIR / PUBLICATION_SUBDIR
        plot_publication_main(ds, scenario=scenario, diff_vmax=diff_vmax, pub_dir=pub_dir)
    elif args.rows_only:
        if not args.no_cleanup:
            removed = cleanup_fig4_outputs()
            print(f"Cleanup: removed {len(removed)} PNG/PDF file(s) from {OUTDIR}")
        for ri, cat in enumerate(CATEGORIES):
            plot_category_row(
                ds,
                scenario=scenario,
                category=cat,
                letter_offset=ri * 3,
                diff_vmax=diff_vmax,
                attach_cbar=(cat == CATEGORIES[-1]),
            )
        include_4x3 = ROWS_ONLY_INCLUDE_4X3 and not args.no_4x3
        if args.with_4x3:
            include_4x3 = True
        if include_4x3:
            plot_grid_4x3(ds, scenario=scenario, diff_vmax=diff_vmax)
            print(
                "4×3 included "
                f"(ROW_FIG_H={ROW_FIG_H}, GS_HSPACE_4X3={GS_HSPACE_4X3}, GS_WSPACE={GS_WSPACE})"
            )
        else:
            print("4×3 skipped (ROWS_ONLY_INCLUDE_4X3=False or --no-4x3)")
    else:
        plot_grid_4x3(ds, scenario=scenario, diff_vmax=diff_vmax)
        if not args.main_only:
            for ri, cat in enumerate(CATEGORIES):
                plot_category_row(
                    ds,
                    scenario=scenario,
                    category=cat,
                    letter_offset=ri * 3,
                    diff_vmax=diff_vmax,
                    attach_cbar=(cat == CATEGORIES[-1]),
                )
            plot_colorbars(scenario=scenario, diff_vmax=diff_vmax)
        plot_standalone_panels(ds, scenario=scenario, diff_vmax=diff_vmax)

    ds.close()
    print("Done:", OUTDIR)


if __name__ == "__main__":
    main()
