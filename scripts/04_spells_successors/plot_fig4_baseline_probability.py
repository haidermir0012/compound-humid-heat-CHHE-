#!/usr/bin/env python3
"""
Figure 4 MAIN CORE (FINAL): baseline successor probability P_base on epoch × SSP grid.

Replaces patchy ΔP maps for main-text Fig. 4 with continuous probability fields
(Fig. 2-style 3×3: rows NF/MC/EC, cols SSP126/245/585, coastlines-only basemap).

Reads epoch-stratified ensemble caches only (epochsNF-MC-EC).

Writes:
  Manuscript final plots/figures/Fig4_postHH/PUBLICATION_Fig4_MAIN_CORE/
    Fig4_MAIN_Pbase_{category}_3x3_epochs_SSP_WITH_COLORBAR.png/.pdf
    Fig4_MAIN_Pbase_{category}_EC_1x3_SSP_WITH_COLORBAR.png/.pdf  (compact EC row)
    README.md
"""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_fig4_successors import (  # noqa: E402
    CBAR_GAP_FRAC,
    CBAR_HEIGHT_FRAC,
    DURATION,
    EPOCH_LABELS,
    EPOCHS,
    GS_WSPACE,
    MAP_ROW_GAP_FRAC,
    PANEL_LETTER_TOP_PAD,
    PNG_DPI,
    PRE_TAG,
    PROB_CMAP,
    PROB_VMAX,
    PROB_VMIN,
    PROJECT_DIR,
    ROW_FIG_W,
    ROW_LABEL_FIG_LEFT,
    ROW_LABEL_X_AXES,
    SSP_COMPARE,
    WINDOW_DAYS,
    _cbar_axes_below_bottom_row,
    _composite_fig_height,
    _finalize_map_stack_layout,
    _grid_height_ratios,
    _place_panel_letters_in_row_gaps,
    _save,
    format_ax,
    get_da,
    load_ssp_datasets,
    plot_on_ax,
    proj,
)

OUTDIR = (
    PROJECT_DIR
    / "Manuscript final plots"
    / "figures"
    / "Fig4_postHH"
    / "PUBLICATION_Fig4_MAIN_CORE"
)

EPOCH_ROW_LABELS = {ep: EPOCH_LABELS[ep] for ep, _, _ in EPOCHS}

# Probability-class inset thresholds (for % land above threshold bar)
PROB_THR_HIGH = 0.5
PROB_THR_MOD = 0.3
PROB_THR_WEAK = 0.1
PROB_CLASS_COLORS = {
    "high": "#b2182b",
    "moderate": "#ef8a62",
    "weak": "#d9d9d9",
    "low": "#92c5de",
}
PROB_CLASS_ORDER = ["high", "moderate", "weak", "low"]

INSET_LON_WEST = 50.0
INSET_LON_EAST = 110.0
INSET_LAT_SOUTH = -44.0
INSET_LAT_NORTH = -38.5

land_mask = xr.open_dataarray(PROJECT_DIR / "land_mask.nc").load()
_lat_weights = xr.DataArray(
    np.cos(np.deg2rad(land_mask.lat.values)),
    dims=["lat"],
    coords={"lat": land_mask.lat},
)


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


def prob_class_fractions(da: xr.DataArray) -> dict[str, float]:
    """Cos(lat)-weighted land-area fractions for P classes (%)."""
    on_land = (land_mask == 1) & np.isfinite(da)
    w = _lat_weights.broadcast_like(da).where(on_land)
    total = float(w.sum())
    if total <= 0:
        return {k: 0.0 for k in PROB_CLASS_COLORS}

    def frac(mask: xr.DataArray) -> float:
        return float(100.0 * w.where(mask).sum() / total)

    return {
        "high": frac(on_land & (da >= PROB_THR_HIGH)),
        "moderate": frac(on_land & (da >= PROB_THR_MOD) & (da < PROB_THR_HIGH)),
        "weak": frac(on_land & (da >= PROB_THR_WEAK) & (da < PROB_THR_MOD)),
        "low": frac(on_land & (da < PROB_THR_WEAK)),
    }


def _inset_rect_geo(ax) -> list[float]:
    ax.figure.canvas.draw()
    (x0, y0) = ax.transData.transform((INSET_LON_WEST, INSET_LAT_SOUTH))
    (x1, y1) = ax.transData.transform((INSET_LON_EAST, INSET_LAT_NORTH))
    inv = ax.transAxes.inverted()
    ax0, ay0 = inv.transform((x0, y0))
    ax1, ay1 = inv.transform((x1, y1))
    return [ax0, ay0, ax1 - ax0, ay1 - ay0]


def add_prob_class_inset(ax, fractions: dict[str, float]) -> None:
    """Compact % land above P thresholds (Fig. 2 inset style)."""
    iax = ax.inset_axes(_inset_rect_geo(ax))
    iax.set_facecolor("none")
    iax.set_zorder(10)
    for spine in iax.spines.values():
        spine.set_visible(False)

    bar_y = 0.06
    bar_h = 0.18
    left = 0.0
    for key in PROB_CLASS_ORDER:
        val = fractions[key]
        if val <= 0:
            continue
        iax.barh(
            bar_y,
            val,
            left=left,
            height=bar_h,
            color=PROB_CLASS_COLORS[key],
            edgecolor="none",
            linewidth=0.0,
        )
        left += val

    iax.set_xlim(0, 106)
    iax.set_ylim(0, 1)
    iax.set_xticks([0, 50, 100])
    iax.set_xticklabels(["0", "50", "100"], fontsize=7.5)
    iax.set_yticks([])
    iax.xaxis.set_ticks_position("top")
    iax.xaxis.set_label_position("top")
    iax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=True,
        labelbottom=False,
        labeltop=True,
        length=1.2,
        width=0.4,
        pad=2.0,
    )
    iax.patch.set_alpha(0.0)
    iax.text(
        101.0,
        bar_y + bar_h / 2.0,
        "%",
        ha="left",
        va="center",
        fontsize=7.5,
        color="0.15",
        clip_on=False,
        zorder=11,
    )


def plot_epochs_by_ssp_grid(
    ds_by_ssp: dict,
    *,
    category: str,
    epochs: list[str],
    out_stem: Path,
    row_labels: bool = True,
    col_titles: bool = True,
    with_insets: bool = True,
    diff_vmax: float = 0.35,
) -> dict[tuple[str, str], dict[str, float]]:
    """
    3×3 grid: rows = epochs (NF/MC/EC), cols = SSP126/245/585.
    Maps show P_baseline for the given successor category.
    """
    nrows, ncols = len(epochs), len(SSP_COMPARE)
    letters = list(string.ascii_lowercase[: nrows * ncols])
    stats: dict[tuple[str, str], dict[str, float]] = {}

    fig = plt.figure(figsize=(ROW_FIG_W, _composite_fig_height(nrows)))
    gs = fig.add_gridspec(
        nrows + 1,
        ncols,
        height_ratios=_grid_height_ratios(nrows),
        hspace=0.0,
        wspace=GS_WSPACE,
    )
    if row_labels:
        fig.subplots_adjust(left=ROW_LABEL_FIG_LEFT)

    im_last = None
    for ri, ep in enumerate(epochs):
        for ci, (ssp_label, ssp_tag) in enumerate(SSP_COMPARE):
            ax = fig.add_subplot(gs[ri, ci], projection=proj)
            da = get_da(
                ds_by_ssp[ssp_tag],
                kind="baseline",
                category=category,
                duration=DURATION,
                epoch=ep,
            )
            im_last = plot_on_ax(
                ax, da, kind="baseline", category=category, diff_vmax=diff_vmax
            )
            format_ax(ax, show_y=(ci == 0), show_x=(ri == nrows - 1))
            if row_labels and ci == 0:
                _add_row_label(ax, EPOCH_ROW_LABELS.get(ep, ep))
            if col_titles and ri == 0:
                ax.set_title(ssp_label, fontsize=9, pad=6)
            if with_insets:
                fr = prob_class_fractions(da)
                stats[(ep, ssp_tag)] = fr
                add_prob_class_inset(ax, fr)

    _finalize_map_stack_layout(fig, n_map_rows=nrows, ncols=ncols)
    _place_panel_letters_in_row_gaps(fig, n_map_rows=nrows, ncols=ncols, letters=letters)

    cax, _ = _cbar_axes_below_bottom_row(fig, n_map_rows=nrows, ncols=ncols, dual=False)
    if im_last is not None:
        cb = fig.colorbar(
            im_last, cax=cax, orientation="horizontal", extend="max", extendrect=False
        )
        cb.set_label(
            r"$P(\mathrm{successor}\;\mathrm{in}\;7\,\mathrm{d}\,|\,\mathrm{HH\;end}\geq 3\,\mathrm{d})$",
            fontsize=9,
            labelpad=2,
        )
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    _save(fig, out_stem)
    plt.close(fig)
    print(f"Wrote {out_stem.name}.png/.pdf")
    return stats


def plot_ec_row_by_ssp(
    ds_by_ssp: dict,
    *,
    category: str,
    epoch: str = "EC",
    out_stem: Path,
    with_insets: bool = True,
    diff_vmax: float = 0.35,
) -> None:
    """Compact 1×3: EC epoch only, cols = SSP126/245/585."""
    letters = ["a", "b", "c"]
    fig = plt.figure(figsize=(ROW_FIG_W, _composite_fig_height(1)))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=_grid_height_ratios(1),
        hspace=0.0,
        wspace=GS_WSPACE,
    )

    im_last = None
    for ci, (ssp_label, ssp_tag) in enumerate(SSP_COMPARE):
        ax = fig.add_subplot(gs[0, ci], projection=proj)
        da = get_da(
            ds_by_ssp[ssp_tag],
            kind="baseline",
            category=category,
            duration=DURATION,
            epoch=epoch,
        )
        im_last = plot_on_ax(
            ax, da, kind="baseline", category=category, diff_vmax=diff_vmax
        )
        format_ax(ax, show_y=(ci == 0), show_x=True)
        ax.set_title(ssp_label, fontsize=9, pad=6)
        ax.text(
            0.0,
            1.02,
            letters[ci],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="0.10",
        )
        if with_insets:
            add_prob_class_inset(ax, prob_class_fractions(da))

    _finalize_map_stack_layout(fig, n_map_rows=1, ncols=3)
    cax, _ = _cbar_axes_below_bottom_row(fig, n_map_rows=1, ncols=3, dual=False)
    if im_last is not None:
        cb = fig.colorbar(
            im_last, cax=cax, orientation="horizontal", extend="max", extendrect=False
        )
        cb.set_label(
            r"$P(\mathrm{successor}\;\mathrm{in}\;7\,\mathrm{d}\,|\,\mathrm{HH\;end}\geq 3\,\mathrm{d})$",
            fontsize=9,
            labelpad=2,
        )
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5, pad=2)

    fig.suptitle(EPOCH_LABELS[epoch], fontsize=10, y=0.995, color="0.15")
    _save(fig, out_stem)
    plt.close(fig)
    print(f"Wrote {out_stem.name}.png/.pdf")


def write_readme(
    *,
    primary_category: str,
    categories_plotted: list[str],
    stats_by_cat: dict[str, dict[tuple[str, str], dict[str, float]]],
) -> None:
    lines = [
        "# Fig. 4 MAIN CORE — Baseline successor probability (3-model ensemble)",
        "",
        "Generated by `plot_fig4_baseline_probability.py`.",
        "",
        "## Main text vs supplementary",
        "",
        "**Main text (this folder):** Continuous **P_base** (baseline successor probability)",
        "after HH run ≥3 d ends — epoch × SSP grid (NF/MC/EC rows, SSP126/245/585 columns).",
        "Clear spatial structure and scenario/epoch gradients; mirrors Fig. 2 emergence layout.",
        "",
        f"**Recommended primary panel:** `Fig4_MAIN_Pbase_{primary_category}_3x3_epochs_SSP_WITH_COLORBAR`",
        "",
        "**Supplementary (SI):** ΔP precondition maps and baseline|preconditioned pairs in",
        "`../PUBLICATION_Fig4_EPOCH_STRATIFIED/` (Fig. 4a precursors, Fig. 4b ΔP compare).",
        "ΔP is scientifically informative but visually sparse (scattered ±3 pp patches).",
        "",
        "## Layout (3×3 full plate)",
        "",
        "|  | SSP126 | SSP245 | SSP585 |",
        "|--|--------|--------|--------|",
        "| **NF (2021–2040)** | a | b | c |",
        "| **MC (2041–2060)** | d | e | f |",
        "| **EC (2081–2100)** | g | h | i |",
        "",
        "Color scale: inferno [0, 1] probability.",
        "Inset bars: cos(lat)-weighted % land in P classes "
        f"(≥{PROB_THR_HIGH}, {PROB_THR_MOD}–{PROB_THR_HIGH}, "
        f"{PROB_THR_WEAK}–{PROB_THR_MOD}, <{PROB_THR_WEAK}).",
        "",
        "## Files",
        "",
    ]
    for cat in categories_plotted:
        lines.append(
            f"- `Fig4_MAIN_Pbase_{cat}_3x3_epochs_SSP_WITH_COLORBAR.png` / `.pdf` — full 3×3 plate"
        )
        lines.append(
            f"- `Fig4_MAIN_Pbase_{cat}_EC_1x3_SSP_WITH_COLORBAR.png` / `.pdf` — compact EC-only row"
        )

    lines.extend(["", "## Metric choice", ""])
    if primary_category == "compound_again":
        lines.extend(
            [
                "**Primary: P_base(compound_again)** — shows tropical hotspot expansion and",
                "strong SSP/epoch amplification (EC SSP585: ~85% land P≥0.3 vs ~26% in NF).",
                "Dry-only P_base is saturated (~90% land P≥0.3 everywhere) and weak for scenario contrast.",
            ]
        )
    else:
        lines.extend(
            [
                "**Primary: P_base(dry_only)** — high baseline dry-spell successor probability;",
                "consider compound_again for clearer scenario/epoch gradients in main text.",
            ]
        )

    lines.extend(["", "## Inset summary (% land P≥0.3, primary category)", ""])
    stats = stats_by_cat.get(primary_category, {})
    for ep, _, _ in EPOCHS:
        row = [f"**{ep}**"]
        for _lbl, ssp_tag in SSP_COMPARE:
            fr = stats.get((ep, ssp_tag), {})
            pct = fr.get("high", 0) + fr.get("moderate", 0)
            row.append(f"{ssp_tag}: {pct:.1f}%")
        lines.append("- " + " | ".join(row))

    (OUTDIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTDIR / 'README.md'}")


def run_main_core(
    *,
    categories: list[str] | None = None,
    with_insets: bool = True,
    primary_category: str = "compound_again",
    min_models: int = 3,
) -> None:
    if categories is None:
        categories = ["compound_again", "dry_only"]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    epochs = [ep for ep, _, _ in EPOCHS]

    print("Loading epoch-stratified SSP126, SSP245, SSP585 ensemble caches...")
    ds_by_ssp = load_ssp_datasets(
        window_days=WINDOW_DAYS,
        duration=DURATION,
        pre_tag=PRE_TAG,
        min_models=min_models,
        epoch_cache=True,
    )

    stats_by_cat: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for cat in categories:
        stem = OUTDIR / f"Fig4_MAIN_Pbase_{cat}_3x3_epochs_SSP_WITH_COLORBAR"
        stats_by_cat[cat] = plot_epochs_by_ssp_grid(
            ds_by_ssp,
            category=cat,
            epochs=epochs,
            out_stem=stem,
            with_insets=with_insets,
        )
        plot_ec_row_by_ssp(
            ds_by_ssp,
            category=cat,
            epoch="EC",
            out_stem=OUTDIR / f"Fig4_MAIN_Pbase_{cat}_EC_1x3_SSP_WITH_COLORBAR",
            with_insets=with_insets,
        )

    write_readme(
        primary_category=primary_category,
        categories_plotted=categories,
        stats_by_cat=stats_by_cat,
    )

    for ds in ds_by_ssp.values():
        ds.close()
    print("Done:", OUTDIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fig. 4 MAIN CORE: P_base epoch×SSP 3×3 plates (continuous probability fields)."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["compound_again", "dry_only"],
        choices=["compound_again", "hot_only", "dry_only"],
        help="Successor categories to plot (default: compound_again dry_only).",
    )
    parser.add_argument(
        "--primary-category",
        default="compound_again",
        choices=["compound_again", "hot_only", "dry_only"],
        help="Category recommended for main text (documented in README).",
    )
    parser.add_argument("--no-insets", action="store_true", help="Skip Fig. 2-style inset bars.")
    parser.add_argument("--min-models", type=int, default=3)
    args = parser.parse_args()

    run_main_core(
        categories=args.categories,
        with_insets=not args.no_insets,
        primary_category=args.primary_category,
        min_models=args.min_models,
    )


if __name__ == "__main__":
    main()
