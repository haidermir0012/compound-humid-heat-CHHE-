"""Shared diverging colormap for ΔHH (RD) manuscript maps (Fig. 2 style)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

RD_VLIM = 12.0
RD_NEUTRAL_LO = -3.0
RD_NEUTRAL_HI = 3.0
RD_NEUTRAL_GREY = "#f0f0f0"


def rd_hh_cmap_norm(
    vlim: float = RD_VLIM,
    neutral_lo: float = RD_NEUTRAL_LO,
    neutral_hi: float = RD_NEUTRAL_HI,
    grey: str = RD_NEUTRAL_GREY,
) -> tuple[mpl.colors.LinearSegmentedColormap, mpl.colors.Normalize]:
    """
    Blue (negative) | grey neutral band | red (positive).

    Matches RdYlBu_r wings outside ±neutral, with a flat grey plateau in
    [neutral_lo, neutral_hi] so modest changes do not read as yellow noise.
    """
    base = plt.cm.RdYlBu_r
    grey_rgba = mpl.colors.to_rgba(grey)
    blue = base(0.0)
    red = base(1.0)

    p_neg = (neutral_lo + vlim) / (2.0 * vlim)
    p_pos = (neutral_hi + vlim) / (2.0 * vlim)

    cdict = {}
    for ch, idx in zip(("red", "green", "blue", "alpha"), range(4)):
        cdict[ch] = [
            (0.0, blue[idx], blue[idx]),
            (p_neg, grey_rgba[idx], grey_rgba[idx]),
            (p_pos, grey_rgba[idx], grey_rgba[idx]),
            (1.0, red[idx], red[idx]),
        ]

    cmap = mpl.colors.LinearSegmentedColormap("rd_hh_grey_neutral", cdict)
    norm = mpl.colors.Normalize(vmin=-vlim, vmax=vlim)
    return cmap, norm
