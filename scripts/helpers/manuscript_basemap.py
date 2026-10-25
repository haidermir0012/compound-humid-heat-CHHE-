"""Shared basemap helpers for Manuscript final plots (Fig. 2–4 style)."""

from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.axes as maxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from ipcc_regions_overlay import add_ipcc_reference_regions

# Standard manuscript map extent / ticks (Fig. 2–4)
DEFAULT_EXTENT = [-130, 155, -55, 55]
DEFAULT_LON_TICKS = [-70, -10, 50, 110]
DEFAULT_LAT_TICKS = [35, 5, -25, -55]


def format_ax_light(
    ax: maxes.Axes,
    *,
    proj: ccrs.Projection,
    extent: list[float] | None = None,
    lon_ticks: list[float] | None = None,
    lat_ticks: list[float] | None = None,
    show_y: bool,
    show_x: bool,
    show_ipcc: bool = False,
) -> None:
    """
    Fig. 2/3/4 standard basemap: grey land, coastlines only (no country borders).

    Set show_ipcc=True on Fig. 2 only for AR6 WGI land subregion outlines.
    """
    extent = extent or DEFAULT_EXTENT
    lon_ticks = lon_ticks if lon_ticks is not None else DEFAULT_LON_TICKS
    lat_ticks = lat_ticks if lat_ticks is not None else DEFAULT_LAT_TICKS

    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="#e6e6e6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    if show_ipcc:
        add_ipcc_reference_regions(
            ax,
            level="detailed",
            land_only=True,
            linewidth=0.35,
            edgecolor="0.40",
            zorder=4,
            label_regions=False,
        )
    ax.coastlines(linewidth=0.45, color="0.35", zorder=5)
    ax.set_xticks(lon_ticks, crs=proj)
    ax.set_yticks(lat_ticks, crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(direction="out", length=2.5, width=0.6, pad=1.5)
    if not show_y:
        ax.set_yticklabels([])
    if not show_x:
        ax.set_xticklabels([])
