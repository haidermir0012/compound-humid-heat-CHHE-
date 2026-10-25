"""IPCC AR6 WGI reference region outlines for Cartopy maps (v4, Iturbide et al. 2020)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cartopy.crs as ccrs
from cartopy.io import shapereader
from shapely.ops import unary_union

# Downloaded from https://github.com/IPCC-WG1/Atlas/tree/main/reference-regions
IPCC_REF_SHP = Path(
    "/intelnvme04/haider/plots/ipcc_ar6_reference_regions/IPCC-WGI-reference-regions-v4.shp"
)

_CONTINENT_LABEL = {
    "AFRICA": "Africa",
    "ASIA": "Asia",
    "CENTRAL-AMERICA": "Central America",
    "EUROPE": "Europe",
    "NORTH-AMERICA": "N. America",
    "OCEANIA": "Oceania",
    "POLAR": "Polar",
    "SOUTH-AMERICA": "S. America",
}


def _collect_geometries(
    reader,
    *,
    land_only: bool,
    ocean_only: bool,
    level: str,
) -> tuple[list, list[tuple[float, float, str]]]:
    """level: 'detailed' (46 land subregions) | 'continental' (8 merged land domains)."""
    geoms: list = []
    labels: list[tuple[float, float, str]] = []

    if level == "continental":
        by_continent: dict[str, list] = defaultdict(list)
        for rec in reader.records():
            at = rec.attributes
            if at.get("Type", "") != "Land":
                continue
            by_continent[at.get("Continent", "UNKNOWN")].append(rec.geometry)

        for cont, parts in sorted(by_continent.items()):
            merged = unary_union(parts)
            geoms.append(merged)
            if not merged.is_empty:
                c = merged.centroid
                labels.append((c.x, c.y, _CONTINENT_LABEL.get(cont, cont)))
        return geoms, labels

    for rec in reader.records():
        at = rec.attributes
        rtype = at.get("Type", "")
        if land_only and rtype != "Land":
            continue
        if ocean_only and rtype != "Ocean":
            continue
        geoms.append(rec.geometry)
        if level == "detailed":
            acr = at.get("Acronym", "")
            if acr:
                g = rec.geometry
                labels.append((g.centroid.x, g.centroid.y, acr))

    return geoms, labels


def add_ipcc_reference_regions(
    ax,
    *,
    shp_path: Path | str = IPCC_REF_SHP,
    land_only: bool = True,
    ocean_only: bool = False,
    level: str = "detailed",
    edgecolor: str = "0.40",
    linewidth: float = 0.35,
    linestyle: str = "-",
    alpha: float = 0.9,
    zorder: int = 4,
    label_regions: bool = False,
    label_fontsize: float = 7,
) -> int:
    """
    Draw IPCC AR6 WGI reference region boundaries on a Cartopy GeoAxes.

    level
        ``continental`` — 8 merged land domains
        ``detailed``    — all 46 AR6 land subregions (default for manuscript)

    Returns number of regions drawn.
    """
    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(
            f"IPCC reference regions shapefile not found: {shp_path}\n"
            "Download: https://github.com/IPCC-WG1/Atlas/tree/main/reference-regions"
        )

    level = level.lower().strip()
    if level not in ("continental", "detailed"):
        raise ValueError("level must be 'continental' or 'detailed'")

    reader = shapereader.Reader(str(shp_path))
    geoms, labels = _collect_geometries(
        reader,
        land_only=land_only,
        ocean_only=ocean_only,
        level=level,
    )

    if not geoms:
        return 0

    lw = linewidth + 0.15 if level == "continental" else linewidth
    # add_geometries (not add_feature) so outlines draw above pcolormesh data layers.
    ax.add_geometries(
        geoms,
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )

    if label_regions:
        for lon, lat, txt in labels:
            ax.text(
                lon,
                lat,
                txt,
                transform=ccrs.PlateCarree(),
                fontsize=label_fontsize,
                ha="center",
                va="center",
                color=edgecolor,
                clip_on=True,
                zorder=zorder + 1,
            )

    return len(geoms)
