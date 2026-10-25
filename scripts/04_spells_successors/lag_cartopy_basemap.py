"""Land borders from Natural Earth country polygons.

Western Sahara (ADM0_A3 ``SAH``) is merged with Morocco (``MAR``) so the internal
Morocco–Western Sahara line is not drawn, consistent with a Morocco-inclusive map.
Other polygons are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.error import URLError

import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.axes as maxes
from shapely.ops import unary_union


def _admin_countries_shp(resolution: str = "50m") -> Path:
    """Resolve Natural Earth admin_0_countries shapefile (prefer local Cartopy cache)."""
    name = f"ne_{resolution}_admin_0_countries.shp"
    env = os.environ.get("CARTOPY_DATA_DIR")
    roots: list[Path] = []
    if env:
        roots.append(Path(env) / "shapefiles" / "natural_earth" / "cultural")
    roots.append(Path(cartopy.config["data_dir"]) / "shapefiles" / "natural_earth" / "cultural")
    roots.append(Path.home() / ".local" / "share" / "cartopy" / "shapefiles" / "natural_earth" / "cultural")
    for r in roots:
        p = r / name
        if p.is_file():
            return p
    return Path(shpreader.natural_earth(resolution=resolution, category="cultural", name="admin_0_countries"))


def add_land_borders_no_disputed(
    ax: maxes.Axes,
    *,
    resolution: str = "50m",
    edgecolor: str = "black",
    linewidth: float = 0.55,
    zorder: float = 4,
    alpha: float = 1.0,
) -> None:
    try:
        fname = _admin_countries_shp(resolution)
    except URLError:
        return
    reader = shpreader.Reader(str(fname))
    geoms: list = []
    mar = None
    sah = None
    for rec in reader.records():
        a3 = rec.attributes.get("ADM0_A3") or rec.attributes.get("adm0_a3")
        g = rec.geometry
        if g is None or g.is_empty:
            continue
        if a3 == "MAR":
            mar = g
            continue
        if a3 == "SAH":
            sah = g
            continue
        geoms.append(g)

    if mar is not None and sah is not None:
        geoms.append(unary_union([mar, sah]))
    else:
        if mar is not None:
            geoms.append(mar)
        if sah is not None:
            geoms.append(sah)

    ax.add_geometries(
        geoms,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle="-",
        zorder=zorder,
        alpha=alpha,
    )
