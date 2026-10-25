#!/usr/bin/env python3
"""
Compute component exceedance frequency maps from raw NEX TW/VPD files:

  A) TW-only exceedance days/yr:   I[ TW > 35°C ]
  B) VPD-only exceedance days/yr:  I[ VPD > 25 hPa ]

For each SSP (126/245/585) and each epoch:
  historical: 1981–2014
  NF:         2021–2040
  MC:         2041–2060
  EC:         2081–2100

Output is the ensemble mean annual frequency (days/yr) averaged across all model-years
successfully processed (3 models × years, matching the HH pipeline).

Writes NetCDF into:
  Manuscript final plots/computed_component_exceedance/<ssp>/
    - <ssp>_historical_tw35_days_mean.nc
    - <ssp>_nf_tw35_days_mean.nc
    - ... similarly for vpd25
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

PROJECT_DIR = Path("/intelnvme04/haider/project/compound_humid_heat")
RAW = Path("/intelnvme04/haider/raw_data")

HIST_DIR = RAW / "nexgddp_historical_1981_2014" / "TW_VPD_LANDONLY_SAFE"
SSP_DIRS = {
    "ssp126": RAW / "nexgddp_ssp126_2015_2100" / "TW_VPD_LANDONLY_SAFE",
    "ssp245": RAW / "nexgddp_ssp245_2015_2100" / "TW_VPD_LANDONLY_SAFE",
    "ssp585": RAW / "nexgddp_ssp585_2015_2100" / "TW_VPD_LANDONLY_SAFE",
}

OUT_BASE = PROJECT_DIR / "Manuscript final plots" / "computed_component_exceedance"
OUT_BASE.mkdir(parents=True, exist_ok=True)

MODELS = ["EC_Earth3", "HadGEM3_GC31_LL", "MPI-ESM1-2-HR"]

THRESH_TW = 35.0
THRESH_VPD = 25.0

EPOCHS = [
    ("historical", 1981, 2014, "historical"),
    ("nf", 2021, 2040, None),  # scenario tag used per SSP
    ("mc", 2041, 2060, None),
    ("ec", 2081, 2100, None),
]

land_mask = xr.open_dataarray(PROJECT_DIR / "land_mask.nc").load()


def _find_file(base_dir: Path, model: str, tag: str, year: int) -> Path | None:
    # Historical files use mixed tag casing; scenario uses ssp### in filenames but also SSP### in some model strings.
    tag_variants = [tag, tag.upper(), tag.capitalize()]
    for tv in tag_variants:
        files = list(base_dir.glob(f"*{model}*{tv}*{year}*.nc"))
        if files:
            return files[0]
    return None


def compute_epoch_component(
    *,
    start: int,
    end: int,
    base_dir: Path,
    tag: str,
    component: str,
) -> tuple[xr.DataArray, int]:
    """
    Return (mean annual exceedance days/yr map, n_members) where n_members is count of model-years used.
    component in {"tw35", "vpd25"}.
    """
    accum = None
    n = 0
    for model in MODELS:
        for yr in tqdm(range(start, end + 1), desc=f"{component} {model} {start}-{end}", leave=False):
            f = _find_file(base_dir, model, tag, yr)
            if f is None:
                continue
            try:
                ds = xr.open_dataset(f)
                tw = ds["tw"]
                if float(tw.max()) > 200:
                    tw = tw - 273.15
                vpd = ds["vpd"]
                if component == "tw35":
                    ex = (tw > THRESH_TW)
                elif component == "vpd25":
                    ex = (vpd > THRESH_VPD)
                else:
                    raise ValueError("component must be 'tw35' or 'vpd25'")
                days = ex.sum("time").astype("float32")
                accum = days if accum is None else (accum + days)
                n += 1
            except Exception:
                pass
            finally:
                try:
                    ds.close()
                except Exception:
                    pass
    if n == 0:
        raise RuntimeError(f"No data processed for component={component} {start}-{end} tag={tag}")
    mean = (accum / n).where(land_mask).astype("float32")
    return mean, n


def run_ssp(ssp: str) -> None:
    if ssp not in SSP_DIRS:
        raise ValueError(f"Unknown SSP {ssp}")
    out_dir = OUT_BASE / ssp
    out_dir.mkdir(parents=True, exist_ok=True)

    for component in ["tw35", "vpd25"]:
        for label, y0, y1, tag in EPOCHS:
            base = HIST_DIR if label == "historical" else SSP_DIRS[ssp]
            tag2 = tag if tag is not None else ssp
            out = out_dir / f"{ssp}_{label}_{component}_days_mean.nc"
            if out.exists():
                print(f"[skip] exists: {out.name}")
                continue
            mean, n = compute_epoch_component(
                start=y0,
                end=y1,
                base_dir=base,
                tag=tag2,
                component=component,
            )
            mean.to_netcdf(out)
            print(f"[write] {out.name} | members={n} | mean={float(mean.mean()):.3f}")


def main() -> None:
    for ssp in ["ssp126", "ssp245", "ssp585"]:
        print(f"\n=== {ssp} ===")
        run_ssp(ssp)


if __name__ == "__main__":
    main()

