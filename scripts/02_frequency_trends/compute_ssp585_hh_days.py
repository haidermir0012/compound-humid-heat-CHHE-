#!/usr/bin/env python3
"""
SSP5-8.5 FINAL COMPUTATION (MATCHED METHOD)
TW > 35°C & VPD > 25 hPa
"""

from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path("/intelnvme04/haider/project/compound_humid_heat")

HIST_DIR = Path("/intelnvme04/haider/raw_data/nexgddp_historical_1981_2014/TW_VPD_LANDONLY_SAFE")
SSP_DIR  = Path("/intelnvme04/haider/raw_data/nexgddp_ssp585_2015_2100/TW_VPD_LANDONLY_SAFE")

OUT_DIR = PROJECT_DIR / "computed_ssp585"
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODELS = ["EC_Earth3", "HadGEM3_GC31_LL", "MPI-ESM1-2-HR"]

THRESH_TW = 35.0
THRESH_VPD = 25.0

land_mask = xr.open_dataarray(PROJECT_DIR / "land_mask.nc").load()
print("✅ Land mask loaded")

def compute_epoch(start, end, base_dir, label, scenario_tag):
    accum = None
    count = 0

    for model in MODELS:
        for yr in tqdm(range(start, end+1), desc=f"{model} {label}", leave=False):

            if scenario_tag == "historical":
                files = list(base_dir.glob(f"*{model}*Historical*{yr}*.nc"))
            else:
                files = list(base_dir.glob(f"*{model}*ssp585*{yr}*.nc"))

            if not files:
                continue

            try:
                ds = xr.open_dataset(files[0])

                tw = ds["tw"]
                if float(tw.max()) > 200:
                    tw = tw - 273.15

                hh = (tw > THRESH_TW) & (ds["vpd"] > THRESH_VPD)

                days = hh.sum("time").astype("float32")

                accum = days if accum is None else accum + days
                count += 1

                ds.close()

            except Exception:
                continue

    if count > 0:
        mean_map = (accum / count).where(land_mask)
        out = OUT_DIR / f"ssp585_{label}_HH_mean.nc"
        mean_map.to_netcdf(out)

        print(f"✅ Saved {label}: mean = {float(mean_map.mean()):.3f}")
        return mean_map

    print(f"❌ No data for {label}")
    return None

print("\nComputing SSP5-8.5...\n")

hist = compute_epoch(1981, 2014, HIST_DIR, "historical", "historical")
nf   = compute_epoch(2021, 2040, SSP_DIR, "nf", "ssp585")
mc   = compute_epoch(2041, 2060, SSP_DIR, "mc", "ssp585")
ec   = compute_epoch(2081, 2100, SSP_DIR, "ec", "ssp585")

print("\n✅ DONE SSP585 COMPUTATION")
print("Output:", OUT_DIR)
