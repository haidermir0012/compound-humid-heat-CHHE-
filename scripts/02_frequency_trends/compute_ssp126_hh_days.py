#!/usr/bin/env python3
"""
SSP1-2.6 Compound Humid-Heat (FINAL CLEAN VERSION)
"""

from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path("/intelnvme04/haider/project/compound_humid_heat")

HIST_DIR = Path("/intelnvme04/haider/raw_data/nexgddp_historical_1981_2014/TW_VPD_LANDONLY_SAFE")
SSP_DIR  = Path("/intelnvme04/haider/raw_data/nexgddp_ssp126_2015_2100/TW_VPD_LANDONLY_SAFE")

OUT_DIR = PROJECT_DIR / "computed_ssp126"
OUT_DIR.mkdir(exist_ok=True, parents=True)

THRESH_TW = 35.0
THRESH_VPD = 25.0
MIN_BASELINE = 0.5
EPS = 1e-6

MODELS = ["EC_Earth3", "HadGEM3_GC31_LL", "MPI-ESM1-2-HR"]

land_mask = xr.open_dataarray(PROJECT_DIR / "land_mask.nc").load()
print("Land mask loaded")

def compute_epoch(start, end, base_dir, label, tag):
    accum = None
    count = 0

    for model in MODELS:
        for yr in tqdm(range(start, end+1), desc=f"{model} {label}", leave=False):

            files = list(base_dir.glob(f"*{model}*{tag}*{yr}*.nc"))
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
        out = OUT_DIR / f"ssp126_{label}_HH_mean.nc"
        mean_map.to_netcdf(out)

        print(f"Saved {label}: mean = {float(mean_map.mean()):.3f} days")
        return mean_map

    print(f"No data for {label}")
    return None

print("\nComputing SSP1-2.6...\n")

epochs = [
    ("historical", 1981, 2014, HIST_DIR, "historical"),
    ("nf", 2021, 2040, SSP_DIR, "ssp126"),
    ("mc", 2041, 2060, SSP_DIR, "ssp126"),
    ("ec", 2081, 2100, SSP_DIR, "ssp126"),
]

data = {}
for name, s, e, path, tag in epochs:
    data[name] = compute_epoch(s, e, path, name, tag)

print("\nComputing RD + logRR...\n")

historical = data["historical"]

for period in ["nf", "mc", "ec"]:
    future = data[period]
    if future is None:
        continue

    rd = (future - historical).where(land_mask)

    logrr = np.log((future + EPS) / (historical + EPS))
    logrr = logrr.where(historical >= MIN_BASELINE).where(land_mask)

    rd.to_netcdf(OUT_DIR / f"ssp126_{period}_RD.nc")
    logrr.to_netcdf(OUT_DIR / f"ssp126_{period}_logRR.nc")

    print(f"{period.upper()} → RD mean = {float(rd.mean()):.2f}, logRR mean = {float(logrr.mean()):.2f}")

print("\nDONE SSP1-2.6")
print("Output:", OUT_DIR)
