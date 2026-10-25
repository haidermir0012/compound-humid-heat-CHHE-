#!/usr/bin/env python3
"""
SSP2-4.5 Compound Humid-Heat (FINAL – publication consistent)
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
SSP_DIR = Path("/intelnvme04/haider/raw_data/nexgddp_ssp245_2015_2100/TW_VPD_LANDONLY_SAFE")
OUTPUT_DIR = PROJECT_DIR / "computed_ssp245"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

THRESH_TW = 35.0
THRESH_VPD = 25.0
MIN_BASELINE = 0.5
EPSILON = 1e-6

MODELS = ["EC_Earth3", "HadGEM3_GC31_LL", "MPI-ESM1-2-HR"]

land_mask = xr.open_dataarray(PROJECT_DIR / "land_mask.nc").load()
print("✅ Land mask loaded")

# ─────────────────────────────────────────────
def process_model(model, start, end, base_dir):

    yearly_maps = []

    for yr in tqdm(range(start, end+1), desc=f"{model} {start}-{end}", leave=False):

        if "ssp245" in str(base_dir):
            pattern = f"*{model}*ssp245*{yr}*.nc"
        else:
            pattern = f"*{model}*{yr}*.nc"

        files = list(base_dir.glob(pattern))
        if not files:
            continue

        try:
            ds = xr.open_dataset(files[0])

            tw = ds["tw"]
            if float(tw.max()) > 200:
                tw = tw - 273.15

            vpd = ds["vpd"]

            hh = (tw > THRESH_TW) & (vpd > THRESH_VPD)

            yearly = hh.sum("time").astype("float32")
            yearly_maps.append(yearly)

            ds.close()

        except:
            continue

    if len(yearly_maps) == 0:
        return None

    return xr.concat(yearly_maps, dim="year").mean("year")

# ─────────────────────────────────────────────
def compute_epoch(start, end, base_dir, label):

    model_maps = []

    for model in MODELS:
        result = process_model(model, start, end, base_dir)
        if result is not None:
            model_maps.append(result)

    if len(model_maps) == 0:
        print(f"❌ No data for {label}")
        return None

    # ✔ CORRECT ENSEMBLE MEAN
    ensemble = xr.concat(model_maps, dim="model").mean("model")
    ensemble = ensemble.where(land_mask)

    outfile = OUTPUT_DIR / f"ssp245_{label}_HH_days.nc"
    ensemble.to_netcdf(outfile)

    print(f" ✅ {label}: mean = {float(ensemble.mean()):.3f} days")

    return ensemble

# ─────────────────────────────────────────────
print("Computing SSP2-4.5 epochs...")

epoch_data = {}
for name, start, end, base_dir in [
    ("historical", 1981, 2014, HIST_DIR),
    ("nf", 2021, 2040, SSP_DIR),
    ("mc", 2041, 2060, SSP_DIR),
    ("ec", 2081, 2100, SSP_DIR)
]:
    epoch_data[name] = compute_epoch(start, end, base_dir, name)

# ─────────────────────────────────────────────
print("\nComputing RD and logRR...")

historical = epoch_data["historical"]

for period in ["nf", "mc", "ec"]:

    future = epoch_data[period]
    if future is None:
        continue

    rd = (future - historical).where(land_mask)

    logrr = np.log((future + EPSILON) / (historical + EPSILON))
    logrr = logrr.where(historical >= MIN_BASELINE).where(land_mask)

    rd.to_netcdf(OUTPUT_DIR / f"ssp245_{period}_RD.nc")
    logrr.to_netcdf(OUTPUT_DIR / f"ssp245_{period}_logRR.nc")

    print(f" ✅ {period.upper()}: RD mean = {float(rd.mean()):.2f}, logRR mean = {float(logrr.mean()):.2f}")

print("\n✅ SSP2-4.5 COMPLETE (FINAL, CONSISTENT)")
print(f"Output directory: {OUTPUT_DIR}")
