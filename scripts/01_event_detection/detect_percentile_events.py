#!/usr/bin/env python3
"""
FINAL STABLE HH PIPELINE (HPC SAFE + PUBLICATION CONSISTENT)

- Percentile-based HH definition
- No LocalCluster
- No manual numpy percentile (avoids memory explosion)
- Chunk-safe quantile along time axis
- tqdm progress per model
- resume support via status.json
"""

from pathlib import Path
import xarray as xr
import numpy as np
import json
from tqdm import tqdm

# =========================
# PATHS
# =========================
RAW = Path("/intelnvme04/haider/raw_data")

BASE = Path("/intelnvme04/haider/project/compound_humid_heat")
OUT = BASE / "percentile_HH_FINAL"
LOG = BASE / "logs"
PROG = BASE / "progress"

OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
PROG.mkdir(parents=True, exist_ok=True)

LOGFILE = LOG / "hh_final.log"
STATUS = PROG / "hh_final.json"

# =========================
SCENARIOS = ["historical", "ssp126", "ssp245", "ssp585"]
MODELS = ["EC_Earth3", "HadGEM3_GC31_LL", "MPI-ESM1-2-HR"]

SCEN_DIR = {
    "historical": "nexgddp_historical_1981_2014",
    "ssp126": "nexgddp_ssp126_2015_2100",
    "ssp245": "nexgddp_ssp245_2015_2100",
    "ssp585": "nexgddp_ssp585_2015_2100"
}

# =========================
def log(msg):
    print(msg)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")

def load_status():
    if STATUS.exists():
        return json.load(open(STATUS))
    return {}

def save_status(s):
    json.dump(s, open(STATUS, "w"), indent=2)

# =========================
def get_files(scen, model):
    base = RAW / SCEN_DIR[scen] / "TW_VPD_LANDONLY_SAFE"
    if not base.exists():
        return []
    return sorted([f for f in base.iterdir()
                   if model in f.name and f.suffix == ".nc"])

# =========================
def open_stack(files):
    return xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks={"time": 365}
    )

# =========================
def compute_thresholds(tw, vpd):
    """
    ONLY SAFE HPC METHOD:
    xarray quantile (chunk-aware, dask-backed)
    """

    tw = tw.transpose("time", "lat", "lon")
    vpd = vpd.transpose("time", "lat", "lon")

    # CRITICAL: avoid rechunk explosion
    tw = tw.chunk({"time": 365})
    vpd = vpd.chunk({"time": 365})

    tw_thr = tw.quantile(0.99, dim="time")
    vpd_thr = vpd.quantile(0.95, dim="time")

    return tw_thr, vpd_thr

# =========================
def process(scen, model):
    files = get_files(scen, model)
    if not files:
        return

    log(f"\n========== {scen} | {model} ==========")

    ds = open_stack(files)

    tw = ds["tw"]
    vpd = ds["vpd"]

    # unit safety
    if float(tw.max().compute()) > 200:
        tw = tw - 273.15

    log("📊 computing percentile thresholds...")
    tw_thr, vpd_thr = compute_thresholds(tw, vpd)

    log("📊 computing HH events...")

    HH = (tw > tw_thr) & (vpd > vpd_thr)
    HH_start = HH & (~HH.shift(time=1, fill_value=False))

    out = xr.Dataset({
        "HH": HH.astype("int8"),
        "HH_start": HH_start.astype("int8")
    })

    out_dir = OUT / scen / model
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "hh_percentile.nc"

    out.to_netcdf(out_file)

    ds.close()

# =========================
def main():
    log("\n🚀 HH PERCENTILE PIPELINE FINAL STARTED")
    log("=" * 60)

    status = load_status()

    for scen in SCENARIOS:
        log(f"\n========== SCENARIO: {scen} ==========")

        for model in tqdm(MODELS, desc=scen):

            key = f"{scen}_{model}"
            if status.get(key):
                continue

            try:
                process(scen, model)
                status[key] = True
                save_status(status)

            except Exception as e:
                log(f"❌ ERROR {scen}-{model}: {e}")

    log("\n✅ ALL DONE")

if __name__ == "__main__":
    main()
