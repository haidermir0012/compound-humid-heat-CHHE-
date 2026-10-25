#!/usr/bin/env python3

from pathlib import Path
import xarray as xr
import numpy as np
import json
from tqdm import tqdm

# =========================
# CONFIG
# =========================
RAW = Path("/intelnvme04/haider/raw_data")

BASE = Path("/intelnvme04/haider/project/compound_humid_heat")
OUT = BASE / "computed_events_v2"
LOG = BASE / "logs"
PROG = BASE / "progress"

OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
PROG.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["historical", "ssp126", "ssp245", "ssp585"]
MODELS = ["EC_Earth3", "HadGEM3_GC31_LL", "MPI-ESM1-2-HR"]

SCEN_DIR_MAP = {
    "historical": "nexgddp_historical_1981_2014",
    "ssp126": "nexgddp_ssp126_2015_2100",
    "ssp245": "nexgddp_ssp245_2015_2100",
    "ssp585": "nexgddp_ssp585_2015_2100"
}

TW_C_THRESHOLD = 35.0
VPD_THRESHOLD = 25.0

LOGFILE = LOG / "run_v2_1.log"
STATUS = PROG / "status.json"

# =========================
# LOGGING & STATUS
# =========================
def log(msg):
    print(msg)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")

def load_status():
    return json.load(open(STATUS)) if STATUS.exists() else {}

def save_status(data):
    json.dump(data, open(STATUS, "w"), indent=2)

# =========================
# FILE MATCHING (SAFER)
# =========================
def get_files(scen, model):
    base = RAW / SCEN_DIR_MAP[scen] / "TW_VPD_LANDONLY_SAFE"
    if not base.exists():
        return []
    # ✅ Safer: exact prefix match instead of substring
    return sorted([f for f in base.iterdir() if f.name.startswith(model) and f.suffix == ".nc"])

# =========================
# PROCESS FILE (HPC-OPTIMIZED)
# =========================
def process(fp, outpath):
    # ✅ Only chunk along time. Let Dask respect native lat/lon layout.
    ds = xr.open_dataset(fp, chunks={"time": 365})

    tw = ds["tw"]
    # ✅ Scientific safeguard: convert to °C ONLY if data is in Kelvin
    if float(tw.max().compute()) > 200:
        tw = tw - 273.15

    vpd = ds["vpd"]

    HH = (tw > TW_C_THRESHOLD) & (vpd > VPD_THRESHOLD)
    # ✅ Standard onset detection. Note: assumes continuous daily calendar (true for NEX-GDDP)
    HH_start = HH & (~HH.shift(time=1, fill_value=False))

    out = xr.Dataset({
        "HH": HH.astype("int8"),
        "HH_start": HH_start.astype("int8")
    })

    # ✅ Dask streams lazily to disk. No .compute() → prevents RAM explosion
    out.to_netcdf(outpath, format="NETCDF4", engine="netcdf4")
    ds.close()

# =========================
# MAIN
# =========================
def main():
    log("\n==============================")
    log("🚀 HH PIPELINE v2.1 (PRODUCTION) STARTED")
    log("==============================\n")

    status = load_status()

    total_files = sum(len(get_files(s, m)) for s in SCENARIOS for m in MODELS)
    processed = 0

    for scen in SCENARIOS:
        log(f"\n📁 Scenario: {scen}")
        for model in MODELS:
            files = get_files(scen, model)
            if not files:
                continue

            key = f"{scen}_{model}"
            done = status.get(key, {})

            log(f"\n🔹 Model: {model} ({len(files)} files)")

            outdir = OUT / scen / model
            outdir.mkdir(parents=True, exist_ok=True)

            for fp in tqdm(files, desc=f"{scen}-{model}", unit="file"):
                name = fp.name
                if done.get(name):
                    continue

                outpath = outdir / name
                try:
                    process(fp, outpath)
                    done[name] = True
                    status[key] = done
                    save_status(status)
                    processed += 1
                except Exception as e:
                    log(f"❌ ERROR {name}: {e}")

                if processed % 5 == 0:
                    log(f"✔ Progress: {processed}/{total_files}")

    log("\n✅ PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
