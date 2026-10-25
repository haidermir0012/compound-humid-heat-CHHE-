#!/usr/bin/env python3
"""
HH LAG PIPELINE (HPC-SAFE)

Computes lag curves:
  P(HH(t+?)=1 | HH_start(t)=1)  for ?=1..MAX_LAG

Notes:
- Uses HH_start derived from HH (independent starts).
- Skips work already done if output exists and is non-empty.
- Writes one NetCDF per (scenario, model).
"""

from __future__ import annotations

import json
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE = Path("/intelnvme04/haider/project/compound_humid_heat")

INPUT = BASE / "percentile_HH_FINAL"
OUT = BASE / "lag_analysis_FINAL"
LOG = BASE / "logs"
PROG = BASE / "progress"

OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
PROG.mkdir(parents=True, exist_ok=True)

LOGFILE = LOG / "hh_lag.log"
STATUS = PROG / "hh_lag_status.json"

SCENARIOS = ["historical", "ssp126", "ssp245", "ssp585"]
MODELS = ["EC_Earth3", "HadGEM3-GC31-LL", "MPI-ESM1-2-HR"]

MAX_LAG = 30

# Chunking: spatial tiling is critical to avoid OOM
CHUNKS = {"time": 365, "lat": 50, "lon": 120}


# =========================
# LOGGING / STATUS
# =========================
def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")


def load_status() -> dict:
    if STATUS.exists():
        return json.load(open(STATUS))
    return {}


def save_status(s: dict) -> None:
    json.dump(s, open(STATUS, "w"), indent=2)


# =========================
# IO HELPERS
# =========================
def _model_variants(model: str) -> list[str]:
    return list(dict.fromkeys([model, model.replace("-", "_"), model.replace("_", "-")]))


def input_file(scen: str, model: str) -> Path | None:
    for m in _model_variants(model):
        f = INPUT / scen / m / "hh_percentile.nc"
        if f.exists():
            return f
    return None


def out_file(scen: str, model: str) -> Path:
    # Always write to canonical model name folder
    return OUT / f"lag_HH_{scen}_{model}.nc"


def load_case(scen: str, model: str) -> xr.Dataset | None:
    f = input_file(scen, model)
    if f is None:
        return None
    return xr.open_dataset(f, chunks=CHUNKS)


# =========================
# CORE LAG FUNCTION
# =========================
def compute_lag_curve(HH_start: xr.DataArray, HH: xr.DataArray, max_lag: int) -> xr.DataArray:
    """
    Returns DataArray(lag, lat, lon): P(HH(t+?)=1 | HH_start(t)=1)
    """
    HH_start = HH_start.fillna(0).astype("int8")
    HH = HH.fillna(0).astype("int8")

    # independent events per grid cell
    denom = HH_start.sum(dim="time")

    lag_probs: list[xr.DataArray] = []
    for tau in tqdm(range(1, max_lag + 1), desc="Computing lags"):
        future = HH.shift(time=-tau).fillna(0).astype("int8")
        hits = (HH_start * future).sum(dim="time")
        prob = xr.where(denom > 0, hits / denom, np.nan).astype("float32")
        lag_probs.append(prob)

    out = xr.concat(lag_probs, dim="lag").assign_coords(lag=np.arange(1, max_lag + 1))
    out.name = "P_HH_given_HHstart"
    return out


# =========================
# PROCESS SINGLE CASE
# =========================
def process(scen: str, model: str) -> bool:
    log(f"\n?? Loading: {scen} | {model}")

    ds = load_case(scen, model)
    if ds is None:
        log("? missing input file")
        return False

    HH = ds["HH"].astype("int8")

    # Strict starts: HH(t)=1 and HH(t-1)=0
    HH_start = (HH == 1) & (HH.shift(time=1, fill_value=0) == 0)
    HH_start = HH_start.astype("int8")

    log("?? Computing lag curve...")
    lag_curve = compute_lag_curve(HH_start, HH, MAX_LAG)

    out = xr.Dataset({"P_HH_given_HHstart": lag_curve})
    target = out_file(scen, model)
    log(f"?? Saving: {target}")

    encoding = {
        "P_HH_given_HHstart": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }
    }

    with dask.config.set(scheduler="threads"):
        out.to_netcdf(target, engine="netcdf4", format="NETCDF4", encoding=encoding)

    ds.close()

    ok = target.exists() and target.stat().st_size > 1000
    log("   ? Success" if ok else "   ? Write failed/empty")
    return ok


# =========================
# MAIN
# =========================
def main() -> None:
    log("\n?? HH LAG PIPELINE STARTED")
    status = load_status()

    for model in MODELS:
        for scen in SCENARIOS:
            key = f"{scen}_{model}"
            target = out_file(scen, model)

            if status.get(key) and target.exists() and target.stat().st_size > 1000:
                continue
            if status.get(key) and not target.exists():
                log(f"?? Status says done but output missing: {key}. Re-running.")

            try:
                ok = process(scen, model)
                if ok:
                    status[key] = True
                    save_status(status)
            except Exception as e:
                log(f"? ERROR {scen}-{model}: {e}")

    log("\n? ALL DONE")


if __name__ == "__main__":
    main()

