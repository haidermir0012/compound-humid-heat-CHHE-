"""Central path configuration for the reproducibility deposit.

Override via environment variables (preferred for reviewers):
  CHHE_PROJECT_ROOT  — project / working directory (outputs land here)
  CHHE_RAW_DATA      — NEX-GDDP + population + Tw/VPD inputs
  CHHE_LAND_MASK     — optional path to land_mask.nc

Defaults match the author's HPC layout but are not required if env vars are set.
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "CHHE_PROJECT_ROOT",
    "/intelnvme04/haider/project/compound_humid_heat",
)).resolve()

RAW_DATA = Path(os.environ.get(
    "CHHE_RAW_DATA",
    "/intelnvme04/haider/raw_data",
)).resolve()

LAND_MASK = Path(os.environ.get(
    "CHHE_LAND_MASK",
    str(PROJECT_ROOT / "land_mask.nc"),
)).resolve()

MANUSCRIPT_PLOTS = PROJECT_ROOT / "Manuscript final plots"

SCEN_DIR = {
    "historical": "nexgddp_historical_1981_2014",
    "ssp126": "nexgddp_ssp126_2015_2100",
    "ssp245": "nexgddp_ssp245_2015_2100",
    "ssp585": "nexgddp_ssp585_2015_2100",
}

MODELS = ["EC_Earth3", "HadGEM3-GC31-LL", "MPI-ESM1-2-HR"]
# Filename variants used in older folders
MODEL_FILE_ALIASES = {
    "EC_Earth3": ["EC_Earth3", "EC-Earth3"],
    "HadGEM3-GC31-LL": ["HadGEM3-GC31-LL", "HadGEM3_GC31_LL"],
    "MPI-ESM1-2-HR": ["MPI-ESM1-2-HR", "MPI_ESM1_2_HR"],
}

TW_C_THRESHOLD = 35.0
VPD_THRESHOLD = 25.0  # hPa

def tw_vpd_dir(scenario: str) -> Path:
    return RAW_DATA / SCEN_DIR[scenario] / "TW_VPD_LANDONLY_SAFE"
