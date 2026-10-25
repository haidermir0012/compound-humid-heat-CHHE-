# Compound humid-heat extremes — analysis code

Code for the manuscript analyses of compound humid-heat extremes (CHHEs): days when wet-bulb temperature (Tw) and vapour pressure deficit (VPD) are jointly extreme. Three NEX-GDDP CMIP6 models are used: EC-Earth3, HadGEM3-GC31-LL, and MPI-ESM1-2-HR. Scenarios are historical, SSP1-2.6, SSP2-4.5, and SSP5-8.5.

Processed NetCDF grids and the raw NEX-GDDP files are not in this repository. Raw drivers are the public NASA NEX-GDDP-CMIP6 archive. Processed grids are deposited separately.

## What each folder does

| Folder | Role | Script |
|---|---|---|
| `scripts/00_tw_vpd` | Daily Tw and VPD from tasmax, tasmin, and hurs | `compute_tw_vpd_*.py` |
| `scripts/01_event_detection` | Fixed threshold (Tw > 35 °C and VPD > 25 hPa) and local tails (historical Tw p99 and VPD p95) | `02_hpc_event_pipeline_v2_1.py`, `07_hh_percentile_pipeline_FINAL.py` |
| `scripts/02_frequency_trends` | Epoch frequency and risk difference | `compute_ssp*_HH_days_FINAL.py` |
| `scripts/03_persistence` | Conditional probability of a CHHE day 1–30 days after an onset. Main-text lags are 1, 3, and 7. | `09_hh_lag_pipeline_FINAL_FIXED.py` |
| `scripts/04_spells_successors` | Spells of at least 3 days and the 7-day successor state | `16_postHH_preconditioned_precip_FINAL.py` |
| `scripts/05_exposure` | Person-days with population held at the 2015soc field | `20_hh_population_exposure_persondays_FINAL.py` |

Figure scripts in those folders read the processed grids and write the manuscript maps. Map helpers are in `scripts/helpers`.

## Paths

Most scripts use the analysis-server layout. On another machine, set:

```bash
export CHHE_PROJECT_ROOT=/path/to/working_directory
export CHHE_RAW_DATA=/path/to/nex_and_population
export CHHE_LAND_MASK=$CHHE_PROJECT_ROOT/land_mask.nc
```

`scripts/utils/path_config.py` reads those variables. Scripts that still name `/intelnvme04/haider/...` expect that layout, or the same folder names placed at those paths. They do not download data.

## Environment

```bash
conda env create -f ENVIRONMENT.yml
conda activate chhe-repro
```

`requirements.txt` lists the same packages for pip.
