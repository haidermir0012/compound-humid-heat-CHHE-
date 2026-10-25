#!/usr/bin/env python3
"""
Build 3-model post-HH precondition ensemble without re-running EC-Earth3.

1) Load existing 1-model EC cache (skip recompute).
2) Compute per-model fields for MPI-ESM1-2-HR and HadGEM3-GC31-LL only.
3) Ensemble mean: (EC + MPI + HadGEM) / 3
4) Write new cache + optional --replot

Uses functions from compute_post_hh_successors.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_DIR = Path("/intelnvme04/haider/project/compound_humid_heat")
SCRIPTS = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "cache_postHH_precondition_FINAL"

EC_CACHE = CACHE_DIR / (
    "postHH_precond_historical_win7_dur3_catcompound_again-hot_only-dry_only-non_extreme_"
    "modsEC_Earth3_predry_spell_L14_wet1_acc5_twQ0.99_vpdQ0.95_dryQ0.95.nc"
)

MODELS_NEW = ["MPI-ESM1-2-HR", "HadGEM3-GC31-LL"]
MODELS_ALL = ["EC_Earth3", "MPI-ESM1-2-HR", "HadGEM3-GC31-LL"]

SCENARIO = "historical"
WINDOW_DAYS = 7
DURATIONS = [3]
CATEGORIES = ["compound_again", "hot_only", "dry_only", "non_extreme"]
PRE_KIND = "dry_spell"
PRE_L = 14
WET_DAY_MM = 1.0
ACCUM_MM = 5.0
TW_Q, VPD_Q, DRY_Q = 0.99, 0.95, 0.95


def _load_s16():
    path = SCRIPTS / "compute_post_hh_successors.py"
    spec = importlib.util.spec_from_file_location("posthh16", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_one_model(s16, model: str) -> tuple[dict, dict]:
    """Return (baseline_probs, precond_probs) per duration/category as numpy arrays."""
    scen = SCENARIO
    tw_files = s16.find_tw_vpd_files(scen, model)
    root = s16.find_model_data_root(scen, model)
    if not tw_files:
        raise FileNotFoundError(f"{model}: no TW/VPD files")
    if root is None:
        raise FileNotFoundError(f"{model}: no precipitation directory under nexgddp_historical")

    print(f"\n--- {model} ---")
    print(f"  TW/VPD: {len(tw_files)} files")
    print(f"  pr root: {root.name}")

    pr_raw, pr_ds = s16.open_pr_mmday(root)
    ds = s16.open_tw_vpd_stack(tw_files)
    try:
        tw_c = s16.ensure_tw_celsius(ds["tw"])
        tw_a, pr_a = xr.align(tw_c, pr_raw, join="inner")
        vpd_a = ds["vpd"].sel(time=tw_a.time)
        pre = s16.precondition_mask(
            pr_a,
            kind=PRE_KIND,
            L=PRE_L,
            wet_day_mm=WET_DAY_MM,
            accum_mm=ACCUM_MM,
        )
        tw_thr, vpd_thr, vpd_dry_thr = s16.compute_thresholds(
            tw_a,
            vpd_a,
            tw_quantile=TW_Q,
            vpd_quantile=VPD_Q,
            vpd_dry_quantile=DRY_Q,
        )
        pb, pp = s16.compute_baseline_and_preconditioned_probs(
            tw_a,
            vpd_a,
            pre,
            tw_thr=tw_thr,
            vpd_thr=vpd_thr,
            vpd_dry_thr=vpd_dry_thr,
            durations=DURATIONS,
            window_days=WINDOW_DAYS,
            categories=CATEGORIES,
        )
        out_b, out_p = {}, {}
        for k in DURATIONS:
            out_b[k], out_p[k] = {}, {}
            for cat in CATEGORIES:
                out_b[k][cat] = pb[k][cat].compute().values.astype("float32")
                out_p[k][cat] = pp[k][cat].compute().values.astype("float32")
        return out_b, out_p
    finally:
        ds.close()
        pr_ds.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ec-cache", type=Path, default=EC_CACHE)
    p.add_argument("--models", nargs="*", default=MODELS_NEW)
    p.add_argument("--replot", action="store_true", help="Run plot_fig4_successors.py after cache write")
    args = p.parse_args()

    ec_path = Path(args.ec_cache)
    if not ec_path.exists():
        raise FileNotFoundError(f"EC cache missing: {ec_path}")

    with xr.open_dataset(ec_path) as ec:
        n_ec = int(ec.attrs.get("n_models", 1))
        if n_ec != 1:
            print(f"Warning: EC cache n_models={n_ec} (expected 1)")
        lat = ec["lat"]
        lon = ec["lon"]
        ec_b = ec["P_baseline"].load()
        ec_p = ec["P_preconditioned"].load()

    s16 = _load_s16()
    per_b: list[np.ndarray] = []
    per_p: list[np.ndarray] = []

    # EC contribution (already in cache)
    per_b.append(
        np.stack(
            [[ec_b.sel(duration=k, category=c).values for c in CATEGORIES] for k in DURATIONS],
            axis=0,
        )
    )
    per_p.append(
        np.stack(
            [[ec_p.sel(duration=k, category=c).values for c in CATEGORIES] for k in DURATIONS],
            axis=0,
        )
    )
    print(f"Loaded EC from {ec_path.name}")

    for model in args.models:
        pb, pp = compute_one_model(s16, model)
        per_b.append(
            np.stack([[pb[k][c] for c in CATEGORIES] for k in DURATIONS], axis=0).astype("float32")
        )
        per_p.append(
            np.stack([[pp[k][c] for c in CATEGORIES] for k in DURATIONS], axis=0).astype("float32")
        )

    n = len(per_b)
    stack_b = np.stack(per_b, axis=0).mean(axis=0)
    stack_p = np.stack(per_p, axis=0).mean(axis=0)
    stack_d = stack_p - stack_b

    out_fp = s16.cache_path(
        cache_dir=CACHE_DIR,
        scen=SCENARIO,
        models=MODELS_ALL,
        durations=DURATIONS,
        window_days=WINDOW_DAYS,
        categories=CATEGORIES,
        pre_kind=PRE_KIND,
        pre_L=PRE_L,
        wet_day_mm=WET_DAY_MM,
        accum_mm=ACCUM_MM,
        tw_quantile=TW_Q,
        vpd_quantile=VPD_Q,
        vpd_dry_quantile=DRY_Q,
    )

    ds_out = xr.Dataset(
        {
            "P_baseline": (("duration", "category", "lat", "lon"), stack_b),
            "P_preconditioned": (("duration", "category", "lat", "lon"), stack_p),
            "P_diff": (("duration", "category", "lat", "lon"), stack_d.astype("float32")),
        },
        coords={
            "duration": np.array(DURATIONS, dtype=np.int32),
            "category": CATEGORIES,
            "lat": lat,
            "lon": lon,
        },
    )
    ds_out.attrs.update(
        scenario=SCENARIO,
        window_days=WINDOW_DAYS,
        precondition_kind=PRE_KIND,
        precondition_L=PRE_L,
        wet_day_mm=WET_DAY_MM,
        accum_mm=ACCUM_MM,
        models=",".join(MODELS_ALL),
        n_models=len(MODELS_ALL),
        merge_note="EC from cache; new models computed in merge_post_hh_ensemble.py",
    )

    print(f"\nWrite 3-model cache ({n} members averaged): {out_fp}")
    s16.write_cache(ds_out, out_fp)
    ds_out.close()

    if args.replot:
        plot_script = Path(__file__).resolve().parent / "plot_fig4_successors.py"
        print(f"Replot: {plot_script}")
        import subprocess

        subprocess.run([sys.executable, str(plot_script), "--min-models", "3"], check=True)

    print("Done.")


if __name__ == "__main__":
    main()
