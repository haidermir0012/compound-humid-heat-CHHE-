#!/usr/bin/env python3
"""
Cache-only: 3-model post-HH precondition ensemble for SSP scenarios.

Computes P_baseline, P_preconditioned, P_diff (duration=3) and writes NetCDF
to cache_postHH_precondition_FINAL/ — no figure exports.

Historical is skipped (already done). Default: ssp126, ssp245, ssp585.
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
RAW = Path("/intelnvme04/haider/raw_data")

MODELS = ["EC_Earth3", "MPI-ESM1-2-HR", "HadGEM3-GC31-LL"]
DEFAULT_SCENARIOS = ["ssp126", "ssp245", "ssp585"]

WINDOW_DAYS = 7
DURATIONS = [3]
CATEGORIES = ["compound_again", "hot_only", "dry_only", "non_extreme"]
PRE_KIND = "dry_spell"
PRE_L = 14
WET_DAY_MM = 1.0
ACCUM_MM = 5.0
TW_Q, VPD_Q, DRY_Q = 0.99, 0.95, 0.95

# EC-Earth3 SSP126 precipitation lives under a non-standard folder layout.
PR_ROOT_OVERRIDE: dict[tuple[str, str], Path] = {
    (
        "ssp126",
        "EC_Earth3",
    ): RAW
    / "nexgddp_ssp126_2015_2100"
    / "EC_Earth_ssp126_2015_2100_merged"
    / "pr_ssp126_raw_ssp_26_nexgddp_v2",
}


def _load_s16():
    path = SCRIPTS / "compute_post_hh_successors.py"
    spec = importlib.util.spec_from_file_location("posthh16", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _open_pr(s16, root: Path) -> tuple[xr.DataArray, xr.Dataset]:
    try:
        return s16.open_pr_mmday(root)
    except FileNotFoundError:
        files = sorted(root.glob("pr*.nc"))
        if not files:
            raise FileNotFoundError(f"No pr*.nc under {root}")
        ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 365}, engine="netcdf4")
        pr = ds["pr"] * 86400.0
        pr = pr.astype("float32")
        pr.name = "pr_mmday"
        return pr, ds


def _pr_root(s16, scen: str, model: str) -> Path | None:
    key = (scen, model)
    if key in PR_ROOT_OVERRIDE:
        p = PR_ROOT_OVERRIDE[key]
        return p if p.exists() else None
    return s16.find_model_data_root(scen, model)


def _expected_cache(s16, scen: str) -> Path:
    return s16.cache_path(
        cache_dir=CACHE_DIR,
        scen=scen,
        models=MODELS,
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


def compute_scenario(s16, scen: str, *, force: bool) -> Path | None:
    cache_fp = _expected_cache(s16, scen)
    if cache_fp.exists() and not force:
        with xr.open_dataset(cache_fp) as ds:
            n = int(ds.attrs.get("n_models", 0))
        print(f"[skip] {scen}: cache exists (n_models={n}): {cache_fp.name}")
        return cache_fp

    print(f"\n{'=' * 60}\nScenario: {scen}\n{'=' * 60}")
    ens_pb: dict[int, dict[str, np.ndarray | None]] = {k: {c: None for c in CATEGORIES} for k in DURATIONS}
    ens_pp: dict[int, dict[str, np.ndarray | None]] = {k: {c: None for c in CATEGORIES} for k in DURATIONS}
    used_models: list[str] = []
    lat_coord = lon_coord = None

    for model in MODELS:
        tw_files = s16.find_tw_vpd_files(scen, model)
        pr_root = _pr_root(s16, scen, model)
        if not tw_files or pr_root is None:
            print(f"  [skip] {model}: tw={len(tw_files)} pr_root={pr_root}")
            continue

        print(f"\n  --- {model} ---")
        print(f"    TW/VPD files: {len(tw_files)}")
        print(f"    pr root: {pr_root}")

        pr_raw, pr_ds = _open_pr(s16, pr_root)
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

            if lat_coord is None:
                lat_coord = pb[DURATIONS[0]][CATEGORIES[0]]["lat"]
                lon_coord = pb[DURATIONS[0]][CATEGORIES[0]]["lon"]

            for k in DURATIONS:
                for cat in CATEGORIES:
                    b = pb[k][cat].compute().values.astype("float32")
                    p = pp[k][cat].compute().values.astype("float32")
                    if ens_pb[k][cat] is None:
                        ens_pb[k][cat] = b
                        ens_pp[k][cat] = p
                    else:
                        ens_pb[k][cat] = ens_pb[k][cat] + b
                        ens_pp[k][cat] = ens_pp[k][cat] + p
            used_models.append(model)
            print(f"    done: {model}")
        finally:
            ds.close()
            pr_ds.close()

    if not used_models:
        print(f"[ERROR] {scen}: no models computed")
        return None

    n_used = len(used_models)

    def stack_probs(ens: dict[int, dict[str, np.ndarray | None]]) -> np.ndarray:
        return np.stack(
            [
                np.stack([ens[k][c] / float(n_used) for c in CATEGORIES], axis=0)  # type: ignore[operator]
                for k in DURATIONS
            ],
            axis=0,
        ).astype("float32")

    data_b = stack_probs(ens_pb)
    data_p = stack_probs(ens_pp)
    data_d = data_p - data_b

    ds_out = xr.Dataset(
        {
            "P_baseline": (("duration", "category", "lat", "lon"), data_b),
            "P_preconditioned": (("duration", "category", "lat", "lon"), data_p),
            "P_diff": (("duration", "category", "lat", "lon"), data_d.astype("float32")),
        },
        coords={
            "duration": np.array(DURATIONS, dtype=np.int32),
            "category": CATEGORIES,
            "lat": lat_coord,
            "lon": lon_coord,
        },
    )
    ds_out.attrs.update(
        scenario=scen,
        window_days=WINDOW_DAYS,
        precondition_kind=PRE_KIND,
        precondition_L=PRE_L,
        wet_day_mm=WET_DAY_MM,
        accum_mm=ACCUM_MM,
        models=",".join(used_models),
        n_models=n_used,
        tw_quantile=TW_Q,
        vpd_quantile=VPD_Q,
        vpd_dry_quantile=DRY_Q,
        merge_note="compute_post_hh_ssp_ensemble.py (cache-only)",
    )

    print(f"\n  Write cache ({n_used} models): {cache_fp.name}")
    s16.write_cache(ds_out, cache_fp)
    ds_out.close()
    return cache_fp


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache-only Fig.4 post-HH SSP ensemble")
    parser.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s16 = _load_s16()

    print("Fig.4 post-HH SSP ensemble (cache-only)")
    print(f"  scenarios: {args.scenarios}")
    print(f"  models target: {MODELS}")
    print(f"  cache dir: {CACHE_DIR}")

    results: list[Path] = []
    for scen in args.scenarios:
        fp = compute_scenario(s16, scen, force=args.force)
        if fp is not None:
            results.append(fp)

    print(f"\nDone. Wrote/skipped {len(results)} scenario cache(s).")
    for fp in results:
        print(f"  {fp}")


if __name__ == "__main__":
    main()
