#!/usr/bin/env python3
"""
Cache-only: 3-model post-HH precondition ensemble by future epoch (NF / MC / EC).

Matches Fig. 5 epoch windows:
  NF: 2021–2040, MC: 2041–2060, EC: 2081–2100

Statistical protocol (aligned with compute_post_hh_ssp_ensemble.py):
  - Grid-cell thresholds: TW Q99, VPD Q95 (HH), VPD Q95 dry tail — from the full
    aligned SSP daily series (2015–2100), same as the existing full-period Fig. 4 caches.
  - Dry-spell precondition: no wet day >= 1 mm in prior 14 days (rolling on full series).
  - Triggers: HH run-end days with duration >= 3 d whose calendar year lies in the epoch.
  - Successor: any hit of category in the next 7 days (may extend slightly past epoch end).
  - Ensemble: unweighted mean of model fields (n=3).

Writes one NetCDF per SSP to cache_postHH_precondition_FINAL/ with an ``epoch`` dimension.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_DIR = Path("/intelnvme04/haider/project/compound_humid_heat")
SCRIPTS = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "cache_postHH_precondition_FINAL"
CHECKPOINT_DIR = CACHE_DIR / "_fig4_epoch_checkpoints"
LOG_DIR = PROJECT_DIR / "Manuscript final plots" / "logs"
RAW = Path("/intelnvme04/haider/raw_data")

MODELS = ["EC_Earth3", "MPI-ESM1-2-HR", "HadGEM3-GC31-LL"]
DEFAULT_SCENARIOS = ["ssp126", "ssp245", "ssp585"]

# Match plot_fig5_population_exposure.EPOCHS
EPOCHS: list[tuple[str, int, int]] = [
    ("NF", 2021, 2040),
    ("MC", 2041, 2060),
    ("EC", 2081, 2100),
]

WINDOW_DAYS = 7
DURATIONS = [3]
CATEGORIES = ["compound_again", "hot_only", "dry_only", "non_extreme"]
PRE_KIND = "dry_spell"
PRE_L = 14
WET_DAY_MM = 1.0
ACCUM_MM = 5.0
TW_Q, VPD_Q, DRY_Q = 0.99, 0.95, 0.95

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


def _slug(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)


def epoch_cache_path(scen: str) -> Path:
    epoch_tag = "epochs" + "-".join(ep[0] for ep in EPOCHS)
    dur_tag = "dur" + "-".join(map(str, DURATIONS))
    cat_tag = "cat" + "-".join(map(_slug, CATEGORIES))
    mod_tag = "mods" + "-".join(map(_slug, MODELS))
    pre_tag = f"pre{_slug(PRE_KIND)}_L{PRE_L}_wet{WET_DAY_MM:g}_acc{ACCUM_MM:g}"
    tw_tag = f"twQ{TW_Q:g}"
    vpd_tag = f"vpdQ{VPD_Q:g}"
    dry_tag = f"dryQ{DRY_Q:g}"
    name = (
        f"postHH_precond_{scen}_win{WINDOW_DAYS}_{epoch_tag}_{dur_tag}_{cat_tag}_"
        f"{mod_tag}_{pre_tag}_{tw_tag}_{vpd_tag}_{dry_tag}.nc"
    )
    return CACHE_DIR / _slug(name)


def prepare_model_epoch_masks(
    s16,
    tw: xr.DataArray,
    vpd: xr.DataArray,
    *,
    tw_thr: xr.DataArray,
    vpd_thr: xr.DataArray,
    vpd_dry_thr: xr.DataArray,
) -> tuple[dict[int, xr.DataArray], dict[str, xr.DataArray]]:
    """HH-end masks and successor windows — computed once per model, reused across epochs."""
    HH = ((tw > tw_thr) & (vpd > vpd_thr)).astype("int8")
    HH_end = s16.hh_run_end_mask(HH).astype(bool)

    hot = tw > tw_thr
    dry = vpd > vpd_dry_thr
    cat_maps: dict[str, xr.DataArray] = {
        "compound_again": hot & dry,
        "hot_only": hot & (~dry),
        "dry_only": (~hot) & dry,
    }
    cat_maps["non_extreme"] = ~(cat_maps["compound_again"] | cat_maps["hot_only"] | cat_maps["dry_only"])

    cat_any: dict[str, xr.DataArray] = {}
    for cat in CATEGORIES:
        cat_any[cat] = s16.successor_any_in_window(cat_maps[cat], window_days=WINDOW_DAYS)

    hh_end_k: dict[int, xr.DataArray] = {}
    for k in DURATIONS:
        hh_end_k[k] = s16.hh_run_end_duration_ge(HH, HH_end, k)
    return hh_end_k, cat_any


def compute_baseline_and_preconditioned_probs_for_epoch(
    pre: xr.DataArray,
    *,
    hh_end_k: dict[int, xr.DataArray],
    cat_any: dict[str, xr.DataArray],
    year_lo: int,
    year_hi: int,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    """Conditional probabilities counting only HH-end triggers in [year_lo, year_hi].

    Masks (HH-end, successor windows, dry-spell pre) are built on the full 2015–2100
    series; only the time reduction is restricted to calendar years in the epoch.
    """
    years = pre.time.dt.year
    trigger_epoch = (years >= year_lo) & (years <= year_hi)
    pre_ep = pre.sel(time=trigger_epoch)

    base: dict[int, dict[str, np.ndarray]] = {}
    pre_d: dict[int, dict[str, np.ndarray]] = {}

    for k in DURATIONS:
        countable = (hh_end_k[k] & trigger_epoch).sel(time=trigger_epoch)
        cat_ep = {cat: cat_any[cat].sel(time=trigger_epoch) for cat in CATEGORIES}

        denom_b = countable.sum(dim="time")
        denom_b = denom_b.where(denom_b > 0)
        denom_p = (countable & pre_ep).sum(dim="time")
        denom_p = denom_p.where(denom_p > 0)

        base[k] = {}
        pre_d[k] = {}
        for cat in CATEGORIES:
            pb = (countable & cat_ep[cat]).sum(dim="time") / denom_b
            pp = (countable & pre_ep & cat_ep[cat]).sum(dim="time") / denom_p
            base[k][cat] = pb.astype("float32").where(np.isfinite(pb)).compute().values
            pre_d[k][cat] = pp.astype("float32").where(np.isfinite(pp)).compute().values

    return base, pre_d


def _checkpoint_path(scen: str, model: str) -> Path:
    return CHECKPOINT_DIR / f"{scen}_{_slug(model)}.npz"


def _save_model_checkpoint(
    path: Path,
    *,
    scen: str,
    model: str,
    pb_by_epoch: dict[str, dict[int, dict[str, np.ndarray]]],
    pp_by_epoch: dict[str, dict[int, dict[str, np.ndarray]]],
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch_labels = [e[0] for e in EPOCHS]
    pb_stack = np.stack(
        [
            np.stack(
                [np.stack([pb_by_epoch[ep][k][c] for c in CATEGORIES], axis=0) for k in DURATIONS],
                axis=0,
            )
            for ep in epoch_labels
        ],
        axis=0,
    ).astype("float32")
    pp_stack = np.stack(
        [
            np.stack(
                [np.stack([pp_by_epoch[ep][k][c] for c in CATEGORIES], axis=0) for k in DURATIONS],
                axis=0,
            )
            for ep in epoch_labels
        ],
        axis=0,
    ).astype("float32")
    np.savez_compressed(
        path,
        pb=pb_stack,
        pp=pp_stack,
        lat=lat.values.astype("float32"),
        lon=lon.values.astype("float32"),
        scenario=scen,
        model=model,
        epoch_labels=np.array(epoch_labels),
    )


def _load_model_checkpoint(path: Path) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], dict[str, dict[int, dict[str, np.ndarray]]], np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    epoch_labels = [str(x) for x in z["epoch_labels"]]
    pb_stack = z["pb"]
    pp_stack = z["pp"]
    pb_by_epoch: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    pp_by_epoch: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for ei, ep in enumerate(epoch_labels):
        pb_by_epoch[ep] = {k: {} for k in DURATIONS}
        pp_by_epoch[ep] = {k: {} for k in DURATIONS}
        for ki, k in enumerate(DURATIONS):
            for ci, cat in enumerate(CATEGORIES):
                pb_by_epoch[ep][k][cat] = pb_stack[ei, ki, ci]
                pp_by_epoch[ep][k][cat] = pp_stack[ei, ki, ci]
    return pb_by_epoch, pp_by_epoch, z["lat"], z["lon"]


def _disk_status(path: Path) -> str:
    usage = shutil.disk_usage(path)
    gb_free = usage.free / 1e9
    pct = 100.0 * usage.used / usage.total
    return f"{gb_free:.0f} GB free ({pct:.0f}% used on {path})"


def validate_cache(path: Path, *, scen: str) -> None:
    """Lightweight QC after write."""
    with xr.open_dataset(path) as ds:
        assert int(ds.attrs.get("n_models", 0)) >= 3, "expected 3-model ensemble"
        assert list(ds.epoch.values) == [e[0] for e in EPOCHS]
        pb = ds["P_baseline"].sel(duration=3, category="dry_only")
        pp = ds["P_preconditioned"].sel(duration=3, category="dry_only")
        pd = ds["P_diff"].sel(duration=3, category="dry_only")
        max_abs_diff = float(np.nanmax(np.abs((pp - pb - pd).values)))
        assert max_abs_diff < 1e-5, f"P_diff mismatch max={max_abs_diff}"
        if float(pb.max()) > 1.0 + 1e-4 or float(pp.max()) > 1.0 + 1e-4:
            raise AssertionError("probabilities exceed 1")
        if float(pb.min()) < -1e-4 or float(pp.min()) < -1e-4:
            raise AssertionError("negative probabilities")
        print(f"  [QC OK] {path.name}")
        for ep in ds.epoch.values:
            d = float(pd.sel(epoch=ep).mean())
            print(f"    {scen} {ep} global mean ΔP(dry_only) = {d:+.4f}")


def _accumulate_model_into_ensemble(
    ens_pb: dict,
    ens_pp: dict,
    pb_by_epoch: dict[str, dict[int, dict[str, np.ndarray]]],
    pp_by_epoch: dict[str, dict[int, dict[str, np.ndarray]]],
) -> None:
    for ep_label, _, _ in EPOCHS:
        for k in DURATIONS:
            for cat in CATEGORIES:
                b = pb_by_epoch[ep_label][k][cat]
                p = pp_by_epoch[ep_label][k][cat]
                if ens_pb[ep_label][k][cat] is None:
                    ens_pb[ep_label][k][cat] = b.astype("float32")
                    ens_pp[ep_label][k][cat] = p.astype("float32")
                else:
                    ens_pb[ep_label][k][cat] = ens_pb[ep_label][k][cat] + b
                    ens_pp[ep_label][k][cat] = ens_pp[ep_label][k][cat] + p


def compute_scenario(s16, scen: str, *, force: bool, resume: bool) -> Path | None:
    cache_fp = epoch_cache_path(scen)
    if cache_fp.exists() and not force:
        with xr.open_dataset(cache_fp) as ds:
            n = int(ds.attrs.get("n_models", 0))
        print(f"[skip] {scen}: epoch cache exists (n_models={n}): {cache_fp.name}")
        validate_cache(cache_fp, scen=scen)
        return cache_fp

    print(f"\n{'=' * 60}\nScenario: {scen} (epochs NF/MC/EC)\n{'=' * 60}")

    n_ep = len(EPOCHS)
    ens_pb = {
        ep: {k: {c: None for c in CATEGORIES} for k in DURATIONS} for ep, _, _ in EPOCHS
    }
    ens_pp = {
        ep: {k: {c: None for c in CATEGORIES} for k in DURATIONS} for ep, _, _ in EPOCHS
    }
    used_models: list[str] = []
    lat_coord = lon_coord = None

    for model in MODELS:
        ckpt = _checkpoint_path(scen, model)
        if resume and ckpt.exists() and not force:
            print(f"\n  --- {model} --- [checkpoint resume]")
            pb_by_epoch, pp_by_epoch, lat_np, lon_np = _load_model_checkpoint(ckpt)
            if lat_coord is None:
                lat_coord = xr.DataArray(lat_np, dims=("lat",))
                lon_coord = xr.DataArray(lon_np, dims=("lon",))
            _accumulate_model_into_ensemble(ens_pb, ens_pp, pb_by_epoch, pp_by_epoch)
            used_models.append(model)
            print(f"    loaded checkpoint: {ckpt.name}")
            continue

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
            hh_end_k, cat_any = prepare_model_epoch_masks(
                s16,
                tw_a,
                vpd_a,
                tw_thr=tw_thr,
                vpd_thr=vpd_thr,
                vpd_dry_thr=vpd_dry_thr,
            )

            pb_by_epoch: dict[str, dict[int, dict[str, np.ndarray]]] = {
                ep: {k: {} for k in DURATIONS} for ep, _, _ in EPOCHS
            }
            pp_by_epoch: dict[str, dict[int, dict[str, np.ndarray]]] = {
                ep: {k: {} for k in DURATIONS} for ep, _, _ in EPOCHS
            }

            for ep_label, y0, y1 in EPOCHS:
                pb, pp = compute_baseline_and_preconditioned_probs_for_epoch(
                    pre,
                    hh_end_k=hh_end_k,
                    cat_any=cat_any,
                    year_lo=y0,
                    year_hi=y1,
                )
                if lat_coord is None:
                    lat_coord = tw_a["lat"]
                    lon_coord = tw_a["lon"]

                for k in DURATIONS:
                    for cat in CATEGORIES:
                        pb_by_epoch[ep_label][k][cat] = pb[k][cat]
                        pp_by_epoch[ep_label][k][cat] = pp[k][cat]
                print(f"    epoch {ep_label} ({y0}-{y1}) done")

            _save_model_checkpoint(
                ckpt,
                scen=scen,
                model=model,
                pb_by_epoch=pb_by_epoch,
                pp_by_epoch=pp_by_epoch,
                lat=lat_coord,
                lon=lon_coord,
            )
            print(f"    checkpoint saved: {ckpt.name}")
            _accumulate_model_into_ensemble(ens_pb, ens_pp, pb_by_epoch, pp_by_epoch)

            used_models.append(model)
            print(f"    done: {model}")
        finally:
            ds.close()
            pr_ds.close()

    if not used_models:
        print(f"[ERROR] {scen}: no models computed")
        return None

    n_used = len(used_models)
    epoch_labels = [e[0] for e in EPOCHS]

    def stack_probs(ens: dict) -> np.ndarray:
        # (epoch, duration, category, lat, lon)
        return np.stack(
            [
                np.stack(
                    [
                        np.stack([ens[ep][k][c] / float(n_used) for c in CATEGORIES], axis=0)
                        for k in DURATIONS
                    ],
                    axis=0,
                )
                for ep in epoch_labels
            ],
            axis=0,
        ).astype("float32")

    data_b = stack_probs(ens_pb)
    data_p = stack_probs(ens_pp)
    data_d = data_p - data_b

    ds_out = xr.Dataset(
        {
            "P_baseline": (("epoch", "duration", "category", "lat", "lon"), data_b),
            "P_preconditioned": (("epoch", "duration", "category", "lat", "lon"), data_p),
            "P_diff": (("epoch", "duration", "category", "lat", "lon"), data_d.astype("float32")),
        },
        coords={
            "epoch": epoch_labels,
            "duration": np.array(DURATIONS, dtype=np.int32),
            "category": CATEGORIES,
            "lat": lat_coord,
            "lon": lon_coord,
        },
    )
    for ep_label, y0, y1 in EPOCHS:
        ds_out["epoch"].attrs[f"{ep_label}_years"] = f"{y0}-{y1}"

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
        threshold_source="full_scenario_daily_series",
        trigger_definition="HH_end_duration_ge_3_calendar_year_in_epoch",
        successor_window_days=WINDOW_DAYS,
        merge_note="compute_post_hh_epoch_ensemble.py",
    )

    print(f"\n  Write epoch cache ({n_used} models): {cache_fp.name}")
    s16.write_cache(ds_out, cache_fp)
    ds_out.close()
    validate_cache(cache_fp, scen=scen)
    for model in used_models:
        ckpt = _checkpoint_path(scen, model)
        if ckpt.exists():
            ckpt.unlink()
            print(f"  removed checkpoint: {ckpt.name}")
    return cache_fp


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig.4 post-HH epoch ensemble (NF/MC/EC)")
    parser.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Load per-model checkpoints when present (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore checkpoints and recompute all models",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    s16 = _load_s16()

    t0 = datetime.now(timezone.utc).astimezone()
    print("=== Fig.4 post-HH epoch ensemble (cache-only) ===")
    print(f"START: {t0.isoformat(timespec='seconds')}")
    print(f"DISK: {_disk_status(CACHE_DIR)}")
    print(f"  scenarios: {args.scenarios}")
    print(f"  epochs: {EPOCHS}")
    print(f"  models target: {MODELS}")
    print(f"  cache dir: {CACHE_DIR}")
    print(f"  resume checkpoints: {args.resume} ({CHECKPOINT_DIR})")

    results: list[Path] = []
    for scen in args.scenarios:
        fp = compute_scenario(s16, scen, force=args.force, resume=args.resume)
        if fp is not None:
            results.append(fp)

    t1 = datetime.now(timezone.utc).astimezone()
    print(f"\nFINISHED: {t1.isoformat(timespec='seconds')}")
    print(f"Done. Wrote/skipped {len(results)} epoch cache(s).")
    for fp in results:
        print(f"  {fp}")


if __name__ == "__main__":
    main()
