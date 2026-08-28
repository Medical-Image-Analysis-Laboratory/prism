#!/usr/bin/env python3
"""Build per-scan comparison metrics for several SVoRT models across datasets.

Produces three tidy CSVs under ``results/`` that the companion notebook
(``ModelComparison.ipynb``) loads and plots:

1. ``slice_metrics.csv``  -- per-scan slice-reconstruction metrics, read directly
   from each run's ``metrics.csv`` (MSE/PSNR/SSIM/NCC/NCC_brain averaged over the
   slices of a scan by ``synthgen/inference.py``).

2. ``motion_metrics.csv`` -- per-scan motion-estimation error (GT vs predicted
   per-slice rigid transforms). GT motion lives at the dataset root
   (``<sub>/<ses>/anat/<name>_motion.npz``, written by generate_stack_database),
   predicted motion at ``<pred>/.../<name>_motion-pred.npz``. With the updated
   inference the predicted motion carries the scanner's real stack id and centered
   slice index (and keeps every slice), so GT and prediction share the same
   per-slice key and are joined DIRECTLY on ``(stack_id, slice_in_scan)`` --
   no stack/slice matching. Near-empty slices are excluded via the per-scan
   ``slice_metrics.csv``. Reported per scan: translation error (mm), rotation
   geodesic error (deg), anchor-point RMSE (the point_loss space), match counts.

3. ``volume_metrics.csv`` -- per-scan volume metrics between the predicted SRR
   volume (``<name>_type-SRRest_T2w.nii.gz``) and the GT volume
   (``derivatives/gt_volume/.../<name>_T2wbiascorrected.nii.gz``). The SRR is
   reconstructed in SVoRT's canonical frame, so the pair is first *rigidly
   registered* (ANTs, Mattes MI, CoM init). Metrics are computed inside the GT
   brain mask after a least-squares intensity match (scale+offset) of the warped
   SRR to the GT: MSE, PSNR, SSIM, NCC.

Run once (slow part is the ~1900 registrations, parallelised):

    python build_metrics.py               # all three, all models/datasets
    python build_metrics.py --only slice motion
    python build_metrics.py --only volume --workers 8

Per-(dataset, model) cache files under ``results/cache/`` make it resumable.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
# Root of the simulated test datasets: one BIDS folder per ill-posedness level,
# `<ROOT>/{Good,Med,Bad}`, as written by `synthgen.generate_stack_database`.
# Override with $PRISM_SIM_TEST_DATA or `--root`.
ROOT = Path(os.environ.get("PRISM_SIM_TEST_DATA", "data/simulated_test"))
DATASETS = ["Good", "Med", "Bad"]

# The four models of the ablation, by training `task_name` (see the README for
# how they map onto the names used in the paper).
MODELS = ["1svort_4cgit", "1svort_sqm", "1svort_reg", "1svort_sqm_reg3"]

# `<ROOT>/<dataset>/derivatives/<PRED_SUBDIR>/<model>/<timestamp>/<VAL>` is where
# `synthgen/inference.py` wrote each run (`save_path` / split / image suffix).
PRED_SUBDIR = "predictions"
VAL = "test_T2wbiascorrected"

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "results"
CACHEDIR = OUTDIR / "cache"


def pred_run_dir(dataset: str, model: str) -> Path:
    """Return ``.../<pred_subdir>/<model>/<timestamp>/<VAL>`` (single timestamp)."""
    base = ROOT / dataset / "derivatives" / PRED_SUBDIR / model
    ts = sorted(p.name for p in base.iterdir() if p.is_dir())
    if len(ts) != 1:
        raise RuntimeError(f"Expected exactly one timestamp under {base}, got {ts}")
    return base / ts[0] / VAL


def scan_name_from(path: Path) -> str:
    """`sub-XXX_ses-YYY` from a `..._motion-pred.npz` / `..._type-SRRest...` file."""
    n = path.name
    for suf in ("_motion-pred.npz", "_type-SRRest_T2w.nii.gz"):
        if n.endswith(suf):
            return n[: -len(suf)]
    raise ValueError(path)


def list_scans(dataset: str, model: str) -> list[str]:
    d = pred_run_dir(dataset, model)
    return sorted(
        scan_name_from(p) for p in d.glob("sub-*/ses-*/anat/*_motion-pred.npz")
    )


# --------------------------------------------------------------------------- #
#  1. Slice metrics (read straight from each run's metrics.csv)
# --------------------------------------------------------------------------- #
def build_slice_metrics() -> pd.DataFrame:
    frames = []
    for ds in DATASETS:
        for model in MODELS:
            csv = pred_run_dir(ds, model) / "metrics.csv"
            df = pd.read_csv(csv)
            df["dataset"] = ds
            df["model"] = model
            df["sub"] = df["name"].str.split("_").str[0]
            df["ses"] = df["name"].str.split("_").str[1]
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTDIR / "slice_metrics.csv", index=False)
    print(f"[slice ] wrote {len(out)} rows -> {OUTDIR/'slice_metrics.csv'}")
    return out


# --------------------------------------------------------------------------- #
#  2. Motion metrics (GT vs predicted per-slice rigid transforms)
#
#  The updated inference emits, per slice, the scanner's real stack id and
#  centered slice index (positions[:, 1], positions[:, 0]) and keeps every slice,
#  so a predicted slice and its GT slice share the SAME key. We therefore join GT
#  and prediction directly on (stack_id, slice_in_scan) -- no stack matching, no
#  index-shift search. Near-empty slices are excluded via the per-scan
#  slice_metrics.csv (the set of scored, non-empty slices).
# --------------------------------------------------------------------------- #
def _rot_geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between stacks of rotation matrices (..., 3, 3)."""
    R = Ra @ np.swapaxes(Rb, -1, -2)
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _load_motion(npz: Path) -> dict:
    d = np.load(npz)
    # per-slice key = (stack_id, slice_in_scan) = (positions[:,1], positions[:,0])
    keys = list(zip(
        d["stack_ids"].astype(int).tolist(),
        np.rint(d["positions"][:, 0]).astype(int).tolist(),
    ))
    return {
        "mat": d["matrix"],          # (N, 3, 4)
        "euler": d["euler"],         # (N, 6): tx,ty,tz,rx,ry,rz (deg)
        "points": d["points"],       # (N, 9)
        "keys": keys,                # list[(stack_id, slice_in_scan)]
    }


def _nonempty_keys(slice_csv: Path) -> set | None:
    """(stack_id, slice_in_scan) of the scored (non-empty) slices, or None.

    Uses the per-scan slice_metrics.csv written by eval/inference. Returns None
    when the file is absent or predates the ``slice_in_scan`` column, in which
    case no non-empty restriction is applied.
    """
    if not slice_csv.exists():
        return None
    df = pd.read_csv(slice_csv)
    if not {"stack_index", "slice_in_scan"}.issubset(df.columns):
        return None
    return set(zip(
        np.rint(df["stack_index"]).astype(int),
        np.rint(df["slice_in_scan"]).astype(int),
    ))


def _motion_metrics_direct(pred: dict, gt: dict, keep: set | None) -> dict:
    """Per-scan motion error by a deterministic per-stack ordered zip.

    Prediction and GT carry the same real ``stack_id`` and the same physical
    slices in the same acquisition order, so within each stack we pair them by
    rank (slices sorted by ``slice_in_scan``). This is robust to the scanner's
    ``subj_missing`` slice dropping, which leaves gaps in the GT ``slice_in_scan``
    that a plain key-merge would miss (the saved stack NIfTI only preserves order,
    not the original index). No fuzzy matching -- just group-by-stack + sort.

    ``n_pred`` counts the *scored* (non-empty) predicted slices, so
    ``n_matched / n_pred`` is the pairing success rate (~1.0). ``n_total_pred`` /
    ``n_gt`` are the full per-scan slice counts for reference.
    """
    gk = np.asarray(gt["keys"])        # (N, 2): stack_id, slice_in_scan
    pk = np.asarray(pred["keys"])
    trans_err, rot_err, point_se = [], [], []
    n_attempted = 0
    for s in np.unique(pk[:, 0]):
        gi = np.where(gk[:, 0] == s)[0]
        pi = np.where(pk[:, 0] == s)[0]
        if len(gi) == 0:
            continue
        gi = gi[np.argsort(gk[gi, 1], kind="stable")]  # ascending slice_in_scan
        pi = pi[np.argsort(pk[pi, 1], kind="stable")]
        for rank in range(min(len(gi), len(pi))):
            k, j = int(pi[rank]), int(gi[rank])
            if keep is not None and (int(pk[k, 0]), int(pk[k, 1])) not in keep:
                continue  # skip near-empty slices
            n_attempted += 1
            trans_err.append(
                float(np.linalg.norm(pred["euler"][k, :3] - gt["euler"][j, :3]))
            )
            rot_err.append(float(_rot_geodesic_deg(
                pred["mat"][k, :3, :3][None], gt["mat"][j, :3, :3][None])[0]))
            point_se.append(float(((pred["points"][k] - gt["points"][j]) ** 2).mean()))
    trans_err = np.asarray(trans_err)
    rot_err = np.asarray(rot_err)
    point_se = np.asarray(point_se)
    n = len(trans_err)
    return {
        "trans_err_mean": float(trans_err.mean()) if n else np.nan,
        "trans_err_median": float(np.median(trans_err)) if n else np.nan,
        "rot_err_mean": float(rot_err.mean()) if n else np.nan,
        "rot_err_median": float(np.median(rot_err)) if n else np.nan,
        "point_rmse": float(np.sqrt(point_se.mean())) if n else np.nan,
        "n_matched": int(n),
        "n_pred": int(n_attempted),
        "n_total_pred": int(len(pk)),
        "n_gt": int(len(gk)),
    }


def build_motion_metrics() -> pd.DataFrame:
    rows = []
    low_match = 0
    for ds in DATASETS:
        scans = list_scans(ds, MODELS[0])
        for name in scans:
            sub, ses = name.split("_")[0], name.split("_")[1]
            gt_npz = ROOT / ds / sub / ses / "anat" / f"{name}_motion.npz"
            if not gt_npz.exists():
                continue
            gt = _load_motion(gt_npz)
            for m in MODELS:
                anat = pred_run_dir(ds, m) / sub / ses / "anat"
                pred_npz = anat / f"{name}_motion-pred.npz"
                if not pred_npz.exists():
                    continue
                pred = _load_motion(pred_npz)
                keep = _nonempty_keys(anat / f"{sub}_slice_metrics.csv")
                r = _motion_metrics_direct(pred, gt, keep)
                if r["n_pred"] and r["n_matched"] / r["n_pred"] < 0.5:
                    low_match += 1
                r.update({"dataset": ds, "model": m, "name": name,
                          "sub": sub, "ses": ses})
                rows.append(r)
        print(f"[motion] {ds}: {len(scans)} scans processed")
    out = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTDIR / "motion_metrics.csv", index=False)
    print(f"[motion] wrote {len(out)} rows -> {OUTDIR/'motion_metrics.csv'}")
    if low_match:
        print(f"[motion] WARNING: {low_match} scans matched <50% of slices on the "
              "(stack_id, slice_in_scan) key. This usually means the predictions "
              "were NOT regenerated with the updated inference (their stack ids / "
              "slice indices don't line up with the GT).")
    return out


# --------------------------------------------------------------------------- #
#  3. Volume metrics (rigid registration + masked metrics)
# --------------------------------------------------------------------------- #
def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _volume_metrics_one(args: tuple) -> dict:
    ds, model, name, gt_path, pred_path = args
    import ants
    from skimage.metrics import structural_similarity as ssim

    out = {"dataset": ds, "model": model, "name": name,
           "sub": name.split("_")[0], "ses": name.split("_")[1]}
    try:
        fixed = ants.image_read(str(gt_path))
        moving = ants.image_read(str(pred_path))
        gt = fixed.numpy()
        mask = gt > 0
        pre = _ncc(moving.numpy(), gt, mask)
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
        warped = reg["warpedmovout"].numpy()
        # least-squares intensity match (scale+offset) of warped SRR to GT in mask
        x = warped[mask]
        y = gt[mask]
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        pm = a * warped + b
        diff = (pm - gt)[mask]
        mse = float((diff ** 2).mean())
        # SSIM inside the brain bounding box (both zeroed outside the mask)
        zz, yy, xx = np.where(mask)
        sl = (slice(zz.min(), zz.max() + 1),
              slice(yy.min(), yy.max() + 1),
              slice(xx.min(), xx.max() + 1))
        m3 = mask[sl]
        gt_c = np.where(m3, gt[sl], 0.0).astype(np.float64)
        pm_c = np.where(m3, np.clip(pm[sl], 0, None), 0.0).astype(np.float64)
        ssim_val = float(ssim(gt_c, pm_c, data_range=1.0))
        out.update({
            "vol_MSE": mse,
            "vol_PSNR": float(10.0 * np.log10(1.0 / (mse + 1e-12))),
            "vol_SSIM": ssim_val,
            "vol_NCC": _ncc(pm, gt, mask),
            "vol_NCC_pre": pre,
            "intensity_scale": float(a),
            "ok": True,
        })
    except Exception as e:  # keep going; flag the failure
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return out


def build_volume_metrics(workers: int, force: bool = False) -> pd.DataFrame:
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    # Limit ITK threads per worker to avoid oversubscription of the 16 cores.
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")
    all_frames = []
    for ds in DATASETS:
        for model in MODELS:
            cache = CACHEDIR / f"volume_{ds}_{model}.csv"
            if cache.exists() and not force:
                all_frames.append(pd.read_csv(cache))
                print(f"[volume] cached {ds}/{model}")
                continue
            d = pred_run_dir(ds, model)
            tasks = []
            for name in list_scans(ds, model):
                sub, ses = name.split("_")[0], name.split("_")[1]
                pred_path = d / sub / ses / "anat" / f"{name}_type-SRRest_T2w.nii.gz"
                gt_path = (ROOT / ds / "derivatives" / "gt_volume" / sub / ses /
                           "anat" / f"{name}_T2wbiascorrected.nii.gz")
                if pred_path.exists() and gt_path.exists():
                    tasks.append((ds, model, name, gt_path, pred_path))
            rows = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_volume_metrics_one, t) for t in tasks]
                for i, f in enumerate(as_completed(futs), 1):
                    rows.append(f.result())
                    if i % 25 == 0 or i == len(futs):
                        print(f"[volume] {ds}/{model}: {i}/{len(futs)}", flush=True)
            df = pd.DataFrame(rows)
            df.to_csv(cache, index=False)
            all_frames.append(df)
    out = pd.concat(all_frames, ignore_index=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTDIR / "volume_metrics.csv", index=False)
    nfail = int((~out["ok"]).sum()) if "ok" in out else 0
    print(f"[volume] wrote {len(out)} rows ({nfail} failed) "
          f"-> {OUTDIR/'volume_metrics.csv'}")
    return out


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def main():
    global ROOT, PRED_SUBDIR, VAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=["slice", "motion", "volume"],
                    default=["slice", "motion", "volume"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="ignore volume cache and recompute")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help=f"root of the simulated test datasets "
                         f"(default: {ROOT}, from $PRISM_SIM_TEST_DATA)")
    ap.add_argument("--pred-subdir", default=PRED_SUBDIR,
                    help=f"derivatives subfolder holding the predictions "
                         f"(default: {PRED_SUBDIR})")
    ap.add_argument("--val-dir", default=VAL,
                    help=f"per-run subfolder written by inference.py, "
                         f"`<split>_<img_suffix>` (default: {VAL})")
    args = ap.parse_args()
    ROOT, PRED_SUBDIR, VAL = args.root, args.pred_subdir, args.val_dir
    if "slice" in args.only:
        build_slice_metrics()
    if "motion" in args.only:
        build_motion_metrics()
    if "volume" in args.only:
        build_volume_metrics(workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
