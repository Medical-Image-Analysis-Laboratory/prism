# Analysis — from predictions to the tables of the paper

`synthgen/inference.py` (via `scripts/evalall.sh`) writes one folder of
predictions per model and test set. These scripts turn them into the numbers
reported in the paper.

## Files

| file | role |
|---|---|
| `build_metrics.py` | per-scan metrics for the three simulated splits → `results/{slice,motion,volume}_metrics.csv` |
| `make_table.py` | Table 2 (simulated data) → `results/comparison_table.tex` |
| `make_table_chuv.py` | Table 4 (real data, slice metrics only) → `results/chuv_table.tex` |
| `stat_tests.py` | paired Wilcoxon tests, Holm-corrected → `results/stat_tests.csv` |

The two committed `.tex` files are the tables of the paper. The per-scan CSVs are
regenerated locally: they carry the subject identifiers of the source datasets,
so they are not redistributed (see `results/.gitignore`).

## Configuration

Where the predictions live:

```bash
export PRISM_SIM_TEST_DATA=/data/simulated_test   # <root>/{Good,Med,Bad}/derivatives/predictions/...
export PRISM_PREDICTIONS=/data/predictions        # <root>/real/<model>/<timestamp>/...
```

`build_metrics.py` also takes `--root`, `--pred-subdir` and `--val-dir` if a run
was written somewhere else. The four models are identified by their training
`task_name`:

| id | paper |
|---|---|
| `1svort_4cgit` | SVoRT |
| `1svort_sqm` | SVoRT (+Q) |
| `1svort_reg` | SVoRT (+R) |
| `1svort_sqm_reg3` | **PRISM** |

## Run

```bash
python build_metrics.py                              # all three metric families
python build_metrics.py --only slice motion          # fast (seconds)
python build_metrics.py --only volume --workers 10   # ~1900 rigid registrations, cached
python make_table.py && python make_table_chuv.py && python stat_tests.py
```

Per-`(dataset, model)` caches under `results/cache/` make the volume step
resumable; `--force` ignores them.

## What each family measures

| family | source | metric | direction |
|---|---|---|---|
| **Motion** | ground-truth `*_motion.npz` (dataset root) vs `*_motion-pred.npz` | translation error (mm), rotation geodesic error (deg), anchor-point RMSE | lower is better |
| **Slice** | each run's `metrics.csv` | `slice_MSE/PSNR/SSIM/NCC/NCC_brain`, per scan averaged over its slices | higher is better (MSE lower) |
| **Volume** | predicted `*_type-SRRest_T2w.nii.gz` vs the reference volume | `vol_MSE/PSNR/SSIM/NCC` after rigid registration | higher is better (MSE lower) |

**Motion.** Predicted motion carries the scanner's real stack id and the centred
slice index and keeps every slice, so ground truth and prediction share a
per-slice key and are joined *directly* on `(stack_id, slice_in_scan)` — no
Hungarian assignment, no index-shift search. Near-empty slices are excluded via
the per-scan `slice_metrics.csv`. (A "<50 % matched" warning means the
predictions were produced by an older inference run.)

**Volume.** The reconstruction lives in SVoRT's canonical frame, so each pair is
rigidly registered (ANTs, Mattes MI, centre-of-mass init) before being scored
inside the reference brain mask. A least-squares scale+offset intensity match
makes MSE/PSNR/SSIM comparable across models; `vol_NCC` is scale-invariant, and
`vol_NCC_pre` is the pre-registration value.
