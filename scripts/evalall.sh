#!/usr/bin/env bash
#
# Evaluate one checkpoint on every test set of the paper: the three simulated
# ill-posedness levels (Good / Med / Bad) and the real clinical cohort.
#
# Usage:
#   ./scripts/evalall.sh <checkpoint> [extra hydra overrides ...]
#
# Dataset locations come from the environment (see configs/paths/default.yaml):
#   PRISM_SIM_TEST_DATA   root of the simulated test sets, <root>/{Good,Med,Bad}
#   PRISM_REAL_DATA       BIDS root of the real, brain-masked stacks
#   PRISM_PREDICTIONS     where the real-data predictions are written
#
# Example:
#   export PRISM_SIM_TEST_DATA=/data/simulated_test
#   export PRISM_REAL_DATA=/data/chuv
#   ./scripts/evalall.sh logs/1svort_sqm_reg3/runs/2026-07-01_12-31-38/checkpoints/epoch_987_*.ckpt
#
# Then build the tables with analysis/build_metrics.py + analysis/make_table*.py.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_checkpoint> [hydra overrides ...]" >&2
    exit 1
fi

ckpt="$1"
shift

if [ ! -f "$ckpt" ]; then
    echo "Error: checkpoint not found: $ckpt" >&2
    exit 1
fi

echo "Evaluating $ckpt"

for cfg in inferencesimgood inferencesimmed inferencesimbad inferencechuv; do
    echo "--- $cfg ---"
    python ./synthgen/inference.py --config-name="$cfg" \
        "ckpt_paths=[$ckpt]" "$@"
done

echo "All four test sets done."
