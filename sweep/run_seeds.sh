#!/usr/bin/env bash
# Runs the adaptive ViT triggers over several seeds. Each run also contains the
# 1xs baseline, so every seed gives its own baseline and ViT numbers together.
# One tag per seed; a run resumes if interrupted, so it can simply be re-run.
#
# Usage (from the repo root):
#   bash sweep/run_seeds.sh                       # seeds 0 1 2
#   SEEDS="0 1 2 3 4" bash sweep/run_seeds.sh
set -euo pipefail

SEEDS=${SEEDS:-"0 1 2"}
PYTHON=${PYTHON:-python}
# New prefix so a rerun does not resume (skip) the outputs of an earlier one.
PREFIX=${PREFIX:-adaptfix}

for s in $SEEDS; do
  echo "=== seed $s ==="
  $PYTHON sweep/run_trigger_sweep.py --tag ${PREFIX}_s$s --seed $s \
      --epsilons 4 8 --budgets 1500 --experts 4 --expert-epochs 20 \
      --user-epochs 60 --adaptive-floor 0.25 --lambda-div 1.0 \
      --div-tau 0.3 --trigger-epochs 20
done

echo "All seeds finished. Results: out/sweep/${PREFIX}_s<seed>/results.csv"
