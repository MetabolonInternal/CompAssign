#!/usr/bin/env bash
set -euo pipefail

# Fixed-train, binned covariate-shift robustness experiment.
# Produces mean±std MAE vs distance bins and ΔMAE plots under output/rt_covshift_holdout/.
# Quick mode uses 500/500/4; full defaults to 1000/1000/4.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Env overrides:
#   SEED=42         → base seed (reps use SEED, SEED+1, ...)
#   REPS=5          → number of replicates
#   RUNS_PER_SPECIES=8 → synthetic runs per species

SEED=${SEED:-42}
REPS=${REPS:-5}
RUNS_PER_SPECIES=${RUNS_PER_SPECIES:-8}

ARGS=(
  --seed "$SEED"
  --reps "$REPS"
  --runs-per-species "$RUNS_PER_SPECIES"
  "$@"
)

echo "Running fixed-train binned covariate-shift with args: ${ARGS[*]}" >&2
poetry run python "$REPO_ROOT/scripts/experiments/rt/covariate_shift/assess_covariate_shift_holdout.py" "${ARGS[@]}"

echo "Done. Check output under: $REPO_ROOT/output/rt_covshift_holdout" >&2
