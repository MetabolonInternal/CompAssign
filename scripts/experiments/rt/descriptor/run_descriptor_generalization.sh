#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for descriptor generalization.
# Aggregated in-run replication via --reps (default 5) using base SEED.
# Quick (500/500/4) or full (1000/1000/4) mode. The Python entry computes both
# cold_start and mix_shift replicates in a single run and emits a combined plot.
#
# Env overrides:
#   SEED=42    → base seed; replicates use SEED+i
#   REPS=5     → number of replicates

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PY_SCRIPT="$REPO_ROOT/scripts/experiments/rt/descriptor/assess_descriptor_generalization.py"
POETRY_RUN=(poetry run python "$PY_SCRIPT")

SEED=${SEED:-42}
REPS=${REPS:-5}
quick_mode=0
declare -a EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./run_descriptor_generalization.sh [--quick] [additional args]

Runs descriptor generalization with in-run replication.
Replicates are controlled by SEED and REPS env vars (default SEED=42, REPS=5).
Each run computes cold_start and mix_shift and writes a combined plot.
--quick switches to 500/500/4 sampling; default is 1000/1000/4.
--plot-only aggregates existing JSONs and generates plots only (no sampling).
Any extra arguments are forwarded to assess_descriptor_generalization.py.
EOF
}

plot_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      quick_mode=1
      shift
      ;;
    --plot-only)
      # Forward to Python, but we will run once (no seed loop)
      plot_only=1
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ $plot_only -eq 1 ]]; then
  echo ">>> Plot-only mode: aggregating existing results"
  "${POETRY_RUN[@]}" ${EXTRA_ARGS:+"${EXTRA_ARGS[@]}"}
  echo "Descriptor generalization plotting complete."
  exit 0
fi

if [[ $quick_mode -eq 1 ]]; then
  echo ">>> Quick run: seed=${SEED}, reps=${REPS} (500/500/4)"
  "${POETRY_RUN[@]}" \
    --quick \
    --seed "$SEED" \
    --reps "$REPS" \
    ${EXTRA_ARGS:+"${EXTRA_ARGS[@]}"}
else
  echo ">>> Full run: seed=${SEED}, reps=${REPS} (1000/1000/4)"
  "${POETRY_RUN[@]}" \
    --seed "$SEED" \
    --reps "$REPS" \
    --draws 1000 \
    --tune 1000 \
    --chains 4 \
    ${EXTRA_ARGS:+"${EXTRA_ARGS[@]}"}
fi

echo "Descriptor generalization experiment complete."
