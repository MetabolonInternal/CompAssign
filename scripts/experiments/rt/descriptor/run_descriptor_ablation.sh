#!/usr/bin/env bash
set -euo pipefail

# Descriptor ablation runner aligned with other RT runners.
# One-shot Python entry with in-Python replication and plot-only support.
#
# Env overrides:
#   SEED=42  → base seed
#   REPS=5   → number of replicates
#
# --quick switches to 500/500/4; default is 1000/1000/4.
# --plot-only aggregates existing JSONs and generates plots only.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PY_SCRIPT="$REPO_ROOT/scripts/experiments/rt/descriptor/assess_descriptor_ablation.py"
POETRY_RUN=(poetry run python "$PY_SCRIPT")

SEED=${SEED:-42}
REPS=${REPS:-5}
quick_mode=0
plot_only=0
declare -a EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./run_descriptor_ablation.sh [--quick|--plot-only] [additional args]

Runs descriptor ablation with in-run replication.
Replicates are controlled by SEED and REPS env vars (default SEED=42, REPS=5).
--quick switches to 500/500/4 sampling; default is 1000/1000/4.
--plot-only aggregates existing JSONs and generates plots only (no sampling).
Any extra arguments are forwarded to assess_descriptor_ablation.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      quick_mode=1
      shift
      ;;
    --plot-only)
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
  echo "Descriptor ablation plotting complete."
  exit 0
fi

if [[ $quick_mode -eq 1 ]]; then
  echo ">>> Quick ablation run: seed=${SEED}, reps=${REPS} (500/500/4)"
  "${POETRY_RUN[@]}" \
    --quick \
    --seed "$SEED" \
    --reps "$REPS" \
    ${EXTRA_ARGS:+"${EXTRA_ARGS[@]}"}
else
  echo ">>> Full ablation run: seed=${SEED}, reps=${REPS} (1000/1000/4)"
  "${POETRY_RUN[@]}" \
    --seed "$SEED" \
    --reps "$REPS" \
    --draws 1000 \
    --tune 1000 \
    --chains 4 \
    ${EXTRA_ARGS:+"${EXTRA_ARGS[@]}"}
fi

echo "Done. Check output under: $REPO_ROOT/output/rt_descriptor_ablation" >&2
