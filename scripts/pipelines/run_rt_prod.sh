#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run the production-style RT loader/trainer.
# Examples:
#   ./scripts/run_rt_prod.sh --example --no-fit    # validate loading only
#   ./scripts/run_rt_prod.sh --example --smoke     # tiny sampler e2e
#   ./scripts/run_rt_prod.sh --data-csv <rt.csv> [--quick]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUT_DIR="${REPO_ROOT}/output/rt_prod"

# Ensure default output dir if not provided
ARGS=("$@")
if [[ " ${ARGS[*]} " != *" --output-dir "* ]]; then
  ARGS+=("--output-dir" "${OUT_DIR}")
fi

poetry run python "${REPO_ROOT}/scripts/pipelines/train_rt_prod.py" "${ARGS[@]}"
