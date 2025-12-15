#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Ensure a deterministic default seed unless the user provided one
ARGS=("$@")
if [[ " ${ARGS[*]} " != *" --seed "* ]]; then
  ARGS+=("--seed" "42")
fi

python -m src.compassign.rt.model_comparison "${ARGS[@]}"
