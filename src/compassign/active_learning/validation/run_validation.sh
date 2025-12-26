#!/usr/bin/env bash
set -euo pipefail

# Active learning validation smoke check. Intended usage: run with no args.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -ne 0 ]]; then
  echo "ERROR: src/compassign/active_learning/validation/run_validation.sh takes no args." >&2
  exit 2
fi

if [[ -f "validation_results.json" ]]; then
  BACKUP_FILE="validation_results.json.backup.$(date +%Y%m%d_%H%M%S)"
  echo "[validation] Backing up existing results to ${BACKUP_FILE}"
  mv validation_results.json "${BACKUP_FILE}"
fi

echo "[validation] Running validation suite..."
poetry run python -m compassign.active_learning.validation.validate_active_learning_complete

echo "[validation] Generating plots..."
mkdir -p plots
poetry run python -m compassign.active_learning.validation.visualize_validation_results

echo "[validation] Done."
