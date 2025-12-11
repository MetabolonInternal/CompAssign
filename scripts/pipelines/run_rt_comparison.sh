#!/usr/bin/env bash
set -euo pipefail

# Easy wrapper to run the 4-way RT comparison with logging and plots.
# Usage:
#   --quick   small dataset, light sampler (sanity check)
# Profile is fixed to 'default' (modelling-focused, full coverage).
#
# Logs are written to: output/rt_comparison/rt_model_comparison.log
# Tail live:   tail -f output/rt_comparison/rt_model_comparison.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

QUICK=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Single profile only
OUT_DIR="$REPO_ROOT/output/rt_comparison/default"
  mkdir -p "$OUT_DIR"
  if [[ $QUICK -eq 1 ]]; then
    echo "[run_rt_comparison] Running (QUICK)..."
    set -x
    poetry run python "$REPO_ROOT/src/utils/rt_model_comparison.py" \
      --quick \
      --output-dir "$OUT_DIR" \
      ${EXTRA_ARGS[@]:-}
    set +x
  else
    echo "[run_rt_comparison] Running..."
    set -x
    poetry run python "$REPO_ROOT/src/utils/rt_model_comparison.py" \
      --output-dir "$OUT_DIR" \
      ${EXTRA_ARGS[@]:-}
    set +x
  fi
  echo
  echo "[run_rt_comparison] Finished. Log at: $OUT_DIR/rt_model_comparison.log"
  echo "Tail: tail -f $OUT_DIR/rt_model_comparison.log"


echo
echo "[run_rt_comparison] All done. See output/rt_comparison/default for results."
