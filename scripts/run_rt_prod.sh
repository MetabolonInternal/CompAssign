#!/usr/bin/env bash

set -euo pipefail

# Simple wrapper to run the production RT pipeline for one or more libraries.
# Usage:
#   ./scripts/run_rt_prod.sh [--quick] [--cap <capN|N>] [--lib <id[,id...]>] [--chains N] [--draws N] [--tune N] [--target-accept X] [--max-treedepth Y]
# Log all output for easier debugging and offline inspection.
LOG_DIR="output/rt_prod/logs"
mkdir -p "${LOG_DIR}"
LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_rt_prod_${LOG_TS}.log"
echo "Logging run_rt_prod.sh output to ${LOG_FILE}"
echo "Command: $0 $*" >> "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

QUICK_FLAG=""
SMOKE_FLAG=""
EXTRA_FLAGS=""
CAP="cap5"
TARGET_ACCEPT="0.9"
FORCE=0
LIB_IDS=()
shopt -s nullglob

# Prefer the dedicated compassign-rt Conda env on remote hosts if available.
PYTHON_BIN="python"
REMOTE_ENV_PYTHON="${HOME}/miniconda3/envs/compassign-rt/bin/python"
if [ -x "${REMOTE_ENV_PYTHON}" ]; then
  PYTHON_BIN="${REMOTE_ENV_PYTHON}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK_FLAG="--quick"
      shift
      ;;
    --superquick|--smoke)
      SMOKE_FLAG="--smoke"
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --cap)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --cap" >&2
        exit 1
      fi
      CAP="$2"
      shift 2
      ;;
    --lib)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --lib" >&2
        exit 1
      fi
      LIB_ARG="$2"
      IFS=',' read -r -a LIB_SPLIT <<< "${LIB_ARG}"
      for lib_val in "${LIB_SPLIT[@]}"; do
        if [[ -n "${lib_val}" ]]; then
          LIB_IDS+=("${lib_val}")
        fi
      done
      shift 2
      ;;
    --target-accept)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --target-accept" >&2
        exit 1
      fi
      TARGET_ACCEPT="$2"
      shift 2
      ;;
    --draws)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --draws" >&2
        exit 1
      fi
      EXTRA_FLAGS+=" --draws $2"
      shift 2
      ;;
    --tune)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --tune" >&2
        exit 1
      fi
      EXTRA_FLAGS+=" --tune $2"
      shift 2
      ;;
    --chains)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --chains" >&2
        exit 1
      fi
      EXTRA_FLAGS+=" --chains $2"
      shift 2
      ;;
    --max-treedepth)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --max-treedepth" >&2
        exit 1
      fi
      EXTRA_FLAGS+=" --max-treedepth $2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--quick] [--superquick] [--force] [--cap <capN|N>] [--lib <id[,id...]>] [--chains <val>] [--draws <val>] [--tune <val>] [--target-accept <val>] [--max-treedepth <val>]" >&2
      exit 1
      ;;
  esac
done

# Default to libs 208 and 209 if no explicit --lib provided.
if [[ ${#LIB_IDS[@]} -eq 0 ]]; then
  LIB_IDS=(208 209)
fi

# Normalize cap (allow numeric like 20 -> cap20)
if [[ "$CAP" =~ ^[0-9]+$ ]]; then
  CAP="cap${CAP}"
fi

find_data_csv() {
  local lib_id=$1
  local cap=$2
  local candidates=(
    "repo_export/lib${lib_id}/${cap}/merged_training_all_lib${lib_id}_${cap}_chemclass_rt_prod.csv"
    "repo_export/lib${lib_id}/${cap}/merged_training_*_lib${lib_id}_${cap}_chemclass_rt_prod.csv"
    "repo_export/merged_training_*_lib${lib_id}_${cap}_chemclass_rt_prod.csv"
  )
  for c in "${candidates[@]}"; do
    for f in $c; do
      echo "$f"
      return 0
    done
  done
  return 1
}

DATA208=$(find_data_csv 208 "$CAP") || { echo "Could not find data CSV for lib208 cap ${CAP}" >&2; exit 1; }
CORES=$(python -c 'import multiprocessing; print(multiprocessing.cpu_count())')

SUFFIX=""
if [[ -n "$QUICK_FLAG" ]]; then
  SUFFIX="_quick"
fi

for LIB_ID in "${LIB_IDS[@]}"; do
  DATA=$(find_data_csv "${LIB_ID}" "$CAP") || { echo "Could not find data CSV for lib${LIB_ID} cap ${CAP}" >&2; exit 1; }

  OUT_DIR="output/rt_prod/lib${LIB_ID}_${CAP}${SUFFIX}"

  if [[ -f "${OUT_DIR}/models/rt_trace.nc" && "${FORCE}" -ne 1 ]]; then
    echo "Skipping lib${LIB_ID} (${CAP}${SUFFIX}) – trace already exists at ${OUT_DIR}/models/rt_trace.nc"
  else
    "${PYTHON_BIN}" scripts/pipelines/train_rt_prod.py \
      --data-csv "$DATA" \
      --cores "$CORES" \
      --include-es-group1 \
      ${SMOKE_FLAG} \
      --target-accept "$TARGET_ACCEPT" \
      ${QUICK_FLAG} \
      ${EXTRA_FLAGS} \
      --output-dir "${OUT_DIR}"
  fi
done
