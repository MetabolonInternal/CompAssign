#!/usr/bin/env bash
set -euo pipefail

# Evaluate trained RT ridge coefficient-summary artifacts on realtest and regenerate plots.
#
# This script expects a run directory created by:
#   ./scripts/run_rt_prod.sh
#
# Usage:
#   ./scripts/run_rt_prod_eval.sh
#   ./scripts/run_rt_prod_eval.sh --cap 100 --libs 208,209
#   ./scripts/run_rt_prod_eval.sh --run-dir output/rt_prod_YYYYMMDD_HHMMSS

usage() {
  cat <<'EOF'
Evaluate RT ridge models on realtest and generate plots.

Usage:
  ./scripts/run_rt_prod_eval.sh [options]

Options:
  --run-dir <path>       Run directory created by ./scripts/run_rt_prod.sh
                         (default: read from output/rt_prod_latest.txt)
  --cap <capN|N>         Cap label under run directory (default: cap100)
  --libs <ids>           Comma-separated lib ids (default: 208,209)
  --chunk-size <int>     Eval chunk size (default: 200000)
  --log-every-chunks <n> Eval progress frequency (default: 5)
  --skip-existing        Skip evals with existing JSON (default)
  --no-skip-existing     Re-run evals even if JSON exists
  --no-plots             Skip plot regeneration
  -h, --help             Show this help text
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR=""
CAP="cap100"
LIBS="208,209"
CHUNK_SIZE="200000"
LOG_EVERY_CHUNKS="5"
SKIP_EXISTING="1"
DO_PLOTS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="${2:-}"
      shift 2
      ;;
    --cap)
      CAP="${2:-}"
      shift 2
      ;;
    --libs)
      LIBS="${2:-}"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="${2:-}"
      shift 2
      ;;
    --log-every-chunks)
      LOG_EVERY_CHUNKS="${2:-}"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING="1"
      shift 1
      ;;
    --no-skip-existing)
      SKIP_EXISTING="0"
      shift 1
      ;;
    --no-plots)
      DO_PLOTS="0"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$CAP" =~ ^[0-9]+$ ]]; then
  CAP="cap${CAP}"
fi

if [[ -z "${RUN_DIR}" ]]; then
  LATEST_PATH="${REPO_ROOT}/output/rt_prod_latest.txt"
  if [[ ! -f "${LATEST_PATH}" ]]; then
    echo "ERROR: --run-dir not provided and missing pointer: ${LATEST_PATH}" >&2
    exit 2
  fi
  RUN_DIR="$(cat "${LATEST_PATH}" | tr -d '[:space:]')"
fi
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${REPO_ROOT}/${RUN_DIR}"
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: Run directory does not exist: ${RUN_DIR}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs"

run_cmd() {
  local log_path="$1"
  shift
  echo "" | tee -a "${log_path}" >/dev/null
  echo "[run] $*" | tee -a "${log_path}"
  # shellcheck disable=SC2068
  "$@" 2>&1 | tee -a "${log_path}"
}

eval_model() {
  local lib_id="$1"
  local coeff_npz="$2"
  local test_csv="$3"
  local log_path="$4"

  local out_json=""
  if [[ "${coeff_npz}" == */models/stage1_coeff_summaries_posterior.npz ]]; then
    out_json="$(dirname "$(dirname "${coeff_npz}")")/results/rt_eval_coeff_summaries_by_support_realtest.json"
  else
    out_json="$(dirname "${coeff_npz}")/results/rt_eval_coeff_summaries_by_support_realtest.json"
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_json}" ]]; then
    echo "[eval] Skip (exists): ${out_json}"
    return 0
  fi

  run_cmd "${log_path}" \
    poetry run python -u scripts/pipelines/eval_rt_coeff_summaries_by_support.py \
      --coeff-npz "${coeff_npz}" \
      --test-csv "${test_csv}" \
      --chunk-size "${CHUNK_SIZE}" \
      --log-every-chunks "${LOG_EVERY_CHUNKS}" \
      --label realtest
}

IFS=',' read -r -a LIB_ARR <<< "${LIBS}"
for lib in "${LIB_ARR[@]}"; do
  lib="$(echo "${lib}" | tr -d '[:space:]')"
  if [[ -z "${lib}" ]]; then
    continue
  fi
  if [[ ! "${lib}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid lib id: ${lib}" >&2
    exit 2
  fi

  TEST_CSV="repo_export/lib${lib}/realtest/merged_training_realtest_lib${lib}_chemclass_rt_prod.csv"
  if [[ ! -f "${TEST_CSV}" ]]; then
    echo "ERROR: Missing realtest CSV: ${TEST_CSV}" >&2
    exit 2
  fi

  echo "[eval] lib${lib} ${CAP}"

  PARTIAL_COEFF="${RUN_DIR}/lib${lib}/${CAP}/features_none/pymc_pooled_species_comp_hier_supercat_cluster_supercat/models/stage1_coeff_summaries_posterior.npz"
  if [[ -f "${PARTIAL_COEFF}" ]]; then
    eval_model "${lib}" "${PARTIAL_COEFF}" "${TEST_CSV}" \
      "${RUN_DIR}/logs/lib${lib}_${CAP}_none_pymc_pooled_species_comp_hier_supercat_cluster_supercat.eval.log"
  else
    echo "[eval] Skip missing partial coeff: ${PARTIAL_COEFF}"
  fi

  SUPERCAT_COEFF="${RUN_DIR}/lib${lib}/${CAP}/features_none/pymc_collapsed_group_species_cluster/models/stage1_coeff_summaries_posterior.npz"
  if [[ -f "${SUPERCAT_COEFF}" ]]; then
    eval_model "${lib}" "${SUPERCAT_COEFF}" "${TEST_CSV}" \
      "${RUN_DIR}/logs/lib${lib}_${CAP}_none_pymc_collapsed_group_species_cluster.eval.log"
  else
    echo "[eval] Skip missing supercat coeff: ${SUPERCAT_COEFF}"
  fi

  SK_COEFF="${RUN_DIR}/lib${lib}/${CAP}/sklearn_ridge_species_cluster/stage1_coeff_summaries.npz"
  if [[ -f "${SK_COEFF}" ]]; then
    eval_model "${lib}" "${SK_COEFF}" "${TEST_CSV}" \
      "${RUN_DIR}/logs/lib${lib}_${CAP}_sklearn_ridge_species_cluster.eval.log"
  fi
done

if [[ "${DO_PLOTS}" == "1" ]]; then
  echo "[eval] Plotting (global + by support + by species_cluster)"
  ./scripts/plot_rt_multilevel.sh --run-dir "${RUN_DIR}" --cap "${CAP}" --libs "${LIBS}"
fi

echo "[eval] Done."
