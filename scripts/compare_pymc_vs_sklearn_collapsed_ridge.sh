#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Compare PyMC collapsed ridge vs sklearn ridge baseline on cap100 -> realtest.

This script:
  1) Trains sklearn per-(species_cluster, comp_id) ridge summaries (Stage-1) for each lib.
  2) Evaluates both sklearn and PyMC collapsed ridge artifacts on realtest via the same evaluator.
  3) Writes a small summary CSV under <run-dir>/analysis/.

Usage:
  ./scripts/compare_pymc_vs_sklearn_collapsed_ridge.sh [options]

Options:
  --run-dir <path>   Run directory (default: output/rt_pymc_multilevel_cap100_latest.txt).
  --cap <capN|N>     Cap label (default: cap100).
  --libs <ids>       Comma-separated lib ids (default: infer from run-dir/lib*).
  --lambda <val>     Ridge penalty to match PyMC fixed lambda (default: 0.0003).
  --dry-run          Print commands and exit.
  -h, --help         Show this help text.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_DIR=""
CAP="cap100"
LIBS=""
LAMBDA="0.0003"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --cap)
      CAP="$2"
      shift 2
      ;;
    --libs)
      LIBS="$2"
      shift 2
      ;;
    --lambda)
      LAMBDA="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ -z "$RUN_DIR" ]]; then
  LATEST_PATH="${REPO_ROOT}/output/rt_pymc_multilevel_cap100_latest.txt"
  if [[ ! -f "${LATEST_PATH}" ]]; then
    echo "ERROR: Missing latest pointer: ${LATEST_PATH}" >&2
    echo "Pass --run-dir explicitly, or create the pointer file." >&2
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

infer_libs() {
  local run_dir="$1"
  local -a out=()
  while IFS= read -r p; do
    local name
    name="$(basename "$p")"
    name="${name#lib}"
    if [[ "$name" =~ ^[0-9]+$ ]]; then
      out+=("$name")
    fi
  done < <(find "${run_dir}" -maxdepth 1 -type d -name 'lib*' | sort)
  (IFS=,; echo "${out[*]}")
}

if [[ -z "${LIBS}" ]]; then
  LIBS="$(infer_libs "${RUN_DIR}")"
fi
if [[ -z "${LIBS}" ]]; then
  echo "ERROR: Could not infer libs under ${RUN_DIR}; pass --libs explicitly." >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/analysis"

echo "[compare] RUN_DIR=${RUN_DIR}"
echo "[compare] CAP=${CAP}"
echo "[compare] LIBS=${LIBS}"
echo "[compare] LAMBDA=${LAMBDA}"

run_cmd() {
  local log_path="$1"
  shift
  echo "" | tee -a "${log_path}" >/dev/null
  echo "[run] $*" | tee -a "${log_path}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  # shellcheck disable=SC2068
  "$@" 2>&1 | tee -a "${log_path}"
}

cd "${REPO_ROOT}"

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

  TRAIN_CSV="repo_export/lib${lib}/${CAP}/merged_training_all_lib${lib}_${CAP}_chemclass_rt_prod.csv"
  TEST_CSV="repo_export/lib${lib}/realtest/merged_training_realtest_lib${lib}_chemclass_rt_prod.csv"

  if [[ ! -f "${TRAIN_CSV}" ]]; then
    echo "ERROR: Missing train CSV: ${TRAIN_CSV}" >&2
    exit 2
  fi
  if [[ ! -f "${TEST_CSV}" ]]; then
    echo "ERROR: Missing test CSV: ${TEST_CSV}" >&2
    exit 2
  fi

  SK_OUT="${RUN_DIR}/lib${lib}/${CAP}/sklearn_ridge_species_cluster"
  mkdir -p "${SK_OUT}"

  PYMC_COEFF="${RUN_DIR}/lib${lib}/${CAP}/features_none/pymc_collapsed_group_species_cluster/models/stage1_coeff_summaries_posterior.npz"
  if [[ ! -f "${PYMC_COEFF}" ]]; then
    echo "ERROR: Missing PyMC coeff artifact: ${PYMC_COEFF}" >&2
    exit 2
  fi

  SK_COEFF="${SK_OUT}/stage1_coeff_summaries.npz"

  # 1) Train sklearn baseline (skip if already exists).
  if [[ ! -f "${SK_COEFF}" ]]; then
    run_cmd "${RUN_DIR}/logs/lib${lib}_${CAP}_sklearn_ridge_species_cluster.train.log" \
      poetry run python -u scripts/pipelines/train_rt_stage1_coeff_summaries.py \
        --data-csv "${TRAIN_CSV}" \
        --output-dir "${SK_OUT}" \
        --seed 42 \
        --lambda-ridge "${LAMBDA}" \
        --include-es-all \
        --feature-center global \
        --feature-rotation none \
        --anchor-expansion none
  else
    echo "[compare] Skip sklearn training (exists): ${SK_COEFF}"
  fi

  # 2) Evaluate sklearn baseline.
  run_cmd "${RUN_DIR}/logs/lib${lib}_${CAP}_sklearn_ridge_species_cluster.eval.log" \
    poetry run python -u scripts/pipelines/eval_rt_coeff_summaries_by_support.py \
      --coeff-npz "${SK_COEFF}" \
      --test-csv "${TEST_CSV}" \
      --chunk-size 200000 \
      --log-every-chunks 5 \
      --label realtest

  # 3) Re-evaluate PyMC collapsed ridge (ensures comparable evaluator/version).
  run_cmd "${RUN_DIR}/logs/lib${lib}_${CAP}_pymc_collapsed_group_species_cluster.reval.log" \
    poetry run python -u scripts/pipelines/eval_rt_coeff_summaries_by_support.py \
      --coeff-npz "${PYMC_COEFF}" \
      --test-csv "${TEST_CSV}" \
      --chunk-size 200000 \
      --log-every-chunks 5 \
      --label realtest
done

# 4) Summarize.
run_cmd "${RUN_DIR}/logs/${CAP}_pymc_vs_sklearn_collapsed_ridge.summary.log" \
  poetry run python -u scripts/pipelines/summarize_pymc_vs_sklearn_collapsed_ridge.py \
    --run-dir "${RUN_DIR}" \
    --cap "${CAP}" \
    --libs "${LIBS}"

