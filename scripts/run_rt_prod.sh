#!/usr/bin/env bash
set -euo pipefail

# Train the RT ridge models used for peak assignment on production-style RT CSVs.
#
# This script trains:
#   1) PyMC ridge (partial pooling; recommended)
#   2) PyMC ridge (supercategory; fast fallback)
#   3) sklearn ridge (supercategory; baseline comparison)
#
# The output directory is a "run dir" compatible with:
#   ./scripts/plot_rt_multilevel.sh
#
# Usage:
#   ./scripts/run_rt_prod.sh
#   ./scripts/run_rt_prod.sh --libs 208 --cap 100
#   ./scripts/run_rt_prod.sh --run-dir output/rt_prod_YYYYMMDD_HHMMSS

usage() {
  cat <<'EOF'
Train RT ridge models (partial pooling + fallbacks) on production RT CSVs.

This writes a run directory with a consistent structure:
  <run-dir>/lib<lib>/<cap>/features_none/<pymc model>/models/stage1_coeff_summaries_posterior.npz
  <run-dir>/lib<lib>/<cap>/sklearn_ridge_species_cluster/stage1_coeff_summaries.npz

Defaults are chosen to match the current report (cap100 -> realtest).

Usage:
  ./scripts/run_rt_prod.sh [options]

Options:
  --run-dir <path>       Output run directory (default: output/rt_prod_<timestamp>)
  --cap <capN|N>         Training cap label (default: cap100)
  --libs <ids>           Comma-separated lib ids (default: 208,209)
  --seed <int>           Random seed (default: 42)
  --quick                Reduced ADVI steps for a smoke run
  --skip-existing        Skip models with existing artifacts (default)
  --no-skip-existing     Retrain even if artifacts exist
  --no-sklearn           Skip training sklearn baseline
  -h, --help             Show this help text
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR=""
CAP="cap100"
LIBS="208,209"
SEED="42"
QUICK="0"
SKIP_EXISTING="1"
TRAIN_SKLEARN="1"

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
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --quick)
      QUICK="1"
      shift 1
      ;;
    --skip-existing)
      SKIP_EXISTING="1"
      shift 1
      ;;
    --no-skip-existing)
      SKIP_EXISTING="0"
      shift 1
      ;;
    --no-sklearn)
      TRAIN_SKLEARN="0"
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
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="output/rt_prod_${TS}"
fi
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${REPO_ROOT}/${RUN_DIR}"
fi

mkdir -p "${RUN_DIR}/logs"
echo "${RUN_DIR}" > "${REPO_ROOT}/output/rt_prod_latest.txt"

ADVI_DRAWS="50"
ADVI_LOG_EVERY="1000"
ADVI_STEPS_SUPERCAT="5000"
ADVI_STEPS_PARTIAL="10000"
if [[ "${QUICK}" == "1" ]]; then
  ADVI_STEPS_SUPERCAT="2000"
  ADVI_STEPS_PARTIAL="3000"
fi

LAMBDA_SLOPES="3e-4"

run_cmd() {
  local log_path="$1"
  shift
  echo "" | tee -a "${log_path}" >/dev/null
  echo "[run] $*" | tee -a "${log_path}"
  # shellcheck disable=SC2068
  "$@" 2>&1 | tee -a "${log_path}"
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

  TRAIN_CSV="repo_export/lib${lib}/${CAP}/merged_training_all_lib${lib}_${CAP}_chemclass_rt_prod.csv"
  if [[ ! -f "${TRAIN_CSV}" ]]; then
    echo "ERROR: Missing train CSV: ${TRAIN_CSV}" >&2
    exit 2
  fi

  echo "[train] lib${lib} ${CAP}"

  # 1) Partial pooling (recommended).
  PARTIAL_OUT="${RUN_DIR}/lib${lib}/${CAP}/features_none/pymc_pooled_species_comp_hier_supercat_cluster_supercat"
  PARTIAL_COEFF="${PARTIAL_OUT}/models/stage1_coeff_summaries_posterior.npz"
  PARTIAL_LOG="${RUN_DIR}/logs/lib${lib}_${CAP}_none_pymc_pooled_species_comp_hier_supercat_cluster_supercat.train.log"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${PARTIAL_COEFF}" ]]; then
    echo "[train] Skip partial pooling (exists): ${PARTIAL_COEFF}"
  else
    run_cmd "${PARTIAL_LOG}" \
      poetry run python -u scripts/pipelines/train_rt_pymc_collapsed_ridge.py \
        --data-csv "${TRAIN_CSV}" \
        --output-dir "${PARTIAL_OUT}" \
        --model partial_pool \
        --seed "${SEED}" \
        --include-es-all \
        --lambda-slopes "${LAMBDA_SLOPES}" \
        --method advi \
        --advi-steps "${ADVI_STEPS_PARTIAL}" \
        --advi-log-every "${ADVI_LOG_EVERY}" \
        --advi-draws "${ADVI_DRAWS}"
  fi

  # 2) Supercategory collapsed ridge (fast fallback).
  SUPERCAT_OUT="${RUN_DIR}/lib${lib}/${CAP}/features_none/pymc_collapsed_group_species_cluster"
  SUPERCAT_COEFF="${SUPERCAT_OUT}/models/stage1_coeff_summaries_posterior.npz"
  SUPERCAT_LOG="${RUN_DIR}/logs/lib${lib}_${CAP}_none_pymc_collapsed_group_species_cluster.train.log"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${SUPERCAT_COEFF}" ]]; then
    echo "[train] Skip supercategory PyMC (exists): ${SUPERCAT_COEFF}"
  else
    run_cmd "${SUPERCAT_LOG}" \
      poetry run python -u scripts/pipelines/train_rt_pymc_collapsed_ridge.py \
        --data-csv "${TRAIN_CSV}" \
        --output-dir "${SUPERCAT_OUT}" \
        --model supercategory \
        --seed "${SEED}" \
        --include-es-all \
        --lambda-slopes "${LAMBDA_SLOPES}" \
        --method advi \
        --advi-steps "${ADVI_STEPS_SUPERCAT}" \
        --advi-log-every "${ADVI_LOG_EVERY}" \
        --advi-draws "${ADVI_DRAWS}"
  fi

  # 3) sklearn ridge baseline (optional).
  if [[ "${TRAIN_SKLEARN}" == "1" ]]; then
    SK_OUT="${RUN_DIR}/lib${lib}/${CAP}/sklearn_ridge_species_cluster"
    SK_COEFF="${SK_OUT}/stage1_coeff_summaries.npz"
    SK_LOG="${RUN_DIR}/logs/lib${lib}_${CAP}_sklearn_ridge_species_cluster.train.log"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${SK_COEFF}" ]]; then
      echo "[train] Skip sklearn ridge (exists): ${SK_COEFF}"
    else
      mkdir -p "${SK_OUT}"
      run_cmd "${SK_LOG}" \
        poetry run python -u scripts/pipelines/train_rt_stage1_coeff_summaries.py \
          --data-csv "${TRAIN_CSV}" \
          --output-dir "${SK_OUT}" \
          --seed "${SEED}" \
          --lambda-ridge "${LAMBDA_SLOPES}" \
          --include-es-all \
          --feature-center global \
          --feature-rotation none \
          --anchor-expansion none
    fi
  fi
done

echo "[train] Done. Next: ./scripts/run_rt_prod_eval.sh --run-dir ${RUN_DIR}"
