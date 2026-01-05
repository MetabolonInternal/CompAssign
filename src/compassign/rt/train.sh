#!/usr/bin/env bash
set -euo pipefail

# Train the RT ridge models used for peak assignment on production-style RT CSVs.
#
# This script trains:
#   1) PyMC ridge (partial pooling; recommended)
#   2) sklearn ridge (supercategory; baseline comparison)
#
# The output directory is a "run dir" compatible with:
#   poetry run python -m compassign.rt.plot_rt_multilevel_results
#
# Intended usage: run with no args (defaults encoded below).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -ne 0 ]]; then
  echo "ERROR: src/compassign/rt/train.sh does not take arguments. Run it with no args." >&2
  exit 2
fi

CAP="cap100"
LIB_IDS=(208 209)

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="output/rt_prod_${TS}"
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${REPO_ROOT}/${RUN_DIR}"
fi

mkdir -p "${RUN_DIR}/logs"
echo "${RUN_DIR}" > "${REPO_ROOT}/output/rt_prod_latest.txt"

run_cmd() {
  local log_path="$1"
  shift
  echo "" | tee -a "${log_path}" >/dev/null
  echo "[run] $*" | tee -a "${log_path}"
  # shellcheck disable=SC2068
  "$@" 2>&1 | tee -a "${log_path}"
}

for lib in "${LIB_IDS[@]}"; do
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
  if [[ -f "${PARTIAL_COEFF}" ]]; then
    echo "[train] Skip partial pooling (exists): ${PARTIAL_COEFF}"
  else
    run_cmd "${PARTIAL_LOG}" \
      poetry run python -u -m compassign.rt.train_rt_pymc_collapsed_ridge \
        --data-csv "${TRAIN_CSV}" \
        --output-dir "${PARTIAL_OUT}" \
        --model partial_pool \
        --include-es-all \
        --method advi \
        --lambda-slopes 3e-4
  fi

  # 2) sklearn ridge baseline.
  SK_OUT="${RUN_DIR}/lib${lib}/${CAP}/sklearn_ridge_species_cluster"
  SK_COEFF="${SK_OUT}/stage1_coeff_summaries.npz"
  SK_LOG="${RUN_DIR}/logs/lib${lib}_${CAP}_sklearn_ridge_species_cluster.train.log"
  if [[ -f "${SK_COEFF}" ]]; then
    echo "[train] Skip sklearn ridge (exists): ${SK_COEFF}"
  else
    mkdir -p "${SK_OUT}"
    run_cmd "${SK_LOG}" \
      poetry run python -u -m compassign.rt.train_rt_stage1_coeff_summaries \
        --data-csv "${TRAIN_CSV}" \
        --output-dir "${SK_OUT}" \
        --include-es-all \
        --lambda-ridge 3e-4 \
        --feature-center global \
        --feature-rotation none \
        --anchor-expansion none
  fi
done

echo "[train] Done. Run dir: ${RUN_DIR}"
