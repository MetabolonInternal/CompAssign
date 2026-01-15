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
LIB_IDS_DEFAULT=(208 209)
LIB_IDS=("${LIB_IDS_DEFAULT[@]}")
if [[ -n "${COMPASSIGN_RT_LIBS:-}" ]]; then
  IFS=',' read -r -a RAW_LIB_IDS <<<"${COMPASSIGN_RT_LIBS}"
  LIB_IDS=()
  for tok in "${RAW_LIB_IDS[@]}"; do
    tok="${tok//[[:space:]]/}"
    if [[ -n "${tok}" ]]; then
      LIB_IDS+=( "${tok}" )
    fi
  done
fi

RUN_DIR="output/rt_ridge_partial_pooling"
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${REPO_ROOT}/${RUN_DIR}"
fi

mkdir -p "${RUN_DIR}/logs"
echo "${RUN_DIR}" > "${REPO_ROOT}/output/rt_prod_latest.txt"

PYMC_METHOD="${COMPASSIGN_RT_PYMC_METHOD:-advi}"
if [[ "${PYMC_METHOD}" != "advi" && "${PYMC_METHOD}" != "map" ]]; then
  echo "ERROR: Invalid COMPASSIGN_RT_PYMC_METHOD=${PYMC_METHOD} (expected 'advi' or 'map')" >&2
  exit 2
fi
PYMC_ADVI_STEPS="${COMPASSIGN_RT_ADVI_STEPS:-}"
PYMC_ADVI_LOG_EVERY="${COMPASSIGN_RT_ADVI_LOG_EVERY:-}"
PYMC_ADVI_DRAWS="${COMPASSIGN_RT_ADVI_DRAWS:-}"
PYMC_MAP_MAXEVAL="${COMPASSIGN_RT_MAP_MAXEVAL:-}"

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
    PYMC_ARGS=(--model partial_pool --include-es-all --method "${PYMC_METHOD}" --lambda-slopes 3e-4)
    if [[ "${PYMC_METHOD}" == "advi" && -n "${PYMC_ADVI_STEPS}" ]]; then
      PYMC_ARGS+=(--advi-steps "${PYMC_ADVI_STEPS}")
    fi
    if [[ "${PYMC_METHOD}" == "advi" && -n "${PYMC_ADVI_LOG_EVERY}" ]]; then
      PYMC_ARGS+=(--advi-log-every "${PYMC_ADVI_LOG_EVERY}")
    fi
    if [[ "${PYMC_METHOD}" == "advi" && -n "${PYMC_ADVI_DRAWS}" ]]; then
      PYMC_ARGS+=(--advi-draws "${PYMC_ADVI_DRAWS}")
    fi
    if [[ "${PYMC_METHOD}" == "map" && -n "${PYMC_MAP_MAXEVAL}" ]]; then
      PYMC_ARGS+=(--map-maxeval "${PYMC_MAP_MAXEVAL}")
    fi
    run_cmd "${PARTIAL_LOG}" \
      poetry run python -u -m compassign.rt.train_rt_pymc_collapsed_ridge \
        --data-csv "${TRAIN_CSV}" \
        --output-dir "${PARTIAL_OUT}" \
        "${PYMC_ARGS[@]}"
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
