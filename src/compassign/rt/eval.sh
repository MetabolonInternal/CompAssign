#!/usr/bin/env bash
set -euo pipefail

# Evaluate trained RT ridge coefficient-summary artifacts on realtest and regenerate plots.
#
# This script expects a run directory created by:
#   ./src/compassign/rt/train.sh
#
# Intended usage: run with no args (defaults encoded below).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -ne 0 ]]; then
  echo "ERROR: src/compassign/rt/eval.sh does not take arguments. Run it with no args." >&2
  exit 2
fi

CAP="cap100"
LIB_IDS=(208 209)
CHUNK_SIZE="200000"
LOG_EVERY_CHUNKS="5"

LATEST_PATH="${REPO_ROOT}/output/rt_prod_latest.txt"
if [[ ! -f "${LATEST_PATH}" ]]; then
  echo "ERROR: Missing run dir pointer: ${LATEST_PATH} (run ./src/compassign/rt/train.sh first)" >&2
  exit 2
fi
RUN_DIR="$(cat "${LATEST_PATH}" | tr -d '[:space:]')"
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
  local common_support_map_csv="$4"
  local log_path="$5"

  local out_json=""
  if [[ "${coeff_npz}" == */models/stage1_coeff_summaries_posterior.npz ]]; then
    out_json="$(dirname "$(dirname "${coeff_npz}")")/results/rt_eval_coeff_summaries_by_support_realtest.json"
  else
    out_json="$(dirname "${coeff_npz}")/results/rt_eval_coeff_summaries_by_support_realtest.json"
  fi
  if [[ -f "${out_json}" ]]; then
    if poetry run python - "${out_json}" "${common_support_map_csv}" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

out_json = Path(sys.argv[1])
expected = str(Path(sys.argv[2]).resolve())
d = json.loads(out_json.read_text())
ok = d.get("common_support_map_csv") == expected
raise SystemExit(0 if ok else 1)
PY
    then
      echo "[eval] Skip (exists, common support ok): ${out_json}"
      return 0
    fi
  fi

  run_cmd "${log_path}" \
    poetry run python -u -m compassign.rt.eval_rt_coeff_summaries_by_support \
      --coeff-npz "${coeff_npz}" \
      --test-csv "${test_csv}" \
      --chunk-size "${CHUNK_SIZE}" \
      --log-every-chunks "${LOG_EVERY_CHUNKS}" \
      --common-support-map-csv "${common_support_map_csv}" \
      --label realtest
}

eval_lasso_supercat() {
  local lib_id="$1"
  local output_dir="$2"
  local test_csv="$3"
  local support_map_csv="$4"
  local log_path="$5"

  local out_json="${output_dir}/results/rt_eval_lasso_realtest.json"
  if [[ -f "${out_json}" ]]; then
    if poetry run python - "${out_json}" "${support_map_csv}" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

out_json = Path(sys.argv[1])
expected = str(Path(sys.argv[2]).resolve())
d = json.loads(out_json.read_text())
ok = d.get("support_map_csv") == expected
raise SystemExit(0 if ok else 1)
PY
    then
      echo "[eval] Skip (exists, support map ok): ${out_json}"
      return 0
    fi
  fi

  local -a cmd=(
    poetry run python -u -m compassign.rt.eval_rt_lasso_baseline_by_species_cluster
      --output-dir "${output_dir}"
      --lib-id "${lib_id}"
      --test-csv "${test_csv}"
      --chunk-size "${CHUNK_SIZE}"
      --label realtest
      --window-multiplier 4
      --min-window 0.001
  )
  if [[ -n "${support_map_csv}" && -f "${support_map_csv}" ]]; then
    cmd+=(--support-map-csv "${support_map_csv}")
  fi
  run_cmd "${log_path}" "${cmd[@]}"
}

for lib in "${LIB_IDS[@]}"; do
  TEST_CSV="repo_export/lib${lib}/realtest/merged_training_realtest_lib${lib}_chemclass_rt_prod.csv"
  if [[ ! -f "${TEST_CSV}" ]]; then
    echo "ERROR: Missing realtest CSV: ${TEST_CSV}" >&2
    exit 2
  fi

  echo "[eval] lib${lib} ${CAP}"

  TRAIN_CSV="repo_export/lib${lib}/${CAP}/merged_training_all_lib${lib}_${CAP}_chemclass_rt_prod.csv"
  if [[ ! -f "${TRAIN_CSV}" ]]; then
    echo "ERROR: Missing train CSV for common support bins: ${TRAIN_CSV}" >&2
    exit 2
  fi
  COMMON_SUPPORT_MAP="${RUN_DIR}/lib${lib}/${CAP}/support_map_species_cluster_comp_${CAP}.csv"
  if [[ ! -f "${COMMON_SUPPORT_MAP}" ]]; then
    echo "[eval] Building common support map: ${COMMON_SUPPORT_MAP}"
    run_cmd "${RUN_DIR}/logs/lib${lib}_${CAP}_common_support_map.build.log" \
      poetry run python -u -m compassign.rt.build_support_map_species_cluster_comp \
        --train-csv "${TRAIN_CSV}" \
        --out-csv "${COMMON_SUPPORT_MAP}"
  fi

  PARTIAL_COEFF="${RUN_DIR}/lib${lib}/${CAP}/features_none/pymc_pooled_species_comp_hier_supercat_cluster_supercat/models/stage1_coeff_summaries_posterior.npz"
  if [[ -f "${PARTIAL_COEFF}" ]]; then
    eval_model "${lib}" "${PARTIAL_COEFF}" "${TEST_CSV}" "${COMMON_SUPPORT_MAP}" \
      "${RUN_DIR}/logs/lib${lib}_${CAP}_none_pymc_pooled_species_comp_hier_supercat_cluster_supercat.eval.log"
  else
    echo "[eval] Skip missing partial coeff: ${PARTIAL_COEFF}"
  fi

  SK_COEFF="${RUN_DIR}/lib${lib}/${CAP}/sklearn_ridge_species_cluster/stage1_coeff_summaries.npz"
  if [[ -f "${SK_COEFF}" ]]; then
    eval_model "${lib}" "${SK_COEFF}" "${TEST_CSV}" "${COMMON_SUPPORT_MAP}" \
      "${RUN_DIR}/logs/lib${lib}_${CAP}_sklearn_ridge_species_cluster.eval.log"
  fi

  # Lasso supercategory baseline (external eslasso models).
  LASSO_OUT="${RUN_DIR}/lib${lib}/${CAP}/lasso_eslasso_species_cluster"
  eval_lasso_supercat "${lib}" "${LASSO_OUT}" "${TEST_CSV}" "${COMMON_SUPPORT_MAP}" \
	"${RUN_DIR}/logs/lib${lib}_${CAP}_lasso_eslasso_species_cluster.eval.log"
done

echo "[eval] Plotting (global + by species_cluster)"
LIBS_CSV="$(IFS=,; echo "${LIB_IDS[*]}")"
poetry run python -u -m compassign.rt.plot_rt_multilevel_results \
  --run-dir "${RUN_DIR}" \
  --cap "${CAP}" \
  --libs "${LIBS_CSV}" \
  --anchor none \
  --tag full

echo "[eval] Syncing report plot images"
REPORT_IMG_DIR="${REPO_ROOT}/docs/models/images/rt_pymc_multilevel_pooling_report"
mkdir -p "${REPORT_IMG_DIR}"
PLOTS_DIR="${RUN_DIR}/plots"
if [[ ! -d "${PLOTS_DIR}" ]]; then
  echo "ERROR: Missing plots dir: ${PLOTS_DIR}" >&2
  exit 2
fi
for lib in "${LIB_IDS[@]}"; do
  for stem in global_comparison by_species_cluster; do
    src="${PLOTS_DIR}/lib${lib}_${stem}_anchor_none_full.png"
    if [[ ! -f "${src}" ]]; then
      echo "ERROR: Missing plot: ${src}" >&2
      exit 2
    fi
    cp -f "${src}" "${REPORT_IMG_DIR}/"
  done
done

echo "[eval] Building report PDF"
(cd "${REPO_ROOT}/docs/models" && latexmk -xelatex -interaction=nonstopmode rt_pymc_multilevel_pooling_report.tex)

echo "[eval] Done."
