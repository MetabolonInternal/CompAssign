#!/usr/bin/env bash
set -uo pipefail

# Additional cap5->realtest experiments for pooling intercepts (comp_hier).
# Intended to answer: does intercept pooling help sparse training more than slope pooling?

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="output/rt_pymc_cap5_comp_hier_${TS}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "[run] RUN_DIR: ${RUN_DIR}"

run_cmd() {
  local log_path="$1"
  shift
  echo "[run] $*" | tee -a "${log_path}"
  "$@" >> "${log_path}" 2>&1
}

run_one() {
  local lib="$1"
  local cap="cap5"
  local train_csv="repo_export/lib${lib}/${cap}/merged_training_all_lib${lib}_${cap}_chemclass_rt_prod.csv"
  local test_csv="repo_export/lib${lib}/realtest/merged_training_realtest_lib${lib}_chemclass_rt_prod.csv"

  if [[ ! -f "${train_csv}" ]]; then
    echo "[run] Missing train_csv: ${train_csv}" | tee -a "${LOG_DIR}/missing_inputs.log"
    return 0
  fi
  if [[ ! -f "${test_csv}" ]]; then
    echo "[run] Missing test_csv: ${test_csv}" | tee -a "${LOG_DIR}/missing_inputs.log"
    return 0
  fi

  local model="pymc_explicit_comp_hier_intercept"
  local out_dir="${RUN_DIR}/lib${lib}/${cap}/${model}"
  local log="${LOG_DIR}/lib${lib}_${cap}_${model}.log"
  mkdir -p "${out_dir}"
  (
    run_cmd "${log}" poetry run python scripts/pipelines/train_rt_pymc_collapsed_ridge.py \
      --data-csv "${train_csv}" \
      --include-es-all --feature-center none --feature-rotation none \
      --anchor-expansion poly2 \
      --lambda-mode fixed --lambda-slopes 3e-4 --lambda-slopes-poly 1e-4 \
      --intercept-mode explicit --intercept-prior comp_hier --tau-comp-prior 0.5 \
      --method map --map-maxeval 50000 \
      --output-dir "${out_dir}" --seed 42
    run_cmd "${log}" poetry run python scripts/pipelines/eval_rt_coeff_summaries_by_support.py \
      --coeff-npz "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
      --test-csv "${test_csv}" \
      --chunk-size 50000 --max-test-rows 0 \
      --label realtest --require-seen-group
  ) || echo "[run] FAILED ${lib} ${cap} ${model}" | tee -a "${LOG_DIR}/failures.log"

  model="pymc_explicit_comp_hier_intercept_cluster_slope_head"
  out_dir="${RUN_DIR}/lib${lib}/${cap}/${model}"
  log="${LOG_DIR}/lib${lib}_${cap}_${model}.log"
  mkdir -p "${out_dir}"
  (
    run_cmd "${log}" poetry run python scripts/pipelines/train_rt_pymc_collapsed_ridge.py \
      --data-csv "${train_csv}" \
      --include-es-all --feature-center none --feature-rotation none \
      --anchor-expansion poly2 \
      --lambda-mode fixed --lambda-slopes 3e-4 --lambda-slopes-poly 1e-4 \
      --intercept-mode explicit --intercept-prior comp_hier --tau-comp-prior 0.5 \
      --slope-head-mode cluster --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
      --method map --map-maxeval 50000 \
      --output-dir "${out_dir}" --seed 42
    run_cmd "${log}" poetry run python scripts/pipelines/eval_rt_coeff_summaries_by_support.py \
      --coeff-npz "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
      --test-csv "${test_csv}" \
      --chunk-size 50000 --max-test-rows 0 \
      --label realtest --require-seen-group
  ) || echo "[run] FAILED ${lib} ${cap} ${model}" | tee -a "${LOG_DIR}/failures.log"
}

for lib in 208 209; do
  run_one "${lib}"
done

python scripts/pipelines/summarize_rt_overnight_results.py --run-dir "${RUN_DIR}"

