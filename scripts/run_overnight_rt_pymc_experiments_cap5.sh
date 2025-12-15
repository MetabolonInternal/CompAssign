#!/usr/bin/env bash
set -uo pipefail

# Smaller / tail-focused grid (cap5 -> realtest) to test whether pooling helps when each group is sparse.
#
# This reuses the same model set as scripts/run_overnight_rt_pymc_experiments.sh but trains on cap5.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="output/rt_pymc_cap5_grid_${TS}"
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

  # Baseline: sklearn stage1 coeff summaries (poly2 anchors; two-lambda)
  local model="sklearn_poly2_anchors_two_lambda"
  local out_dir="${RUN_DIR}/lib${lib}/${cap}/${model}"
  local log="${LOG_DIR}/lib${lib}_${cap}_${model}.log"
  mkdir -p "${out_dir}"
  (
    run_cmd "${log}" poetry run python scripts/pipelines/train_rt_stage1_coeff_summaries.py \
      --data-csv "${train_csv}" \
      --include-es-all --feature-center none --feature-rotation none \
      --lambda-ridge 3e-4 --anchor-expansion poly2 --lambda-ridge-poly 1e-4 \
      --output-dir "${out_dir}" --seed 42
    run_cmd "${log}" poetry run python scripts/pipelines/eval_rt_coeff_summaries_by_support.py \
      --coeff-npz "${out_dir}/stage1_coeff_summaries.npz" \
      --test-csv "${test_csv}" \
      --chunk-size 50000 --max-test-rows 0 \
      --label realtest --require-seen-group
  ) || echo "[run] FAILED ${lib} ${cap} ${model}" | tee -a "${LOG_DIR}/failures.log"

  # PyMC collapsed ridge: poly2 anchors with fixed two-lambda (matches baseline point RMSE; global sigma_y).
  model="pymc_collapsed_poly2_fixed_two_lambda"
  out_dir="${RUN_DIR}/lib${lib}/${cap}/${model}"
  log="${LOG_DIR}/lib${lib}_${cap}_${model}.log"
  mkdir -p "${out_dir}"
  (
    run_cmd "${log}" poetry run python scripts/pipelines/train_rt_pymc_collapsed_ridge.py \
      --data-csv "${train_csv}" \
      --include-es-all --feature-center none --feature-rotation none \
      --anchor-expansion poly2 \
      --lambda-mode fixed --lambda-slopes 3e-4 --lambda-slopes-poly 1e-4 \
      --intercept-mode collapsed --method map --map-maxeval 20000 \
      --output-dir "${out_dir}" --seed 42
    run_cmd "${log}" poetry run python scripts/pipelines/eval_rt_coeff_summaries_by_support.py \
      --coeff-npz "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
      --test-csv "${test_csv}" \
      --chunk-size 50000 --max-test-rows 0 \
      --label realtest --require-seen-group
  ) || echo "[run] FAILED ${lib} ${cap} ${model}" | tee -a "${LOG_DIR}/failures.log"

  # PyMC pooled slopes by species_cluster: should be most useful when cap is small (tail-like training).
  model="pymc_explicit_flat_cluster_slope_head_tau0p5"
  out_dir="${RUN_DIR}/lib${lib}/${cap}/${model}"
  log="${LOG_DIR}/lib${lib}_${cap}_${model}.log"
  mkdir -p "${out_dir}"
  (
    run_cmd "${log}" poetry run python scripts/pipelines/train_rt_pymc_collapsed_ridge.py \
      --data-csv "${train_csv}" \
      --include-es-all --feature-center none --feature-rotation none \
      --anchor-expansion poly2 \
      --lambda-mode fixed --lambda-slopes 3e-4 --lambda-slopes-poly 1e-4 \
      --intercept-mode explicit --intercept-prior flat \
      --slope-head-mode cluster --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
      --method map --map-maxeval 50000 \
      --output-dir "${out_dir}" --seed 42
    run_cmd "${log}" poetry run python scripts/pipelines/eval_rt_coeff_summaries_by_support.py \
      --coeff-npz "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
      --test-csv "${test_csv}" \
      --chunk-size 50000 --max-test-rows 0 \
      --label realtest --require-seen-group
  ) || echo "[run] FAILED ${lib} ${cap} ${model}" | tee -a "${LOG_DIR}/failures.log"

  model="pymc_explicit_flat_cluster_slope_head_tau0p1"
  out_dir="${RUN_DIR}/lib${lib}/${cap}/${model}"
  log="${LOG_DIR}/lib${lib}_${cap}_${model}.log"
  mkdir -p "${out_dir}"
  (
    run_cmd "${log}" poetry run python scripts/pipelines/train_rt_pymc_collapsed_ridge.py \
      --data-csv "${train_csv}" \
      --include-es-all --feature-center none --feature-rotation none \
      --anchor-expansion poly2 \
      --lambda-mode fixed --lambda-slopes 3e-4 --lambda-slopes-poly 1e-4 \
      --intercept-mode explicit --intercept-prior flat \
      --slope-head-mode cluster --tau-w-prior 0.1 --w0-prior-sigma 1.0 \
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

