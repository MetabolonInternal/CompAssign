#!/usr/bin/env bash
set -uo pipefail

# Overnight RT experiments focusing on "where PyMC helps":
# - pooled slope heads (cluster / comp_id)
# - tail-sliced evaluation by training support (n_obs per group)
#
# Outputs: output/rt_pymc_overnight_YYYYMMDD_HHMMSS/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="output/rt_pymc_overnight_${TS}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "${RUN_DIR}" > "output/rt_pymc_overnight_latest.txt"
echo "[run] RUN_DIR: ${RUN_DIR}"

SUMMARY_CSV="${RUN_DIR}/summary.csv"
echo "lib,cap,model,rmse,cov95,rmse_bin_3_5,group_rmse_p90_bin_3_5,json_path" > "${SUMMARY_CSV}"

append_summary() {
  local lib="$1"
  local cap="$2"
  local model="$3"
  local json_path="$4"
  poetry run python - "${lib}" "${cap}" "${model}" "${json_path}" <<'PY' >> "${SUMMARY_CSV}"
import json
import sys

lib, cap, model, path = sys.argv[1:5]
with open(path, "r") as f:
    d = json.load(f)
m = d.get("metrics", {})
rmse = float(m.get("rmse", float("nan")))
cov = float(m.get("coverage_95", float("nan")))
tail_rmse = float("nan")
tail_p90 = float("nan")
for row in d.get("support_metrics", []):
    if row.get("support_bin") == "3-5":
        tail_rmse = float(row.get("rmse", float("nan")))
        tail_p90 = float(row.get("rmse_p90", float("nan")))
        break
print(f"{lib},{cap},{model},{rmse:.6f},{cov:.3f},{tail_rmse:.6f},{tail_p90:.6f},{path}")
PY
}

run_cmd() {
  local log_path="$1"
  shift
  echo "[run] $*" | tee -a "${log_path}"
  "$@" >> "${log_path}" 2>&1
}

run_one() {
  local lib="$1"
  local cap="$2"
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
  local json_path="${out_dir}/results/rt_eval_coeff_summaries_by_support_realtest.json"
  if [[ -f "${json_path}" ]]; then
    append_summary "${lib}" "${cap}" "${model}" "${json_path}"
  fi

  # PyMC collapsed ridge: poly2 anchors with fixed two-lambda (matches baseline RMSE; global sigma_y).
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
  json_path="${out_dir}/results/rt_eval_coeff_summaries_by_support_realtest.json"
  if [[ -f "${json_path}" ]]; then
    append_summary "${lib}" "${cap}" "${model}" "${json_path}"
  fi

  # PyMC pooled slopes by species_cluster: explores tail shrinkage (random-slope head).
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
  json_path="${out_dir}/results/rt_eval_coeff_summaries_by_support_realtest.json"
  if [[ -f "${json_path}" ]]; then
    append_summary "${lib}" "${cap}" "${model}" "${json_path}"
  fi

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
  json_path="${out_dir}/results/rt_eval_coeff_summaries_by_support_realtest.json"
  if [[ -f "${json_path}" ]]; then
    append_summary "${lib}" "${cap}" "${model}" "${json_path}"
  fi

  # Optional: comp_id slope head (can be large / unstable); keep last so failures don't block other runs.
  model="pymc_explicit_flat_compid_slope_head_tau0p5"
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
      --slope-head-mode comp_id --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
      --method advi --advi-steps 8000 --advi-draws 200 \
      --output-dir "${out_dir}" --seed 42
    run_cmd "${log}" poetry run python scripts/pipelines/eval_rt_coeff_summaries_by_support.py \
      --coeff-npz "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
      --test-csv "${test_csv}" \
      --chunk-size 50000 --max-test-rows 0 \
      --label realtest --require-seen-group
  ) || echo "[run] FAILED ${lib} ${cap} ${model}" | tee -a "${LOG_DIR}/failures.log"
  json_path="${out_dir}/results/rt_eval_coeff_summaries_by_support_realtest.json"
  if [[ -f "${json_path}" ]]; then
    append_summary "${lib}" "${cap}" "${model}" "${json_path}"
  fi
}

for lib in 208 209; do
  run_one "${lib}" "cap100"
done


echo "[run] DONE. Summary: ${SUMMARY_CSV}"
echo "[run] Logs: ${LOG_DIR}"
