#!/usr/bin/env bash
set -euo pipefail

# End-to-end cap100 experiments for multi-level RT modeling.
#
# Default run is intended to be "one and done" for choosing a model form:
# - No pooling at species level: one model per (species_cluster, comp_id)
# - Unpooled species model: one model per (species, comp_id)
# - Partial pooling variants (species nested in species_cluster):
#     - intercept pooling only
#     - slope pooling only
#     - intercept + slope pooling
# - Chemistry-informed intercept hierarchy (ChemBERTa PCA-20 embeddings + compound_class head)
# - Legacy eslasso baseline (external_repos/sally/new_models/eslasso)
#
# Writes: output/rt_pymc_multilevel_cap100_YYYYMMDD_HHMMSS/
#
# Usage:
#   bash scripts/run_rt_pymc_multilevel_cap100.sh
#   bash scripts/run_rt_pymc_multilevel_cap100.sh --libs 209 --anchor-expansions none

usage() {
  cat <<'EOF'
Usage: run_rt_pymc_multilevel_cap100.sh [options]

Runs end-to-end training + evaluation for multilevel RT models (cap -> realtest).

Options:
  --parallel [n]               Run jobs in parallel (default: use all CPU cores).
                               Optional numeric arg caps parallelism.
  --libs <csv>                 Comma-separated libs (default: 208,209)
  --cap <capN>                 Training cap directory (default: cap100)
  --run-dir <path>             Output directory (default: output/rt_pymc_multilevel_cap100_<timestamp>)
  --seed <int>                 Random seed (default: 42)

  --models <csv>               Comma-separated (default: baseline,subgroup,pooled_intercepts,pooled,chem_hier,lasso)
                               - baseline:      (species_cluster, comp_id) collapsed ridge
                               - subgroup:      (species, comp_id) collapsed ridge
                               - pooled_*:      (species, comp_id) explicit intercepts with supercategory-aware pooling
                               - comp_hier:     (species, comp_id) explicit intercepts with compound hierarchy
                               - chem_hier:     (species, comp_id) explicit intercepts with chemistry hierarchy (ChemBERTa PCA-20)
                               - lasso:         legacy eslasso baseline by species_cluster (evaluation only)
  --method <advi|map|nuts>     Inference method for PyMC (default: advi)
  --map-maxeval <int>          Max evaluations for MAP optimization (default: 50000)
  --max-train-rows <int>       Cap training rows (default: 0 = all)
  --max-groups <int>           Cap number of (group_id, comp_id) groups (default: 0 = all)
  --max-test-rows <int>        Cap evaluation rows (default: 0 = all)

  --anchor-expansions <csv>    Comma-separated: none,poly2 (default: none)
                               By default (when this flag is not provided), baseline also runs poly2 once for comparison.
  --include-es-all             Include all ES_* covariates (default)
  --no-include-es-all          Exclude ES_* covariates

  --chem-embeddings-path <p>   Chemistry embedding parquet for chem_hier

  --advi-steps <int>           ADVI steps (overrides defaults; see below)
  --advi-draws <int>           ADVI posterior draws (default: 50)
  --advi-log-every <int>       Print ADVI progress every N iters (default: 1000)
                               Defaults when --advi-steps is not set:
                               - collapsed models: 5000 steps
                               - explicit hierarchical models: 10000 steps

  --chunk-size <int>           Eval CSV chunk size (default: 200000)
  --log-every-chunks <int>     Eval progress frequency (default: 5)

  --lambda-mode <fixed|learn>  Ridge precision handling (default: fixed)
  --lambda-slopes <float>      Ridge penalty for slopes (default: 3e-4)
  --lambda-slopes-poly <float> Ridge penalty for poly2 terms (default: 1e-4)

  --skip-existing              Skip jobs with existing eval JSON (default)
  --no-skip-existing           Re-run even if outputs exist
  --rebuild-summary            Rebuild summary.csv from existing JSONs

  -h, --help                   Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LIBS_CSV="208,209"
CAP="cap100"
RUN_DIR=""
SEED="42"

MODELS_CSV="baseline,subgroup,pooled_intercepts,pooled,chem_hier,lasso"
METHOD="advi"
MAP_MAXEVAL="50000"
MAX_TRAIN_ROWS="0"
MAX_GROUPS="0"
MAX_TEST_ROWS="0"

ANCHOR_EXPANSIONS_CSV="none"
INCLUDE_ES_ALL="1"
CHEM_EMBEDDINGS_PATH="resources/metabolites/embeddings_chemberta_pca20.parquet"

ADVI_STEPS="10000"
ADVI_DRAWS="50"
ADVI_LOG_EVERY="1000"
ADVI_STEPS_COLLAPSED_DEFAULT="5000"
ADVI_STEPS_EXPLICIT_DEFAULT="10000"
ADVI_STEPS_SET="0"

CHUNK_SIZE="200000"
LOG_EVERY_CHUNKS="5"

SKIP_EXISTING="1"
REBUILD_SUMMARY="0"

LAMBDA_MODE="fixed"
LAMBDA_SLOPES="3e-4"
LAMBDA_SLOPES_POLY="1e-4"

PARALLEL="0"
PARALLEL_N=""

LIBS_SET="0"
MODELS_SET="0"
ANCHOR_EXPANSIONS_SET="0"
INCLUDE_ES_ALL_SET="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      PARALLEL="1"
      # Optional value: if next arg is a positive integer, treat as N.
      if [[ $# -ge 2 && "${2:-}" =~ ^[0-9]+$ && "${2:-}" -gt 0 ]]; then
        PARALLEL_N="${2}"
        shift 2
      else
        shift 1
      fi
      ;;
    --libs)
      LIBS_CSV="${2:-}"
      LIBS_SET="1"
      shift 2
      ;;
    --cap)
      CAP="${2:-}"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --models)
      MODELS_CSV="${2:-}"
      MODELS_SET="1"
      shift 2
      ;;
    --method)
      METHOD="${2:-}"
      shift 2
      ;;
    --map-maxeval)
      MAP_MAXEVAL="${2:-}"
      shift 2
      ;;
    --max-train-rows)
      MAX_TRAIN_ROWS="${2:-}"
      shift 2
      ;;
    --max-groups)
      MAX_GROUPS="${2:-}"
      shift 2
      ;;
    --max-test-rows)
      MAX_TEST_ROWS="${2:-}"
      shift 2
      ;;
    --anchor-expansions)
      ANCHOR_EXPANSIONS_CSV="${2:-}"
      ANCHOR_EXPANSIONS_SET="1"
      shift 2
      ;;
    --include-es-all)
      INCLUDE_ES_ALL="1"
      INCLUDE_ES_ALL_SET="1"
      shift 1
      ;;
    --no-include-es-all)
      INCLUDE_ES_ALL="0"
      INCLUDE_ES_ALL_SET="1"
      shift 1
      ;;
    --chem-embeddings-path)
      CHEM_EMBEDDINGS_PATH="${2:-}"
      shift 2
      ;;
    --advi-steps)
      # Applies to both collapsed + explicit models.
      ADVI_STEPS="${2:-}"
      ADVI_STEPS_SET="1"
      shift 2
      ;;
    --advi-draws)
      ADVI_DRAWS="${2:-}"
      shift 2
      ;;
    --advi-log-every)
      ADVI_LOG_EVERY="${2:-}"
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
    --lambda-mode)
      LAMBDA_MODE="${2:-}"
      shift 2
      ;;
    --lambda-slopes)
      LAMBDA_SLOPES="${2:-}"
      shift 2
      ;;
    --lambda-slopes-poly)
      LAMBDA_SLOPES_POLY="${2:-}"
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
    --rebuild-summary)
      REBUILD_SUMMARY="1"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[run] ERROR: unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="output/rt_pymc_multilevel_cap100_${TS}"
fi

detect_cores() {
  local n=""
  if command -v getconf >/dev/null 2>&1; then
    n="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  fi
  if [[ -z "${n}" && "$(uname -s 2>/dev/null || echo "")" == "Darwin" ]]; then
    if command -v sysctl >/dev/null 2>&1; then
      n="$(sysctl -n hw.ncpu 2>/dev/null || true)"
    fi
  fi
  if [[ -z "${n}" ]] && command -v nproc >/dev/null 2>&1; then
    n="$(nproc 2>/dev/null || true)"
  fi
  if [[ -z "${n}" || ! "${n}" =~ ^[0-9]+$ || "${n}" -le 0 ]]; then
    n="1"
  fi
  echo "${n}"
}

MAX_PARALLEL="1"
DEFER_SUMMARY="0"
if [[ "${PARALLEL}" == "1" ]]; then
  if [[ -n "${PARALLEL_N}" ]]; then
    MAX_PARALLEL="${PARALLEL_N}"
  else
    MAX_PARALLEL="$(detect_cores)"
  fi
  if [[ ! "${MAX_PARALLEL}" =~ ^[0-9]+$ || "${MAX_PARALLEL}" -le 0 ]]; then
    echo "[run] ERROR: invalid --parallel value '${MAX_PARALLEL}' (must be positive int)" >&2
    exit 2
  fi
  if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
    DEFER_SUMMARY="1"
  fi
fi

if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
  # Avoid accidental oversubscription when running multiple jobs: keep BLAS/OpenMP single-threaded per job.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  echo "[run] THREADS: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS}"
fi

LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
echo "${RUN_DIR}" > "output/rt_pymc_multilevel_cap100_latest.txt"

echo "[run] RUN_DIR: ${RUN_DIR}"
echo "[run] CAP: ${CAP}"
echo "[run] MODELS: ${MODELS_CSV}"
echo "[run] METHOD: ${METHOD}"
echo "[run] PARALLEL: enabled=${PARALLEL} max_parallel=${MAX_PARALLEL}"
if [[ "${METHOD}" == "advi" ]]; then
  if [[ "${ADVI_STEPS_SET}" == "1" ]]; then
    echo "[run] ADVI: steps=${ADVI_STEPS} draws=${ADVI_DRAWS} log_every=${ADVI_LOG_EVERY}"
  else
    echo "[run] ADVI: steps_collapsed=${ADVI_STEPS_COLLAPSED_DEFAULT} steps_explicit=${ADVI_STEPS_EXPLICIT_DEFAULT} draws=${ADVI_DRAWS} log_every=${ADVI_LOG_EVERY}"
  fi
fi
echo "[run] EVAL: chunk_size=${CHUNK_SIZE} log_every_chunks=${LOG_EVERY_CHUNKS}"
echo "[run] LIMITS: max_train_rows=${MAX_TRAIN_ROWS} max_groups=${MAX_GROUPS} max_test_rows=${MAX_TEST_ROWS}"
echo "[run] FEATURES: include_es_all=${INCLUDE_ES_ALL} anchor_expansions=${ANCHOR_EXPANSIONS_CSV}"
echo "[run] CHEM: embeddings_path=${CHEM_EMBEDDINGS_PATH}"
echo "[run] RIDGE: lambda_mode=${LAMBDA_MODE} lambda_slopes=${LAMBDA_SLOPES} lambda_slopes_poly=${LAMBDA_SLOPES_POLY}"
echo "[run] RESUME: skip_existing=${SKIP_EXISTING} rebuild_summary=${REBUILD_SUMMARY}"
if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
  echo "[run] Parallel mode: per-job stdout/stderr is written to logs/*.log (use tail -f)."
fi

SUMMARY_CSV="${RUN_DIR}/summary.csv"
LASSO_SUMMARY_CSV="${RUN_DIR}/summary_lasso.csv"

write_summary_header() {
  echo "lib,cap,anchor_expansion,model,group_col,n_used,n_test,skipped_missing_group,rmse,mae,cov95,pred_std_mean,interval_width_mean,rmse_le1,rmse_p90_le1,rmse_2,rmse_p90_2,rmse_3_5,rmse_p90_3_5,json_path" > "${SUMMARY_CSV}"
}

write_lasso_summary_header() {
  echo "lib,cap,rmse,mae,cov95,n_test_rows_seen,n_rows_evaluated,skipped_no_super,skipped_no_model,json_path" > "${LASSO_SUMMARY_CSV}"
}

summary_has_row() {
  local lib="$1"
  local cap="$2"
  local anchor_expansion="$3"
  local model="$4"
  if [[ ! -f "${SUMMARY_CSV}" ]]; then
    return 1
  fi
  if command -v rg >/dev/null 2>&1; then
    rg -q "^${lib},${cap},${anchor_expansion},${model}," "${SUMMARY_CSV}"
  else
    grep -q "^${lib},${cap},${anchor_expansion},${model}," "${SUMMARY_CSV}"
  fi
}

model_enabled() {
  local want="$1"
  shift
  local m
  for m in "$@"; do
    if [[ "${m}" == "${want}" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "${REBUILD_SUMMARY}" == "1" ]]; then
  write_summary_header
elif [[ ! -f "${SUMMARY_CSV}" ]]; then
  write_summary_header
fi

count_csv_rows() {
  local csv="$1"
  local n
  n="$(wc -l < "${csv}")"
  if [[ "${n}" -le 0 ]]; then
    echo "0"
  else
    # subtract header
    echo "$((n - 1))"
  fi
}

append_summary() {
  if [[ "${DEFER_SUMMARY}" == "1" && "${FORCE_SUMMARY:-0}" != "1" ]]; then
    return 0
  fi
  local lib="$1"
  local cap="$2"
  local anchor_expansion="$3"
  local model="$4"
  local json_path="$5"
  if summary_has_row "${lib}" "${cap}" "${anchor_expansion}" "${model}"; then
    return 0
  fi
  poetry run python - "${lib}" "${cap}" "${anchor_expansion}" "${model}" "${json_path}" <<'PY' >> "${SUMMARY_CSV}"
import json
import sys

lib, cap, anchor_expansion, model, path = sys.argv[1:6]
with open(path, "r") as f:
    d = json.load(f)
m = d.get("metrics", {})
rmse = float(m.get("rmse", float("nan")))
mae = float(m.get("mae", float("nan")))
cov = float(m.get("coverage_95", float("nan")))
pred_std_mean = float(m.get("pred_std_mean", float("nan")))
interval_width_mean = float(m.get("interval_width_mean", float("nan")))
n_used = int(d.get("n_used", 0))
n_test = int(d.get("n_test", 0))
skipped = int(d.get("skipped_missing_group", 0))
group_col = str(d.get("group_col", "species_cluster"))

def pick(label: str, key: str) -> float:
    for row in d.get("support_metrics", []):
        if row.get("support_bin") == label:
            return float(row.get(key, float("nan")))
    return float("nan")

rmse_le1 = pick("<= 1", "rmse")
p90_le1 = pick("<= 1", "rmse_p90")
rmse_2 = pick("2-2", "rmse")
if rmse_2 != rmse_2:  # nan
    rmse_2 = pick("2", "rmse")
p90_2 = pick("2-2", "rmse_p90")
if p90_2 != p90_2:  # nan
    p90_2 = pick("2", "rmse_p90")
rmse_3_5 = pick("3-5", "rmse")
p90_3_5 = pick("3-5", "rmse_p90")

print(
    f"{lib},{cap},{anchor_expansion},{model},{group_col},{n_used},{n_test},{skipped},"
    f"{rmse:.6f},{mae:.6f},{cov:.3f},{pred_std_mean:.6f},{interval_width_mean:.6f},"
    f"{rmse_le1:.6f},{p90_le1:.6f},{rmse_2:.6f},{p90_2:.6f},{rmse_3_5:.6f},{p90_3_5:.6f},{path}"
)
PY
}

append_lasso_summary() {
  local lib="$1"
  local cap="$2"
  local json_path="$3"
  if [[ ! -f "${LASSO_SUMMARY_CSV}" ]]; then
    write_lasso_summary_header
  fi
  if command -v rg >/dev/null 2>&1; then
    if rg -q "^${lib},${cap}," "${LASSO_SUMMARY_CSV}"; then
      return 0
    fi
  else
    if grep -q "^${lib},${cap}," "${LASSO_SUMMARY_CSV}"; then
      return 0
    fi
  fi
  poetry run python - "${lib}" "${cap}" "${json_path}" <<'PY' >> "${LASSO_SUMMARY_CSV}"
import json
import sys

lib, cap, path = sys.argv[1:4]
with open(path, "r") as f:
    d = json.load(f)
m = d.get("metrics", {})
rmse = float(m.get("rmse", float("nan")))
mae = float(m.get("mae", float("nan")))
cov = float(m.get("coverage_95", float("nan")))
print(
    f"{lib},{cap},{rmse:.6g},{mae:.6g},{cov:.6g},"
    f"{int(d.get('n_test_rows_seen', 0))},{int(d.get('n_rows_evaluated', 0))},"
    f"{int(d.get('skipped_no_super', 0))},{int(d.get('skipped_no_model', 0))},{path}"
)
PY
}

run_cmd() {
  local log_path="$1"
  shift
  if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
    {
      echo ""
      echo "[run] $*"
      "$@"
      echo ""
    } >> "${log_path}" 2>&1
  else
    echo "" | tee -a "${log_path}"
    echo "[run] $*" | tee -a "${log_path}"
    # Keep output visible in the foreground (progress bars / ETA), and also log it.
    "$@" 2>&1 | tee -a "${log_path}"
    echo "" | tee -a "${log_path}"
  fi
}

train_and_eval() {
  local lib="$1"
  local cap="$2"
  local anchor_expansion="$3"
  local model="$4"
  local out_dir="$5"
  local coeff_npz="$6"
  local train_log="$7"
  local eval_log="$8"
  local n_test_rows="$9"
  shift 9

  mkdir -p "${out_dir}"

  local eval_max_rows="${n_test_rows}"
  if [[ "${MAX_TEST_ROWS}" -gt 0 && "${MAX_TEST_ROWS}" -lt "${n_test_rows}" ]]; then
    eval_max_rows="${MAX_TEST_ROWS}"
  fi

  local train_csv="repo_export/lib${lib}/${cap}/merged_training_all_lib${lib}_${cap}_chemclass_rt_prod.csv"
  local test_csv="repo_export/lib${lib}/realtest/merged_training_realtest_lib${lib}_chemclass_rt_prod.csv"

  local json_path="${out_dir}/results/rt_eval_coeff_summaries_by_support_realtest.json"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${json_path}" ]]; then
    echo "[run] SKIP (already evaluated): ${json_path}" | tee -a "${eval_log}"
    append_summary "${lib}" "${cap}" "${anchor_expansion}" "${model}" "${json_path}"
    return 0
  fi

  if [[ -f "${coeff_npz}" ]]; then
    echo "[run] SKIP TRAIN (coeff exists): ${coeff_npz}" | tee -a "${train_log}"
  else
    local -a train_cmd=(
      poetry run python -u scripts/pipelines/train_rt_pymc_collapsed_ridge.py
      --data-csv "${train_csv}"
      --output-dir "${out_dir}"
      --seed "${SEED}"
    )
    if [[ "${MAX_TRAIN_ROWS}" -gt 0 ]]; then
      train_cmd+=(--max-train-rows "${MAX_TRAIN_ROWS}")
    fi
    if [[ "${MAX_GROUPS}" -gt 0 ]]; then
      train_cmd+=(--max-groups "${MAX_GROUPS}")
    fi
    train_cmd+=("$@")
    run_cmd "${train_log}" "${train_cmd[@]}"
  fi

  if [[ -f "${json_path}" ]]; then
    echo "[run] SKIP EVAL (json exists): ${json_path}" | tee -a "${eval_log}"
  else
    run_cmd "${eval_log}" poetry run python -u scripts/pipelines/eval_rt_coeff_summaries_by_support.py \
      --coeff-npz "${coeff_npz}" \
      --test-csv "${test_csv}" \
      --chunk-size "${CHUNK_SIZE}" \
      --max-test-rows "${eval_max_rows}" \
      --log-every-chunks "${LOG_EVERY_CHUNKS}" \
      --label realtest
  fi

  if [[ -f "${json_path}" ]]; then
    append_summary "${lib}" "${cap}" "${anchor_expansion}" "${model}" "${json_path}"
  fi
}

eval_lasso() {
  local lib="$1"
  local cap="$2"
  local out_dir="$3"
  local eval_log="$4"
  local n_test_rows="$5"

  local models_root="${ROOT_DIR}/external_repos/sally/new_models/eslasso"
  if [[ ! -d "${models_root}" ]]; then
    echo "[run] SKIP lasso (missing models_root): ${models_root}" | tee -a "${LOG_DIR}/missing_inputs.log"
    return 0
  fi

  mkdir -p "${out_dir}"

  local eval_max_rows="${n_test_rows}"
  if [[ "${MAX_TEST_ROWS}" -gt 0 && "${MAX_TEST_ROWS}" -lt "${n_test_rows}" ]]; then
    eval_max_rows="${MAX_TEST_ROWS}"
  fi

  local test_csv="repo_export/lib${lib}/realtest/merged_training_realtest_lib${lib}_chemclass_rt_prod.csv"
  local json_path="${out_dir}/results/rt_eval_lasso_realtest.json"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${json_path}" ]]; then
    echo "[run] SKIP (already evaluated): ${json_path}" | tee -a "${eval_log}"
    append_lasso_summary "${lib}" "${cap}" "${json_path}"
    return 0
  fi

  run_cmd "${eval_log}" poetry run python -u scripts/pipelines/eval_rt_lasso_baseline_by_species_cluster.py \
    --output-dir "${out_dir}" \
    --lib-id "${lib}" \
    --test-csv "${test_csv}" \
    --chunk-size "${CHUNK_SIZE}" \
    --max-test-rows "${eval_max_rows}" \
    --label realtest

  if [[ -f "${json_path}" ]]; then
    append_lasso_summary "${lib}" "${cap}" "${json_path}"
  fi
}

case "${METHOD}" in
  advi|map|nuts)
    ;;
  *)
    echo "[run] ERROR: invalid --method '${METHOD}' (expected: advi, map, nuts)" >&2
    exit 2
    ;;
esac

METHOD_ARGS_COLLAPSED=(--method "${METHOD}")
METHOD_ARGS_EXPLICIT=(--method "${METHOD}")
if [[ "${METHOD}" == "advi" ]]; then
  steps_collapsed="${ADVI_STEPS_COLLAPSED_DEFAULT}"
  steps_explicit="${ADVI_STEPS_EXPLICIT_DEFAULT}"
  if [[ "${ADVI_STEPS_SET}" == "1" ]]; then
    steps_collapsed="${ADVI_STEPS}"
    steps_explicit="${ADVI_STEPS}"
  fi
  METHOD_ARGS_COLLAPSED+=(--advi-steps "${steps_collapsed}" --advi-log-every "${ADVI_LOG_EVERY}" --advi-draws "${ADVI_DRAWS}")
  METHOD_ARGS_EXPLICIT+=(--advi-steps "${steps_explicit}" --advi-log-every "${ADVI_LOG_EVERY}" --advi-draws "${ADVI_DRAWS}")
elif [[ "${METHOD}" == "map" ]]; then
  METHOD_ARGS_COLLAPSED+=(--map-maxeval "${MAP_MAXEVAL}")
  METHOD_ARGS_EXPLICIT+=(--map-maxeval "${MAP_MAXEVAL}")
fi

IFS=',' read -r -a libs <<< "${LIBS_CSV}"
if [[ ${#libs[@]} -eq 0 ]]; then
  echo "[run] ERROR: --libs is empty" >&2
  exit 2
fi

anchor_expansions=()
read -r -a anchor_expansions <<< "${ANCHOR_EXPANSIONS_CSV//,/ }"

if [[ ${#anchor_expansions[@]} -eq 0 ]]; then
  echo "[run] ERROR: empty anchor_expansions (--anchor-expansions '${ANCHOR_EXPANSIONS_CSV}')" >&2
  exit 2
fi
for ae in "${anchor_expansions[@]}"; do
  if [[ "${ae}" != "none" && "${ae}" != "poly2" ]]; then
    echo "[run] ERROR: invalid anchor expansion: ${ae} (expected: none, poly2)" >&2
    exit 2
  fi
done

baseline_extra_anchor_expansions=()
if [[ "${ANCHOR_EXPANSIONS_SET}" == "0" ]]; then
  # Default behavior: only baseline runs poly2 (to answer the poly2 question without
  # ballooning the full cartesian product of model × expansion).
  baseline_extra_anchor_expansions=("poly2")
fi

models=()
read -r -a model_tokens <<< "${MODELS_CSV//,/ }"
if [[ ${#model_tokens[@]} -eq 0 ]]; then
  echo "[run] ERROR: empty models (--models '${MODELS_CSV}')" >&2
  exit 2
fi
for token in "${model_tokens[@]}"; do
  case "${token}" in
    baseline|subgroup|pooled_intercepts|pooled_slopes|pooled|comp_hier|chem_hier|lasso)
      ;;
    *)
      echo "[run] ERROR: invalid model '${token}' (expected: baseline, subgroup, pooled_intercepts, pooled_slopes, pooled, comp_hier, chem_hier, lasso)" >&2
      exit 2
      ;;
  esac
  seen="0"
  for existing in "${models[@]:-}"; do
    if [[ "${existing}" == "${token}" ]]; then
      seen="1"
      break
    fi
  done
  if [[ "${seen}" == "0" ]]; then
    models+=("${token}")
  fi
done

RUN_LASSO="0"
models_pymc=()
for m in "${models[@]:-}"; do
  if [[ "${m}" == "lasso" ]]; then
    RUN_LASSO="1"
  else
    models_pymc+=("${m}")
  fi
done

if [[ "${REBUILD_SUMMARY}" == "1" ]]; then
  while IFS= read -r path; do
    # Example path: .../lib208/cap100/features_none/<model>/results/rt_eval_coeff_summaries_by_support_realtest.json
    rel="${path#${RUN_DIR}/}"
    lib="$(echo "${rel}" | cut -d/ -f1 | sed 's/^lib//')"
    cap="$(echo "${rel}" | cut -d/ -f2)"
    anchor_dir="$(echo "${rel}" | cut -d/ -f3)"
    anchor_expansion="${anchor_dir#features_}"
    model="$(echo "${rel}" | cut -d/ -f4)"
    append_summary "${lib}" "${cap}" "${anchor_expansion}" "${model}" "${path}"
  done < <(find "${RUN_DIR}" -type f -path "*/results/rt_eval_coeff_summaries_by_support_realtest.json" | sort)

  if [[ "${RUN_LASSO}" == "1" ]]; then
    write_lasso_summary_header
    while IFS= read -r path; do
      rel="${path#${RUN_DIR}/}"
      lib="$(echo "${rel}" | cut -d/ -f1 | sed 's/^lib//')"
      cap="$(echo "${rel}" | cut -d/ -f2)"
      append_lasso_summary "${lib}" "${cap}" "${path}"
    done < <(find "${RUN_DIR}" -type f -path "*/results/rt_eval_lasso_realtest.json" | sort)
  fi
fi

n_models_pymc=${#models_pymc[@]}
total_jobs=0
for lib in "${libs[@]}"; do
  if [[ "${RUN_LASSO}" == "1" ]]; then
    total_jobs=$((total_jobs + 1))
  fi
  # baseline extra poly2 (default only)
  if model_enabled "baseline" "${models_pymc[@]:-}" && [[ "${ANCHOR_EXPANSIONS_SET}" == "0" ]]; then
    total_jobs=$((total_jobs + 1))
  fi
  total_jobs=$((total_jobs + ${#anchor_expansions[@]} * n_models_pymc))
done
job_i=0

pids=()
labels=()
jobs_done=0
jobs_failed=0
STATUS_EVERY_SEC=60
_last_status_ts=0

print_status() {
  local now running remaining
  now="$(date +%s)"
  if [[ "${_last_status_ts}" -eq 0 || $((now - _last_status_ts)) -ge "${STATUS_EVERY_SEC}" ]]; then
    running="${#pids[@]}"
    remaining=$(( total_jobs - jobs_done - jobs_failed ))
    echo "[run] PROGRESS: done=${jobs_done} failed=${jobs_failed} running=${running} remaining=${remaining} total=${total_jobs}"
    _last_status_ts="${now}"
  fi
}

reap_finished_jobs() {
  local i pid label status
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    label="${labels[$i]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      status=0
      wait "${pid}" || status=$?
      if [[ "${status}" -eq 0 ]]; then
        jobs_done=$((jobs_done + 1))
        echo "[run] DONE (${jobs_done}/${total_jobs}): ${label}"
      else
        jobs_failed=$((jobs_failed + 1))
        echo "[run] FAILED (${status}) (${jobs_done}+${jobs_failed}/${total_jobs}): ${label}" | tee -a "${LOG_DIR}/failures.log"
      fi
      print_status
      unset 'pids[$i]'
      unset 'labels[$i]'
    fi
  done
  # Compact arrays
  pids=("${pids[@]:-}")
  labels=("${labels[@]:-}")
}

wait_for_slot() {
  if [[ "${MAX_PARALLEL}" -le 1 ]]; then
    return 0
  fi
  while [[ ${#pids[@]} -ge "${MAX_PARALLEL}" ]]; do
    reap_finished_jobs
    if [[ ${#pids[@]} -ge "${MAX_PARALLEL}" ]]; then
      print_status
      sleep 1
    fi
  done
}

wait_all_jobs() {
  if [[ "${MAX_PARALLEL}" -le 1 ]]; then
    return 0
  fi
  while [[ ${#pids[@]} -gt 0 ]]; do
    reap_finished_jobs
    if [[ ${#pids[@]} -gt 0 ]]; then
      print_status
      sleep 1
    fi
  done
}

for lib in "${libs[@]}"; do
  train_csv="repo_export/lib${lib}/${CAP}/merged_training_all_lib${lib}_${CAP}_chemclass_rt_prod.csv"
  test_csv="repo_export/lib${lib}/realtest/merged_training_realtest_lib${lib}_chemclass_rt_prod.csv"
  if [[ ! -f "${train_csv}" ]]; then
    echo "[run] Missing train_csv: ${train_csv}" | tee -a "${LOG_DIR}/missing_inputs.log"
    continue
  fi
  if [[ ! -f "${test_csv}" ]]; then
    echo "[run] Missing test_csv: ${test_csv}" | tee -a "${LOG_DIR}/missing_inputs.log"
    continue
  fi

  echo ""
  echo "[run] ===== lib${lib} ${CAP} ====="
  n_train_rows="$(count_csv_rows "${train_csv}")"
  n_test_rows="$(count_csv_rows "${test_csv}")"
  echo "[run] train_rows=${n_train_rows} test_rows=${n_test_rows}"

  if [[ "${RUN_LASSO}" == "1" ]]; then
    model="lasso_eslasso_species_cluster"
    out_dir="${RUN_DIR}/lib${lib}/${CAP}/${model}"
    eval_log="${LOG_DIR}/lib${lib}_${CAP}_${model}.eval.log"
    job_i=$((job_i + 1))
    echo ""
    echo "[run] JOB ${job_i}/${total_jobs}: lib${lib} ${CAP} model=${model}"
    job_label="lib${lib} ${CAP} model=${model}"
    if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
      wait_for_slot
      echo "[run] START: ${job_label}"
      (
        eval_lasso "${lib}" "${CAP}" "${out_dir}" "${eval_log}" "${n_test_rows}"
      ) &
      pids+=("$!")
      labels+=("${job_label}")
    else
      (
        eval_lasso "${lib}" "${CAP}" "${out_dir}" "${eval_log}" "${n_test_rows}"
      ) || echo "[run] FAILED ${job_label}" | tee -a "${LOG_DIR}/failures.log"
    fi
  fi

  for anchor_expansion in "${anchor_expansions[@]}"; do
    FEATURE_ARGS=()
    if [[ "${INCLUDE_ES_ALL}" == "1" ]]; then
      FEATURE_ARGS+=(--include-es-all)
    fi
    if [[ "${anchor_expansion}" == "poly2" ]]; then
      FEATURE_ARGS+=(--feature-center none --feature-rotation none --anchor-expansion poly2)
    else
      FEATURE_ARGS+=(--feature-center global --feature-rotation none --anchor-expansion none)
    fi

    RIDGE_ARGS=(--lambda-mode "${LAMBDA_MODE}" --lambda-slopes "${LAMBDA_SLOPES}")
    if [[ "${anchor_expansion}" == "poly2" ]]; then
      RIDGE_ARGS+=(--lambda-slopes-poly "${LAMBDA_SLOPES_POLY}")
    fi

    feature_dir="${RUN_DIR}/lib${lib}/${CAP}/features_${anchor_expansion}"

    # 1) Baseline: supercategory grouping (status quo)
    if model_enabled "baseline" "${models_pymc[@]:-}"; then
      model="pymc_collapsed_group_species_cluster"
      out_dir="${feature_dir}/${model}"
      train_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.train.log"
      eval_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.eval.log"
      job_i=$((job_i + 1))
      echo ""
      echo "[run] JOB ${job_i}/${total_jobs}: lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      job_label="lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
        wait_for_slot
        echo "[run] START: ${job_label}"
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col species_cluster \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode collapsed \
            "${METHOD_ARGS_COLLAPSED[@]}"
        ) &
        pids+=("$!")
        labels+=("${job_label}")
      else
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col species_cluster \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode collapsed \
            "${METHOD_ARGS_COLLAPSED[@]}"
        ) || echo "[run] FAILED ${job_label}" | tee -a "${LOG_DIR}/failures.log"
      fi
    fi

    # 2) Subgroup grouping: curate-like subgroup id
    if model_enabled "subgroup" "${models_pymc[@]:-}"; then
      model="pymc_collapsed_group_species"
      out_dir="${feature_dir}/${model}"
      train_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.train.log"
      eval_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.eval.log"
      job_i=$((job_i + 1))
      echo ""
      echo "[run] JOB ${job_i}/${total_jobs}: lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      job_label="lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
        wait_for_slot
        echo "[run] START: ${job_label}"
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col "species" \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode collapsed \
            "${METHOD_ARGS_COLLAPSED[@]}"
        ) &
        pids+=("$!")
        labels+=("${job_label}")
      else
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col "species" \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode collapsed \
            "${METHOD_ARGS_COLLAPSED[@]}"
        ) || echo "[run] FAILED ${job_label}" | tee -a "${LOG_DIR}/failures.log"
      fi
    fi

    # 3) Partial pooling (intercepts-only): supercategory-aware compound hierarchy.
    if model_enabled "pooled_intercepts" "${models_pymc[@]:-}"; then
      model="pymc_pooled_species_comp_hier_supercat"
      out_dir="${feature_dir}/${model}"
      train_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.train.log"
      eval_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.eval.log"
      job_i=$((job_i + 1))
      echo ""
      echo "[run] JOB ${job_i}/${total_jobs}: lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      job_label="lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
        wait_for_slot
        echo "[run] START: ${job_label}"
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col "species" \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode explicit \
            --intercept-prior comp_hier_supercat \
            --tau-mu-prior 0.5 --tau-comp-prior 0.5 --tau-b-prior 0.5 \
            --t0-prior-sigma 10.0 \
            "${METHOD_ARGS_EXPLICIT[@]}"
        ) &
        pids+=("$!")
        labels+=("${job_label}")
      else
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col "species" \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode explicit \
            --intercept-prior comp_hier_supercat \
            --tau-mu-prior 0.5 --tau-comp-prior 0.5 --tau-b-prior 0.5 \
            --t0-prior-sigma 10.0 \
            "${METHOD_ARGS_EXPLICIT[@]}"
        ) || echo "[run] FAILED ${job_label}" | tee -a "${LOG_DIR}/failures.log"
      fi
    fi

    # 4) Partial pooling (slopes-only): supercategory-aware slope head; flat explicit intercepts.
    if model_enabled "pooled_slopes" "${models_pymc[@]:-}"; then
      model="pymc_pooled_species_flat_cluster_supercat"
      out_dir="${feature_dir}/${model}"
      train_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.train.log"
      eval_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.eval.log"
      job_i=$((job_i + 1))
      echo ""
      echo "[run] JOB ${job_i}/${total_jobs}: lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      job_label="lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
        wait_for_slot
        echo "[run] START: ${job_label}"
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col "species" \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode explicit \
            --intercept-prior flat \
            --slope-head-mode cluster_supercat \
            --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
            --t0-prior-sigma 10.0 \
            "${METHOD_ARGS_EXPLICIT[@]}"
        ) &
        pids+=("$!")
        labels+=("${job_label}")
      else
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col "species" \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode explicit \
            --intercept-prior flat \
            --slope-head-mode cluster_supercat \
            --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
            --t0-prior-sigma 10.0 \
            "${METHOD_ARGS_EXPLICIT[@]}"
        ) || echo "[run] FAILED ${job_label}" | tee -a "${LOG_DIR}/failures.log"
      fi
    fi

    # 5) Partial pooling (intercepts + slopes): supercategory-aware compound hierarchy + slope head.
    if model_enabled "pooled" "${models_pymc[@]:-}"; then
      model="pymc_pooled_species_comp_hier_supercat_cluster_supercat"
      out_dir="${feature_dir}/${model}"
      train_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.train.log"
      eval_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.eval.log"
      job_i=$((job_i + 1))
      echo ""
      echo "[run] JOB ${job_i}/${total_jobs}: lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      job_label="lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
        wait_for_slot
        echo "[run] START: ${job_label}"
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col "species" \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode explicit \
            --intercept-prior comp_hier_supercat \
            --slope-head-mode cluster_supercat \
            --tau-mu-prior 0.5 --tau-comp-prior 0.5 --tau-b-prior 0.5 \
            --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
            --t0-prior-sigma 10.0 \
            "${METHOD_ARGS_EXPLICIT[@]}"
        ) &
        pids+=("$!")
        labels+=("${job_label}")
      else
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col "species" \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode explicit \
            --intercept-prior comp_hier_supercat \
            --slope-head-mode cluster_supercat \
            --tau-mu-prior 0.5 --tau-comp-prior 0.5 --tau-b-prior 0.5 \
            --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
            --t0-prior-sigma 10.0 \
            "${METHOD_ARGS_EXPLICIT[@]}"
        ) || echo "[run] FAILED ${job_label}" | tee -a "${LOG_DIR}/failures.log"
      fi
    fi

    # 6) Compound hierarchy intercepts (no chemistry embeddings), plus supercategory slope head.
    if model_enabled "comp_hier" "${models_pymc[@]:-}"; then
      model="pymc_pooled_species_comp_hier_cluster_supercat"
      out_dir="${feature_dir}/${model}"
      train_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.train.log"
      eval_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.eval.log"
      job_i=$((job_i + 1))
      echo ""
      echo "[run] JOB ${job_i}/${total_jobs}: lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      job_label="lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
      if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
        wait_for_slot
        echo "[run] START: ${job_label}"
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col "species" \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode explicit \
            --intercept-prior comp_hier \
            --slope-head-mode cluster_supercat \
            --tau-mu-prior 0.5 --tau-comp-prior 0.5 --tau-b-prior 0.5 \
            --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
            --t0-prior-sigma 10.0 \
            "${METHOD_ARGS_EXPLICIT[@]}"
        ) &
        pids+=("$!")
        labels+=("${job_label}")
      else
        (
          train_and_eval \
            "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
            "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
            "${train_log}" "${eval_log}" "${n_test_rows}" \
            --group-col "species" \
            "${FEATURE_ARGS[@]}" \
            "${RIDGE_ARGS[@]}" \
            --intercept-mode explicit \
            --intercept-prior comp_hier \
            --slope-head-mode cluster_supercat \
            --tau-mu-prior 0.5 --tau-comp-prior 0.5 --tau-b-prior 0.5 \
            --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
            --t0-prior-sigma 10.0 \
            "${METHOD_ARGS_EXPLICIT[@]}"
        ) || echo "[run] FAILED ${job_label}" | tee -a "${LOG_DIR}/failures.log"
      fi
    fi

    # 7) Chemistry hierarchy intercepts (ChemBERTa PCA-20), plus supercategory slope head.
    if model_enabled "chem_hier" "${models_pymc[@]:-}"; then
      if [[ ! -f "${CHEM_EMBEDDINGS_PATH}" ]]; then
        echo "[run] SKIP chem_hier (missing embeddings): ${CHEM_EMBEDDINGS_PATH}" | tee -a "${LOG_DIR}/missing_inputs.log"
      else
        model="pymc_pooled_species_chem_hier_cluster_supercat"
        out_dir="${feature_dir}/${model}"
        train_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.train.log"
        eval_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.eval.log"
        job_i=$((job_i + 1))
        echo ""
        echo "[run] JOB ${job_i}/${total_jobs}: lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
        job_label="lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
        if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
          wait_for_slot
          echo "[run] START: ${job_label}"
          (
            train_and_eval \
              "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
              "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
              "${train_log}" "${eval_log}" "${n_test_rows}" \
              --group-col "species" \
              "${FEATURE_ARGS[@]}" \
              "${RIDGE_ARGS[@]}" \
              --intercept-mode explicit \
              --intercept-prior chem_hier \
              --chem-embeddings-path "${CHEM_EMBEDDINGS_PATH}" \
              --tau-mu-prior 0.5 --tau-t-prior 0.5 --tau-b-prior 0.5 \
              --theta-prior-sigma 1.0 --t0-prior-sigma 10.0 \
              --slope-head-mode cluster_supercat \
              --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
              "${METHOD_ARGS_EXPLICIT[@]}"
          ) &
          pids+=("$!")
          labels+=("${job_label}")
        else
          (
            train_and_eval \
              "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
              "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
              "${train_log}" "${eval_log}" "${n_test_rows}" \
              --group-col "species" \
              "${FEATURE_ARGS[@]}" \
              "${RIDGE_ARGS[@]}" \
              --intercept-mode explicit \
              --intercept-prior chem_hier \
              --chem-embeddings-path "${CHEM_EMBEDDINGS_PATH}" \
              --tau-mu-prior 0.5 --tau-t-prior 0.5 --tau-b-prior 0.5 \
              --theta-prior-sigma 1.0 --t0-prior-sigma 10.0 \
              --slope-head-mode cluster_supercat \
              --tau-w-prior 0.5 --w0-prior-sigma 1.0 \
              "${METHOD_ARGS_EXPLICIT[@]}"
          ) || echo "[run] FAILED ${job_label}" | tee -a "${LOG_DIR}/failures.log"
        fi
      fi
    fi
  done

  # Default-only: run baseline poly2 once, without expanding the full cartesian product.
  if [[ "${ANCHOR_EXPANSIONS_SET}" == "0" ]] && model_enabled "baseline" "${models_pymc[@]:-}"; then
    anchor_expansion="poly2"
    FEATURE_ARGS=()
    if [[ "${INCLUDE_ES_ALL}" == "1" ]]; then
      FEATURE_ARGS+=(--include-es-all)
    fi
    FEATURE_ARGS+=(--feature-center none --feature-rotation none --anchor-expansion poly2)

    RIDGE_ARGS=(--lambda-mode "${LAMBDA_MODE}" --lambda-slopes "${LAMBDA_SLOPES}" --lambda-slopes-poly "${LAMBDA_SLOPES_POLY}")

    feature_dir="${RUN_DIR}/lib${lib}/${CAP}/features_${anchor_expansion}"
    model="pymc_collapsed_group_species_cluster"
    out_dir="${feature_dir}/${model}"
    train_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.train.log"
    eval_log="${LOG_DIR}/lib${lib}_${CAP}_${anchor_expansion}_${model}.eval.log"
    job_i=$((job_i + 1))
    echo ""
    echo "[run] JOB ${job_i}/${total_jobs}: lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
    job_label="lib${lib} ${CAP} anchor=${anchor_expansion} model=${model}"
    if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
      wait_for_slot
      echo "[run] START: ${job_label}"
      (
        train_and_eval \
          "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
          "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
          "${train_log}" "${eval_log}" "${n_test_rows}" \
          --group-col species_cluster \
          "${FEATURE_ARGS[@]}" \
          "${RIDGE_ARGS[@]}" \
          --intercept-mode collapsed \
          "${METHOD_ARGS_COLLAPSED[@]}"
      ) &
      pids+=("$!")
      labels+=("${job_label}")
    else
      (
        train_and_eval \
          "${lib}" "${CAP}" "${anchor_expansion}" "${model}" "${out_dir}" \
          "${out_dir}/models/stage1_coeff_summaries_posterior.npz" \
          "${train_log}" "${eval_log}" "${n_test_rows}" \
          --group-col species_cluster \
          "${FEATURE_ARGS[@]}" \
          "${RIDGE_ARGS[@]}" \
          --intercept-mode collapsed \
          "${METHOD_ARGS_COLLAPSED[@]}"
      ) || echo "[run] FAILED ${job_label}" | tee -a "${LOG_DIR}/failures.log"
    fi
  fi
done

wait_all_jobs

if [[ "${DEFER_SUMMARY}" == "1" ]]; then
  echo "[run] Rebuilding summary.csv (parallel mode; avoid concurrent writes)"
  FORCE_SUMMARY="1"
  write_summary_header
  while IFS= read -r path; do
    rel="${path#${RUN_DIR}/}"
    lib="$(echo "${rel}" | cut -d/ -f1 | sed 's/^lib//')"
    cap="$(echo "${rel}" | cut -d/ -f2)"
    anchor_dir="$(echo "${rel}" | cut -d/ -f3)"
    anchor_expansion="${anchor_dir#features_}"
    model="$(echo "${rel}" | cut -d/ -f4)"
    append_summary "${lib}" "${cap}" "${anchor_expansion}" "${model}" "${path}"
  done < <(find "${RUN_DIR}" -type f -path "*/results/rt_eval_coeff_summaries_by_support_realtest.json" | sort)
  unset FORCE_SUMMARY
fi

write_analysis() {
  local run_dir="$1"
  local out_dir="${run_dir}/analysis"
  mkdir -p "${out_dir}"
  poetry run python - "${run_dir}" "${out_dir}" <<'PY'
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_species_mapping(lib_id: int) -> Path | None:
    candidates = sorted(
        glob.glob(f"repo_export/lib{lib_id}/species_mapping/**/*_lib{lib_id}_species_mapping.csv")
    )
    return Path(candidates[0]) if candidates else None


def _load_species_maps(lib_id: int) -> tuple[dict[int, int], dict[int, str]]:
    mapping_csv = _find_species_mapping(lib_id)
    if mapping_csv is None or not mapping_csv.exists():
        return {}, {}
    df = pd.read_csv(mapping_csv)
    # species -> species_cluster
    spec = df[["species", "species_cluster"]].dropna().drop_duplicates().copy()
    spec["species"] = spec["species"].astype(int)
    spec["species_cluster"] = spec["species_cluster"].astype(int)
    # deterministic mapping
    if spec.groupby("species")["species_cluster"].nunique().max() != 1:
        raise SystemExit(f"Non-deterministic species->species_cluster mapping in {mapping_csv}")
    spec_to_cluster = dict(zip(spec["species"].tolist(), spec["species_cluster"].tolist(), strict=True))
    # cluster -> label
    lab = df[["species_cluster", "species_group_raw"]].dropna().drop_duplicates().copy()
    lab["species_cluster"] = lab["species_cluster"].astype(int)
    cluster_to_label = {}
    for cluster_id, sub in lab.groupby("species_cluster"):
        vals = sorted({str(x).strip() for x in sub["species_group_raw"].dropna().tolist()})
        if vals:
            cluster_to_label[int(cluster_id)] = " / ".join(vals)
    return spec_to_cluster, cluster_to_label


def _parse_run_metadata(path: Path, run_dir: Path) -> dict[str, object]:
    # Example:
    #   <run_dir>/lib208/cap100/features_none/<model>/results/rt_eval_coeff_summaries_by_group_realtest.csv
    rel = path.relative_to(run_dir)
    lib = int(str(rel.parts[0]).replace("lib", ""))
    cap = str(rel.parts[1])
    anchor_dir = str(rel.parts[2])
    anchor_expansion = anchor_dir.replace("features_", "")
    model = str(rel.parts[3])
    return {"lib": lib, "cap": cap, "anchor_expansion": anchor_expansion, "model": model}


def _weighted_row_agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    n = df["n_obs_test"].astype(float)
    sse = (df["rmse"].astype(float) ** 2) * n
    sae = df["mae"].astype(float) * n
    scov = df["coverage_95"].astype(float) * n
    sw = df["interval_width_mean"].astype(float) * n
    out = (
        df.assign(_n=n, _sse=sse, _sae=sae, _scov=scov, _sw=sw)
        .groupby(by, as_index=False)
        .agg(
            n_obs_test=("_n", "sum"),
            sse=("_sse", "sum"),
            sae=("_sae", "sum"),
            scov=("_scov", "sum"),
            sw=("_sw", "sum"),
            n_groups=("group_key", "count"),
        )
    )
    out["rmse"] = np.sqrt(out["sse"] / out["n_obs_test"])
    out["mae"] = out["sae"] / out["n_obs_test"]
    out["coverage_95"] = out["scov"] / out["n_obs_test"]
    out["interval_width_mean"] = out["sw"] / out["n_obs_test"]
    return out.drop(columns=["sse", "sae", "scov", "sw"])


def _quantiles(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    return {
        "p50": float(np.quantile(v, 0.50)),
        "p90": float(np.quantile(v, 0.90)),
        "p99": float(np.quantile(v, 0.99)),
    }


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    group_paths = sorted(
        run_dir.glob("lib*/**/results/rt_eval_coeff_summaries_by_group_realtest.csv")
    )
    support_paths = sorted(
        run_dir.glob("lib*/**/results/rt_eval_coeff_summaries_by_support_realtest.csv")
    )
    lasso_paths = sorted(run_dir.glob("lib*/**/results/rt_eval_lasso_by_species_cluster_realtest.csv"))
    lasso_json_paths = sorted(run_dir.glob("lib*/**/results/rt_eval_lasso_realtest.json"))

    cluster_rows: list[dict[str, object]] = []
    cluster_comp_rows: list[dict[str, object]] = []

    for path in group_paths:
        meta = _parse_run_metadata(path, run_dir)
        lib = int(meta["lib"])
        spec_to_cluster, cluster_to_label = _load_species_maps(lib)

        df = pd.read_csv(path)
        # Determine which column holds the group id.
        if "supercat_id" in df.columns:
            df = df.rename(columns={"supercat_id": "species_cluster"})
        if "species_cluster" not in df.columns and "species" in df.columns:
            # Species-level model; map species -> species_cluster using repo_export mapping.
            if not spec_to_cluster:
                raise SystemExit(f"Missing species mapping for lib{lib}; cannot map species -> species_cluster")
            df["species"] = df["species"].astype(int)
            df["species_cluster"] = df["species"].map(spec_to_cluster)
        if "species_cluster" not in df.columns:
            raise SystemExit(f"Could not find species_cluster for {path}")

        df["species_cluster"] = df["species_cluster"].astype(int)
        df = df[df["n_obs_test"].astype(int) > 0].copy()
        if df.empty:
            continue

        # (A) Row-level metrics by species_cluster.
        agg_cluster = _weighted_row_agg(df, by=["species_cluster"])
        for _, row in agg_cluster.iterrows():
            sc = int(row["species_cluster"])
            cluster_rows.append(
                {
                    **meta,
                    "species_cluster": sc,
                    "species_group_raw": cluster_to_label.get(sc, str(sc)),
                    "n_obs_test": int(row["n_obs_test"]),
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "coverage_95": float(row["coverage_95"]),
                    "interval_width_mean": float(row["interval_width_mean"]),
                    "n_groups": int(row["n_groups"]),
                }
            )

        # (B) Comparable within-cluster variance by collapsing to (species_cluster, comp_id).
        comp_agg = _weighted_row_agg(df, by=["species_cluster", "comp_id"])
        for (sc, sub) in comp_agg.groupby("species_cluster"):
            qs = _quantiles(sub["rmse"].to_numpy(dtype=float, copy=False))
            cluster_comp_rows.append(
                {
                    **meta,
                    "species_cluster": int(sc),
                    "species_group_raw": cluster_to_label.get(int(sc), str(int(sc))),
                    "n_comp_groups": int(len(sub)),
                    "rmse_comp_p50": float(qs["p50"]),
                    "rmse_comp_p90": float(qs["p90"]),
                    "rmse_comp_p99": float(qs["p99"]),
                }
            )

    if cluster_rows:
        pd.DataFrame(cluster_rows).sort_values(
            ["lib", "cap", "anchor_expansion", "model", "species_cluster"]
        ).to_csv(out_dir / "cluster_row_metrics.csv", index=False)

    if cluster_comp_rows:
        pd.DataFrame(cluster_comp_rows).sort_values(
            ["lib", "cap", "anchor_expansion", "model", "species_cluster"]
        ).to_csv(out_dir / "cluster_comp_rmse_quantiles.csv", index=False)

    # Support-bin comparison for species-level models (subgroup vs pooled).
    support_out = []
    for path in support_paths:
        meta = _parse_run_metadata(path, run_dir)
        model = str(meta["model"])
        if "pymc_collapsed_group_species" not in model and "pymc_pooled_species" not in model:
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            support_out.append(
                {
                    **meta,
                    "support_bin": str(row.get("support_bin")),
                    "n_groups": int(row.get("n_groups", 0)) if pd.notna(row.get("n_groups", np.nan)) else 0,
                    "n_groups_with_test": int(row.get("n_groups_with_test", 0))
                    if pd.notna(row.get("n_groups_with_test", np.nan))
                    else 0,
                    "n_obs_test": int(row.get("n_obs_test", 0)) if pd.notna(row.get("n_obs_test", np.nan)) else 0,
                    "rmse": float(row.get("rmse", np.nan)),
                    "mae": float(row.get("mae", np.nan)),
                    "coverage_95": float(row.get("coverage_95", np.nan)),
                    "rmse_p90": float(row.get("rmse_p90", np.nan)),
                    "rmse_p99": float(row.get("rmse_p99", np.nan)),
                }
            )
    if support_out:
        pd.DataFrame(support_out).sort_values(
            ["lib", "cap", "anchor_expansion", "model", "support_bin"]
        ).to_csv(out_dir / "support_metrics_species_models.csv", index=False)

    # Lasso baseline metrics (eslasso) for reference.
    lasso_cluster_rows: list[dict[str, object]] = []
    for path in lasso_paths:
        rel = path.relative_to(run_dir)
        lib_part = next((p for p in rel.parts if str(p).startswith("lib")), None)
        lib = int(str(lib_part).replace("lib", "")) if lib_part else -1
        cap = next((p for p in rel.parts if str(p).startswith("cap")), "cap100")
        _, cluster_to_label = _load_species_maps(lib) if lib > 0 else ({}, {})

        df = pd.read_csv(path)
        if "species_cluster" not in df.columns:
            continue
        df["species_cluster"] = df["species_cluster"].astype(int)
        for _, row in df.iterrows():
            sc = int(row["species_cluster"])
            lasso_cluster_rows.append(
                {
                    "lib": int(lib),
                    "cap": str(cap),
                    "anchor_expansion": "none",
                    "model": "lasso_eslasso_species_cluster",
                    "species_cluster": sc,
                    "species_group_raw": cluster_to_label.get(sc, str(sc)),
                    "n_obs_test": int(row.get("n_obs", 0)) if pd.notna(row.get("n_obs", np.nan)) else 0,
                    "rmse": float(row.get("rmse", np.nan)),
                    "mae": float(row.get("mae", np.nan)),
                    "coverage_95": float(row.get("coverage_95", np.nan)),
                    "interval_width_mean": float("nan"),
                    "n_groups": 0,
                }
            )
    if lasso_cluster_rows:
        pd.DataFrame(lasso_cluster_rows).sort_values(["lib", "cap", "species_cluster"]).to_csv(
            out_dir / "lasso_cluster_metrics.csv", index=False
        )

    lasso_global_rows: list[dict[str, object]] = []
    for path in lasso_json_paths:
        rel = path.relative_to(run_dir)
        lib_part = next((p for p in rel.parts if str(p).startswith("lib")), None)
        lib = int(str(lib_part).replace("lib", "")) if lib_part else -1
        cap = next((p for p in rel.parts if str(p).startswith("cap")), "cap100")
        payload = json.loads(Path(path).read_text())
        m = payload.get("metrics", {})
        lasso_global_rows.append(
            {
                "lib": int(lib),
                "cap": str(cap),
                "model": "lasso_eslasso_species_cluster",
                "rmse": float(m.get("rmse", float("nan"))),
                "mae": float(m.get("mae", float("nan"))),
                "coverage_95": float(m.get("coverage_95", float("nan"))),
                "n_test_rows_seen": int(payload.get("n_test_rows_seen", 0)),
                "n_rows_evaluated": int(payload.get("n_rows_evaluated", 0)),
                "skipped_no_super": int(payload.get("skipped_no_super", 0)),
                "skipped_no_model": int(payload.get("skipped_no_model", 0)),
                "json_path": str(path),
            }
        )
    if lasso_global_rows:
        pd.DataFrame(lasso_global_rows).sort_values(["lib", "cap"]).to_csv(
            out_dir / "lasso_global_metrics.csv", index=False
        )


if __name__ == "__main__":
    main()
PY
  echo "[run] Wrote analysis to ${out_dir}"
}

write_analysis "${RUN_DIR}"

echo "[run] DONE. Summary: ${SUMMARY_CSV}"
echo "[run] Logs: ${LOG_DIR}"
