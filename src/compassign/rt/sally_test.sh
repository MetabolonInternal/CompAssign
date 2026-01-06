#!/usr/bin/env bash
set -euo pipefail

# Reproduce preliminary end-to-end Sally evaluation runs used in the RT report.
#
# This script runs Sally in the `sally` conda environment (via `conda run`) and writes
# outputs under `external_repos/sally/out/` with fixed directory names so the LaTeX report
# can link to stable artifacts.
#
# Intended usage: run with no args (defaults encoded below).
#
# Notes:
# - Requires access to the Sally repo under external_repos/ and working Docker credentials.
# - Uses Sally's local cache for speed on repeated runs (do not set CACHE_S3_PATH).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -ne 0 ]]; then
  echo "ERROR: src/compassign/rt/sally_test.sh does not take arguments. Run it with no args." >&2
  exit 2
fi

SALLY_DIR="${REPO_ROOT}/external_repos/sally"
MODEL_DIR_REL="./new_models"

# CompAssign covariates CSV (host path) mounted into the Sally container by pipeline.sh.
LIB208_COV_CSV="${REPO_ROOT}/repo_export/lib208/realtest/merged_training_realtest_lib208_chemclass_rt_prod.csv"
LIB209_COV_CSV="${REPO_ROOT}/repo_export/lib209/realtest/merged_training_realtest_lib209_chemclass_rt_prod.csv"

if [[ ! -d "${SALLY_DIR}" ]]; then
  echo "ERROR: Missing Sally repo: ${SALLY_DIR}" >&2
  exit 2
fi
if [[ ! -d "${SALLY_DIR}/new_models" ]]; then
  echo "ERROR: Missing Sally model directory: ${SALLY_DIR}/new_models" >&2
  exit 2
fi
if [[ ! -f "${LIB208_COV_CSV}" ]]; then
  echo "ERROR: Missing lib208 covariates CSV: ${LIB208_COV_CSV}" >&2
  exit 2
fi
if [[ ! -f "${LIB209_COV_CSV}" ]]; then
  echo "ERROR: Missing lib209 covariates CSV: ${LIB209_COV_CSV}" >&2
  exit 2
fi

# Default behavior: re-run from scratch to ensure outputs match the current code.
# To abort before deletion, Ctrl+C during the countdown below.
RERUN_FROM_SCRATCH=1

safe_rm_out_dir() {
  local out_rel="$1"
  case "${out_rel}" in
    out/prod_*) ;;
    *)
      echo "ERROR: Refusing to delete unexpected output dir: ${out_rel}" >&2
      exit 2
      ;;
  esac
  rm -rf -- "${SALLY_DIR}/${out_rel}"
}

lookup_species_info_from_cov_csv() {
  local ssid="$1"
  local csv_path="$2"
  # CSV schema is production RT CSV: sampleset_id, worksheet_id, task_id, species, species_cluster, ...
  awk -F',' -v ssid="${ssid}" '$1==ssid {print $4, $5; exit}' "${csv_path}"
}

run_sally_eval() {
  local ssid="$1"
  local out_rel="$2"
  shift 2
  echo ""
  echo "[run] ssid=${ssid} out=${out_rel}"
  echo "[run] args: $*"
  (
    cd "${SALLY_DIR}"
    conda run --no-capture-output -n sally \
      ./scripts/sally-dev evaluate "${ssid}" -o "${out_rel}" -m "${MODEL_DIR_REL}" "$@"
  )
}

run_eslasso_baseline() {
  local ssid="$1"
  local out_rel="$2"
  local regression_matrix="$3"
  local libs_csv="$4"

  run_sally_eval "${ssid}" "${out_rel}" \
    --model-type eslasso \
    --regression-species-matrix "${regression_matrix}" \
    --libs "${libs_csv}" \
    --peak-assignment-method baseline \
    --evaluation-mode mdsutils \
    --force-curate
}

run_compassign_pp_ridge() {
  local ssid="$1"
  local out_rel="$2"
  local regression_matrix="$3"
  local libs_csv="$4"
  local cov_csv="$5"

  read -r species_id species_cluster_id < <(lookup_species_info_from_cov_csv "${ssid}" "${cov_csv}")
  if [[ -z "${species_id:-}" || -z "${species_cluster_id:-}" ]]; then
    echo "ERROR: Unable to infer species/species_cluster for SSID ${ssid} from ${cov_csv}" >&2
    exit 2
  fi

  run_sally_eval "${ssid}" "${out_rel}" \
    --model-type compassign_pp_ridge \
    --regression-species-matrix "${regression_matrix}" \
    --libs "${libs_csv}" \
    --peak-assignment-method baseline \
    --evaluation-mode mdsutils \
    --force-curate \
    --compassign-rt-species "${species_id}" \
    --compassign-rt-species-cluster "${species_cluster_id}" \
    --compassign-rt-covariates-csv "${cov_csv}" \
    --compassign-rt-allow-missing-task-covariates
}

echo "[sally_test] Sally repo: ${SALLY_DIR}"
echo "[sally_test] lib208 covariates CSV: ${LIB208_COV_CSV}"
echo "[sally_test] lib209 covariates CSV: ${LIB209_COV_CSV}"

# Fixed runs used in docs/models/rt_pymc_multilevel_pooling_report.tex
OUT_DIRS=(
  out/prod_12307_supercat6_eslasso_baseline
  out/prod_12307_supercat6_compassign_pp_ridge_stage1only_check
  out/prod_12609_supercat8_eslasso_baseline
  out/prod_12609_supercat8_compassign_pp_ridge
  out/prod_12725_supercat6_eslasso_baseline
  out/prod_12725_supercat6_compassign_pp_ridge
  out/prod_20159_supercat8_eslasso_baseline_lib209
  out/prod_20159_supercat8_compassign_pp_ridge_lib209
  out/prod_23059_supercat6_eslasso_baseline_lib209
  out/prod_23059_supercat6_compassign_pp_ridge_lib209
)

if [[ "${RERUN_FROM_SCRATCH}" == "1" ]]; then
  TO_DELETE=()
  for d in "${OUT_DIRS[@]}"; do
    if [[ -d "${SALLY_DIR}/${d}" ]]; then
      TO_DELETE+=( "${d}" )
    fi
  done
  if [[ "${#TO_DELETE[@]}" -gt 0 ]]; then
    echo ""
    echo "[sally_test] About to delete and recreate these output dirs:"
    for d in "${TO_DELETE[@]}"; do
      echo "  - ${SALLY_DIR}/${d}"
    done
    echo "[sally_test] Ctrl+C now to abort (countdown: 10s)"
    sleep 10
    for d in "${TO_DELETE[@]}"; do
      safe_rm_out_dir "${d}"
    done
  fi
fi

run_eslasso_baseline 12307 out/prod_12307_supercat6_eslasso_baseline \
  supercategory_6_non_human_cells_and_plants 208,209
run_compassign_pp_ridge 12307 out/prod_12307_supercat6_compassign_pp_ridge_stage1only_check \
  supercategory_6_non_human_cells_and_plants 208 "${LIB208_COV_CSV}"

run_eslasso_baseline 12609 out/prod_12609_supercat8_eslasso_baseline \
  supercategory_8_human_cells 208
run_compassign_pp_ridge 12609 out/prod_12609_supercat8_compassign_pp_ridge \
  supercategory_8_human_cells 208 "${LIB208_COV_CSV}"

run_eslasso_baseline 12725 out/prod_12725_supercat6_eslasso_baseline \
  supercategory_6_non_human_cells_and_plants 208
run_compassign_pp_ridge 12725 out/prod_12725_supercat6_compassign_pp_ridge \
  supercategory_6_non_human_cells_and_plants 208 "${LIB208_COV_CSV}"

run_eslasso_baseline 20159 out/prod_20159_supercat8_eslasso_baseline_lib209 \
  supercategory_8_human_cells 209
run_compassign_pp_ridge 20159 out/prod_20159_supercat8_compassign_pp_ridge_lib209 \
  supercategory_8_human_cells 209 "${LIB209_COV_CSV}"

run_eslasso_baseline 23059 out/prod_23059_supercat6_eslasso_baseline_lib209 \
  supercategory_6_non_human_cells_and_plants 209
run_compassign_pp_ridge 23059 out/prod_23059_supercat6_compassign_pp_ridge_lib209 \
  supercategory_6_non_human_cells_and_plants 209 "${LIB209_COV_CSV}"

echo ""
echo "[sally_test] Summary"
conda run --no-capture-output -n sally python - <<'PY'
import pickle
from pathlib import Path

ROOT = Path("external_repos/sally/out")

RUNS = [
    # supercat, ssid, model, run_dir, model_key, lib_id
    ("6", 12307, "ESLASSO", "prod_12307_supercat6_eslasso_baseline", "ESLASSO", 208),
    (
        "6",
        12307,
        "COMPASSIGN_PP_RIDGE",
        "prod_12307_supercat6_compassign_pp_ridge_stage1only_check",
        "COMPASSIGN_PP_RIDGE",
        208,
    ),
    ("8", 12609, "ESLASSO", "prod_12609_supercat8_eslasso_baseline", "ESLASSO", 208),
    (
        "8",
        12609,
        "COMPASSIGN_PP_RIDGE",
        "prod_12609_supercat8_compassign_pp_ridge",
        "COMPASSIGN_PP_RIDGE",
        208,
    ),
    ("6", 12725, "ESLASSO", "prod_12725_supercat6_eslasso_baseline", "ESLASSO", 208),
    ("6", 12725, "COMPASSIGN_PP_RIDGE", "prod_12725_supercat6_compassign_pp_ridge", "COMPASSIGN_PP_RIDGE", 208),
    ("8", 20159, "ESLASSO", "prod_20159_supercat8_eslasso_baseline_lib209", "ESLASSO", 209),
    (
        "8",
        20159,
        "COMPASSIGN_PP_RIDGE",
        "prod_20159_supercat8_compassign_pp_ridge_lib209",
        "COMPASSIGN_PP_RIDGE",
        209,
    ),
    ("6", 23059, "ESLASSO", "prod_23059_supercat6_eslasso_baseline_lib209", "ESLASSO", 209),
    (
        "6",
        23059,
        "COMPASSIGN_PP_RIDGE",
        "prod_23059_supercat6_compassign_pp_ridge_lib209",
        "COMPASSIGN_PP_RIDGE",
        209,
    ),
]


def load_summary(run_dir: str, ssid: int, model_key: str) -> dict:
    eval_dir = ROOT / run_dir / "evaluation"
    matches = sorted(eval_dir.glob(f"dataForDB_{ssid}_AC_*_{model_key}_mdsutils_summary.pkl"))
    if not matches:
        raise FileNotFoundError(f"Missing mdsutils summary for {run_dir} ssid={ssid} model={model_key}")
    with matches[0].open("rb") as f:
        return pickle.load(f)


print("lib\tsupercat\tssid\tmodel\tprecision\trecall\tts")
for supercat, ssid, model, run_dir, model_key, lib_id in RUNS:
    obj = load_summary(run_dir, ssid, model_key)
    per = obj["per_lib"]
    per_lib = per[per["lib_id"] == lib_id].copy()
    if per_lib.empty:
        raise RuntimeError(f"No lib{lib_id} row in per_lib for {run_dir} ({model_key})")
    row = per_lib.iloc[0]
    print(
        f"{lib_id}\t{supercat}\t{ssid}\t{model}\t{row['precision']:.4f}\t{row['recall']:.4f}\t{row['ts']:.4f}"
    )
PY

echo ""
echo "[sally_test] Done. Outputs under: ${SALLY_DIR}/out"
