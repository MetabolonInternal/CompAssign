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
# - Oracle-backed runs require VPN access (default provider is Oracle).
# - Uses Sally's local cache for speed on repeated runs (do not set CACHE_S3_PATH).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -ne 0 ]]; then
  echo "ERROR: src/compassign/rt/sally_test.sh does not take arguments. Run it with no args." >&2
  exit 2
fi

SALLY_DIR="${REPO_ROOT}/external_repos/sally"
MODEL_DIR_REL="./new_models"

# RT run directory containing the CompAssign coefficient-summary artifacts used for staging into Sally.
# This is produced by `./src/compassign/rt/train.sh`.
RT_RUN_DIR="${REPO_ROOT}/output/rt_ridge_partial_pooling"
RT_MODELS_REL="cap100/features_none/pymc_pooled_species_comp_hier_supercat_cluster_supercat/models"

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

stage_compassign_pp_ridge_models() {
  if [[ ! -d "${RT_RUN_DIR}" ]]; then
    echo "ERROR: Missing RT run dir: ${RT_RUN_DIR}" >&2
    exit 2
  fi

  for lib in 208 209; do
    local src_dir="${RT_RUN_DIR}/lib${lib}/${RT_MODELS_REL}"
    if [[ ! -d "${src_dir}" ]]; then
      echo "ERROR: Missing models dir for lib${lib}: ${src_dir}" >&2
      exit 2
    fi

    local coeff_npz="${src_dir}/stage1_coeff_summaries_posterior.npz"
    local backoff_npz="${src_dir}/partial_pool_backoff_summaries.npz"
    if [[ ! -f "${coeff_npz}" ]]; then
      echo "ERROR: Missing ${coeff_npz}" >&2
      exit 2
    fi
    if [[ ! -f "${backoff_npz}" ]]; then
      echo "ERROR: Missing ${backoff_npz}" >&2
      exit 2
    fi

    echo "[sally_test] Staging COMPASSIGN_PP_RIDGE artifacts for lib${lib} from: ${src_dir}"
    local dest_dir="${SALLY_DIR}/new_models/compassign_pp_ridge/lib${lib}"
    mkdir -p "${dest_dir}"
    cp -f "${coeff_npz}" "${dest_dir}/stage1_coeff_summaries_posterior.npz"
    cp -f "${backoff_npz}" "${dest_dir}/partial_pool_backoff_summaries.npz"
  done
}

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
  rm -rf -- "${SALLY_DIR:?}/${out_rel:?}"
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

run_eslasso() {
  local ssid="$1"
  local out_rel="$2"
  local regression_matrix="$3"
  local libs_csv="$4"
  local peak_method="$5"

  run_sally_eval "${ssid}" "${out_rel}" \
    --model-type eslasso \
    --regression-species-matrix "${regression_matrix}" \
    --libs "${libs_csv}" \
    --peak-assignment-method "${peak_method}" \
    --evaluation-mode mdsutils \
    --force-curate
}

run_compassign_pp_ridge() {
  local ssid="$1"
  local out_rel="$2"
  local regression_matrix="$3"
  local libs_csv="$4"
  local cov_csv="$5"
  local peak_method="${6:-mixture_model}"

  read -r species_id species_cluster_id < <(lookup_species_info_from_cov_csv "${ssid}" "${cov_csv}")
  if [[ -z "${species_id:-}" || -z "${species_cluster_id:-}" ]]; then
    echo "ERROR: Unable to infer species/species_cluster for SSID ${ssid} from ${cov_csv}" >&2
    exit 2
  fi
  run_sally_eval "${ssid}" "${out_rel}" \
    --model-type compassign_pp_ridge \
    --regression-species-matrix "${regression_matrix}" \
    --libs "${libs_csv}" \
    --peak-assignment-method "${peak_method}" \
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
stage_compassign_pp_ridge_models

# Fixed runs for end-to-end Sally evaluation.
# The initial subset is referenced by docs/models/rt_pymc_multilevel_pooling_report.tex; additional SSIDs
# are included so we have 2 test-fold SSIDs per (lib, supercategory) (see data/split_outputs/train_test_split_all.csv).
OUT_DIRS=(
  out/prod_12307_supercat6_eslasso_baseline
  out/prod_12307_supercat6_eslasso_mixture_model
  out/prod_12307_supercat6_compassign_pp_ridge_baseline
  out/prod_12307_supercat6_compassign_pp_ridge_mixture_model
  out/prod_12609_supercat8_eslasso_baseline
  out/prod_12609_supercat8_eslasso_mixture_model
  out/prod_12609_supercat8_compassign_pp_ridge_baseline
  out/prod_12609_supercat8_compassign_pp_ridge_mixture_model
  out/prod_13208_supercat8_eslasso_baseline
  out/prod_13208_supercat8_eslasso_mixture_model
  out/prod_13208_supercat8_compassign_pp_ridge_baseline
  out/prod_13208_supercat8_compassign_pp_ridge_mixture_model
  out/prod_12725_supercat6_eslasso_baseline
  out/prod_12725_supercat6_eslasso_mixture_model
  out/prod_12725_supercat6_compassign_pp_ridge_baseline
  out/prod_12725_supercat6_compassign_pp_ridge_mixture_model
  out/prod_20814_supercat8_eslasso_baseline_lib209
  out/prod_20814_supercat8_eslasso_mixture_model_lib209
  out/prod_20814_supercat8_compassign_pp_ridge_baseline_lib209
  out/prod_20814_supercat8_compassign_pp_ridge_mixture_model_lib209
  out/prod_20159_supercat8_eslasso_baseline_lib209
  out/prod_20159_supercat8_eslasso_mixture_model_lib209
  out/prod_20159_supercat8_compassign_pp_ridge_baseline_lib209
  out/prod_20159_supercat8_compassign_pp_ridge_mixture_model_lib209
  out/prod_23146_supercat6_eslasso_baseline_lib209
  out/prod_23146_supercat6_eslasso_mixture_model_lib209
  out/prod_23146_supercat6_compassign_pp_ridge_baseline_lib209
  out/prod_23146_supercat6_compassign_pp_ridge_mixture_model_lib209
  out/prod_23059_supercat6_eslasso_baseline_lib209
  out/prod_23059_supercat6_eslasso_mixture_model_lib209
  out/prod_23059_supercat6_compassign_pp_ridge_baseline_lib209
  out/prod_23059_supercat6_compassign_pp_ridge_mixture_model_lib209
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

run_eslasso 12307 out/prod_12307_supercat6_eslasso_baseline \
  supercategory_6_non_human_cells_and_plants 208,209 baseline
run_eslasso 12307 out/prod_12307_supercat6_eslasso_mixture_model \
  supercategory_6_non_human_cells_and_plants 208,209 mixture_model
run_compassign_pp_ridge 12307 out/prod_12307_supercat6_compassign_pp_ridge_baseline \
  supercategory_6_non_human_cells_and_plants 208 "${LIB208_COV_CSV}" baseline
run_compassign_pp_ridge 12307 out/prod_12307_supercat6_compassign_pp_ridge_mixture_model \
  supercategory_6_non_human_cells_and_plants 208 "${LIB208_COV_CSV}" mixture_model

run_eslasso 12609 out/prod_12609_supercat8_eslasso_baseline \
  supercategory_8_human_cells 208 baseline
run_eslasso 12609 out/prod_12609_supercat8_eslasso_mixture_model \
  supercategory_8_human_cells 208 mixture_model
run_compassign_pp_ridge 12609 out/prod_12609_supercat8_compassign_pp_ridge_baseline \
  supercategory_8_human_cells 208 "${LIB208_COV_CSV}" baseline
run_compassign_pp_ridge 12609 out/prod_12609_supercat8_compassign_pp_ridge_mixture_model \
  supercategory_8_human_cells 208 "${LIB208_COV_CSV}" mixture_model

run_eslasso 13208 out/prod_13208_supercat8_eslasso_baseline \
  supercategory_8_human_cells 208 baseline
run_eslasso 13208 out/prod_13208_supercat8_eslasso_mixture_model \
  supercategory_8_human_cells 208 mixture_model
run_compassign_pp_ridge 13208 out/prod_13208_supercat8_compassign_pp_ridge_baseline \
  supercategory_8_human_cells 208 "${LIB208_COV_CSV}" baseline
run_compassign_pp_ridge 13208 out/prod_13208_supercat8_compassign_pp_ridge_mixture_model \
  supercategory_8_human_cells 208 "${LIB208_COV_CSV}" mixture_model

run_eslasso 12725 out/prod_12725_supercat6_eslasso_baseline \
  supercategory_6_non_human_cells_and_plants 208 baseline
run_eslasso 12725 out/prod_12725_supercat6_eslasso_mixture_model \
  supercategory_6_non_human_cells_and_plants 208 mixture_model
run_compassign_pp_ridge 12725 out/prod_12725_supercat6_compassign_pp_ridge_baseline \
  supercategory_6_non_human_cells_and_plants 208 "${LIB208_COV_CSV}" baseline
run_compassign_pp_ridge 12725 out/prod_12725_supercat6_compassign_pp_ridge_mixture_model \
  supercategory_6_non_human_cells_and_plants 208 "${LIB208_COV_CSV}" mixture_model

run_eslasso 20814 out/prod_20814_supercat8_eslasso_baseline_lib209 \
  supercategory_8_human_cells 209 baseline
run_eslasso 20814 out/prod_20814_supercat8_eslasso_mixture_model_lib209 \
  supercategory_8_human_cells 209 mixture_model
run_compassign_pp_ridge 20814 out/prod_20814_supercat8_compassign_pp_ridge_baseline_lib209 \
  supercategory_8_human_cells 209 "${LIB209_COV_CSV}" baseline
run_compassign_pp_ridge 20814 out/prod_20814_supercat8_compassign_pp_ridge_mixture_model_lib209 \
  supercategory_8_human_cells 209 "${LIB209_COV_CSV}" mixture_model

run_eslasso 20159 out/prod_20159_supercat8_eslasso_baseline_lib209 \
  supercategory_8_human_cells 209 baseline
run_eslasso 20159 out/prod_20159_supercat8_eslasso_mixture_model_lib209 \
  supercategory_8_human_cells 209 mixture_model
run_compassign_pp_ridge 20159 out/prod_20159_supercat8_compassign_pp_ridge_baseline_lib209 \
  supercategory_8_human_cells 209 "${LIB209_COV_CSV}" baseline
run_compassign_pp_ridge 20159 out/prod_20159_supercat8_compassign_pp_ridge_mixture_model_lib209 \
  supercategory_8_human_cells 209 "${LIB209_COV_CSV}" mixture_model

run_eslasso 23146 out/prod_23146_supercat6_eslasso_baseline_lib209 \
  supercategory_6_non_human_cells_and_plants 209 baseline
run_eslasso 23146 out/prod_23146_supercat6_eslasso_mixture_model_lib209 \
  supercategory_6_non_human_cells_and_plants 209 mixture_model
run_compassign_pp_ridge 23146 out/prod_23146_supercat6_compassign_pp_ridge_baseline_lib209 \
  supercategory_6_non_human_cells_and_plants 209 "${LIB209_COV_CSV}" baseline
run_compassign_pp_ridge 23146 out/prod_23146_supercat6_compassign_pp_ridge_mixture_model_lib209 \
  supercategory_6_non_human_cells_and_plants 209 "${LIB209_COV_CSV}" mixture_model

run_eslasso 23059 out/prod_23059_supercat6_eslasso_baseline_lib209 \
  supercategory_6_non_human_cells_and_plants 209 baseline
run_eslasso 23059 out/prod_23059_supercat6_eslasso_mixture_model_lib209 \
  supercategory_6_non_human_cells_and_plants 209 mixture_model
run_compassign_pp_ridge 23059 out/prod_23059_supercat6_compassign_pp_ridge_baseline_lib209 \
  supercategory_6_non_human_cells_and_plants 209 "${LIB209_COV_CSV}" baseline
run_compassign_pp_ridge 23059 out/prod_23059_supercat6_compassign_pp_ridge_mixture_model_lib209 \
  supercategory_6_non_human_cells_and_plants 209 "${LIB209_COV_CSV}" mixture_model

echo ""
echo "[sally_test] Summary"
conda run --no-capture-output -n sally python - <<'PY'
import pickle
from pathlib import Path

ROOT = Path("external_repos/sally/out")

RUNS = [
    # supercat, ssid, label, run_dir, model_key, lib_id
    ("6", 12307, "ESLASSO + baseline", "prod_12307_supercat6_eslasso_baseline", "ESLASSO", 208),
    ("6", 12307, "ESLASSO + mixture", "prod_12307_supercat6_eslasso_mixture_model", "ESLASSO", 208),
    (
        "6",
        12307,
        "COMPASSIGN_PP_RIDGE + baseline",
        "prod_12307_supercat6_compassign_pp_ridge_baseline",
        "COMPASSIGN_PP_RIDGE",
        208,
    ),
    (
        "6",
        12307,
        "COMPASSIGN_PP_RIDGE + mixture",
        "prod_12307_supercat6_compassign_pp_ridge_mixture_model",
        "COMPASSIGN_PP_RIDGE",
        208,
    ),
    ("8", 12609, "ESLASSO + baseline", "prod_12609_supercat8_eslasso_baseline", "ESLASSO", 208),
    ("8", 12609, "ESLASSO + mixture", "prod_12609_supercat8_eslasso_mixture_model", "ESLASSO", 208),
    (
        "8",
        12609,
        "COMPASSIGN_PP_RIDGE + baseline",
        "prod_12609_supercat8_compassign_pp_ridge_baseline",
        "COMPASSIGN_PP_RIDGE",
        208,
    ),
    (
        "8",
        12609,
        "COMPASSIGN_PP_RIDGE + mixture",
        "prod_12609_supercat8_compassign_pp_ridge_mixture_model",
        "COMPASSIGN_PP_RIDGE",
        208,
    ),
    ("8", 13208, "ESLASSO + baseline", "prod_13208_supercat8_eslasso_baseline", "ESLASSO", 208),
    ("8", 13208, "ESLASSO + mixture", "prod_13208_supercat8_eslasso_mixture_model", "ESLASSO", 208),
    (
        "8",
        13208,
        "COMPASSIGN_PP_RIDGE + baseline",
        "prod_13208_supercat8_compassign_pp_ridge_baseline",
        "COMPASSIGN_PP_RIDGE",
        208,
    ),
    (
        "8",
        13208,
        "COMPASSIGN_PP_RIDGE + mixture",
        "prod_13208_supercat8_compassign_pp_ridge_mixture_model",
        "COMPASSIGN_PP_RIDGE",
        208,
    ),
    ("6", 12725, "ESLASSO + baseline", "prod_12725_supercat6_eslasso_baseline", "ESLASSO", 208),
    ("6", 12725, "ESLASSO + mixture", "prod_12725_supercat6_eslasso_mixture_model", "ESLASSO", 208),
    (
        "6",
        12725,
        "COMPASSIGN_PP_RIDGE + baseline",
        "prod_12725_supercat6_compassign_pp_ridge_baseline",
        "COMPASSIGN_PP_RIDGE",
        208,
    ),
    (
        "6",
        12725,
        "COMPASSIGN_PP_RIDGE + mixture",
        "prod_12725_supercat6_compassign_pp_ridge_mixture_model",
        "COMPASSIGN_PP_RIDGE",
        208,
    ),
    ("8", 20814, "ESLASSO + baseline", "prod_20814_supercat8_eslasso_baseline_lib209", "ESLASSO", 209),
    ("8", 20814, "ESLASSO + mixture", "prod_20814_supercat8_eslasso_mixture_model_lib209", "ESLASSO", 209),
    (
        "8",
        20814,
        "COMPASSIGN_PP_RIDGE + baseline",
        "prod_20814_supercat8_compassign_pp_ridge_baseline_lib209",
        "COMPASSIGN_PP_RIDGE",
        209,
    ),
    (
        "8",
        20814,
        "COMPASSIGN_PP_RIDGE + mixture",
        "prod_20814_supercat8_compassign_pp_ridge_mixture_model_lib209",
        "COMPASSIGN_PP_RIDGE",
        209,
    ),
    ("8", 20159, "ESLASSO + baseline", "prod_20159_supercat8_eslasso_baseline_lib209", "ESLASSO", 209),
    ("8", 20159, "ESLASSO + mixture", "prod_20159_supercat8_eslasso_mixture_model_lib209", "ESLASSO", 209),
    (
        "8",
        20159,
        "COMPASSIGN_PP_RIDGE + baseline",
        "prod_20159_supercat8_compassign_pp_ridge_baseline_lib209",
        "COMPASSIGN_PP_RIDGE",
        209,
    ),
    (
        "8",
        20159,
        "COMPASSIGN_PP_RIDGE + mixture",
        "prod_20159_supercat8_compassign_pp_ridge_mixture_model_lib209",
        "COMPASSIGN_PP_RIDGE",
        209,
    ),
    ("6", 23146, "ESLASSO + baseline", "prod_23146_supercat6_eslasso_baseline_lib209", "ESLASSO", 209),
    ("6", 23146, "ESLASSO + mixture", "prod_23146_supercat6_eslasso_mixture_model_lib209", "ESLASSO", 209),
    (
        "6",
        23146,
        "COMPASSIGN_PP_RIDGE + baseline",
        "prod_23146_supercat6_compassign_pp_ridge_baseline_lib209",
        "COMPASSIGN_PP_RIDGE",
        209,
    ),
    (
        "6",
        23146,
        "COMPASSIGN_PP_RIDGE + mixture",
        "prod_23146_supercat6_compassign_pp_ridge_mixture_model_lib209",
        "COMPASSIGN_PP_RIDGE",
        209,
    ),
    ("6", 23059, "ESLASSO + baseline", "prod_23059_supercat6_eslasso_baseline_lib209", "ESLASSO", 209),
    ("6", 23059, "ESLASSO + mixture", "prod_23059_supercat6_eslasso_mixture_model_lib209", "ESLASSO", 209),
    (
        "6",
        23059,
        "COMPASSIGN_PP_RIDGE + baseline",
        "prod_23059_supercat6_compassign_pp_ridge_baseline_lib209",
        "COMPASSIGN_PP_RIDGE",
        209,
    ),
    (
        "6",
        23059,
        "COMPASSIGN_PP_RIDGE + mixture",
        "prod_23059_supercat6_compassign_pp_ridge_mixture_model_lib209",
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
        f"{lib_id}\t{supercat}\t{ssid}\t{model}"
        f"\t{row['precision']:.4f}\t{row['recall']:.4f}\t{row['ts']:.4f}"
    )
PY

echo ""
echo "[sally_test] Mixture model diagnostics plots"
conda run --no-capture-output -n sally python src/compassign/rt/plot_sally_mixture_model_diagnostics.py \
  --out-root external_repos/sally/out \
  --run-glob "prod_*" \
  --output-subdir evaluation \
  --latex-copy docs/models/images/rt_pymc_multilevel_pooling_report/mixture_model_example_ssid12307_lib208.png \
  --latex-ssid 12307 \
  --latex-lib 208

echo ""
echo "[sally_test] Done. Outputs under: ${SALLY_DIR}/out"
