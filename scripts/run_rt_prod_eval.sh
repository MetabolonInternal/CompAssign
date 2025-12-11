#!/usr/bin/env bash

set -euo pipefail

# Run RT production evaluations for hierarchical (PyMC) and lasso baseline
# models on real-test RT CSVs (libs 208, 209).
#
# Flags:
#   --only-hierarchical  Run only hierarchical model evaluations
#   --only-lasso         Run only lasso baseline evaluations
#   --no-compare         Skip plotting hierarchical vs lasso comparison (default: compare if inputs exist)
#   --cap N              Use models trained at cap-N (default: 5, using output/rt_prod/lib{lib}_cap5)
#
# Assumes:
#   - You are in the repo root
#   - The cap-N models have been trained:
#       output/rt_prod/lib208_capN
#       output/rt_prod/lib209_capN
#   - The following CSVs exist:
#       repo_export/merged_training_realtest_lib208_chemclass_rt_prod.csv
#       repo_export/merged_training_realtest_lib209_chemclass_rt_prod.csv

RUN_HIER=1
RUN_LASSO=1
RUN_COMPARE=1
TRAIN_CAP=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-hierarchical|--only-pymc|--only-hier)
      RUN_LASSO=0
      shift
      ;;
    --only-lasso)
      RUN_HIER=0
      shift
      ;;
    --no-compare)
      RUN_COMPARE=0
      shift
      ;;
    --cap|--train-cap)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1 (expected cap number, e.g. 5 or 50)" >&2
        exit 1
      fi
      TRAIN_CAP="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--only-hierarchical|--only-lasso|--no-compare]" >&2
      exit 1
      ;;
  esac
done

run_pair() {
  local lib_id=$1
  local label=$2    # cap10 or realtest
  local csv=$3

  local train_dir="output/rt_prod/lib${lib_id}_cap${TRAIN_CAP}"

  # Skip this lib entirely if the test CSV is missing, so the script
  # can still run (and compare) for the libs that are available.
  if [[ ! -f "$csv" ]]; then
    echo "== Skipping lib ${lib_id} (${label}): missing test CSV ${csv} =="
    return
  fi

  if [[ "$label" == "cap10" ]]; then
    local hier_json="rt_eval_cap10_streaming.json"
    local lasso_json="rt_eval_lasso_cap10.json"
  else
  local hier_json="rt_eval_realtest_streaming.json"
  local lasso_json="rt_eval_lasso_realtest.json"
  fi

  local hier_trace="${train_dir}/models/rt_trace.nc"
  local hier_config="${train_dir}/config.json"

  # If the PyMC model is missing, skip the entire lib (including lasso).
  if [[ ! -f "$hier_config" || ! -f "$hier_trace" ]]; then
    echo "== Skipping lib ${lib_id} (${label}): missing PyMC model under ${train_dir} =="
    return
  fi

  if [[ $RUN_HIER -eq 1 ]]; then
    echo "== Hierarchical (streaming): ${label} (lib ${lib_id}) =="
    python scripts/pipelines/eval_rt_prod_streaming.py \
      --train-output-dir "$train_dir" \
      --test-csv "$csv" \
      --chunk-size 50000 \
      --max-test-rows 0 \
      --n-samples 200 \
      --label "$label" \
      --output-json "$train_dir/results/${hier_json}"
  fi

  if [[ $RUN_LASSO -eq 1 ]]; then
    echo "== Lasso baseline: ${label} (lib ${lib_id}) =="
    python scripts/pipelines/eval_rt_lasso_baseline.py \
      --train-output-dir "$train_dir" \
      --lib-id "$lib_id" \
      --test-csv "$csv" \
      --chunk-size 50000 \
      --max-test-rows 0 \
      --label "$label" \
      --output-json "$train_dir/results/${lasso_json}"
  fi
}

run_pair 208 "realtest" "repo_export/lib208/realtest/merged_training_realtest_lib208_chemclass_rt_prod.csv"
run_pair 209 "realtest" "repo_export/lib209/realtest/merged_training_realtest_lib209_chemclass_rt_prod.csv"

if [[ $RUN_COMPARE -eq 1 ]]; then
  echo "== Generating hierarchical vs lasso comparison plots by species group =="
  compare_plot() {
    local lib_id=$1
    local label=$2

    # Hierarchical results are taken from the selected TRAIN_CAP model directory.
    local hier_csv="output/rt_prod/lib${lib_id}_cap${TRAIN_CAP}/results/rt_eval_streaming_by_species_group_${label}.csv"

    # Prefer lasso results co-located with the hier model; otherwise fall back to cap5 lasso outputs.
    local lasso_csv_self="output/rt_prod/lib${lib_id}_cap${TRAIN_CAP}/results/rt_eval_lasso_by_species_group_${label}.csv"
    local lasso_csv_cap5="output/rt_prod/lib${lib_id}_cap5/results/rt_eval_lasso_by_species_group_${label}.csv"

    local lasso_csv=""
    if [[ -f "$lasso_csv_self" ]]; then
      lasso_csv="$lasso_csv_self"
    elif [[ -f "$lasso_csv_cap5" ]]; then
      lasso_csv="$lasso_csv_cap5"
    fi

    if [[ ! -f "$hier_csv" ]]; then
      echo "[compare] Missing hierarchical CSV: $hier_csv (skip)"
      return
    fi
    if [[ -z "$lasso_csv" ]]; then
      echo "[compare] Missing lasso CSV for lib${lib_id} ${label} (skip)"
      return
    fi

    local out_dir
    out_dir=$(dirname "$hier_csv")
    python scripts/pipelines/compare_rt_models_by_group.py \
      --hier-csv "$hier_csv" \
      --lasso-csv "$lasso_csv" \
      --label "$label" \
      --output "${out_dir}/rt_eval_compare_${label}_rmse_by_species_group.png"
  }

  compare_plot 208 realtest
  compare_plot 209 realtest
fi

echo "== RT production evaluation runs complete =="
