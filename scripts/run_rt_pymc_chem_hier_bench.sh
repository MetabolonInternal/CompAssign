#!/usr/bin/env bash

set -euo pipefail

# Benchmark the partially-collapsed PyMC RT model with chemistry hierarchy intercepts (capN -> realtest).
#
# Examples:
#   ./scripts/run_rt_pymc_chem_hier_bench.sh
#   ./scripts/run_rt_pymc_chem_hier_bench.sh --cap 5 --lib 208,209 --method advi
#   ./scripts/run_rt_pymc_chem_hier_bench.sh --method nuts --max-groups 200
#
# Notes on NUTS:
# - NUTS is only practical on small subsets because this model has O(G + K + C) latent variables.
# - You can still run NUTS on the full dataset (max-groups=0), but expect it to be extremely slow.

CAP="cap5"
METHOD="advi" # advi|map|nuts
LIB_IDS=()

SEED="42"
LAMBDA_SLOPES="1e-3"

MAP_MAXEVAL="20000"

ADVI_STEPS="5000"
ADVI_DRAWS="200"

NUTS_DRAWS="200"
NUTS_TUNE="200"
NUTS_CHAINS="2"
TARGET_ACCEPT="0.9"
MAX_GROUPS=""

CHEM_EMBEDDINGS_PATH="resources/metabolites/embeddings_chemberta_pca20.parquet"
CHEM_FEATURE_CENTER="global"

PPC_DRAWS="200"
PPC_CHUNK_SIZE="50000"
PPC_MAX_TEST_ROWS="0"

SKIP_BASELINE=0
SKIP_EVAL=0
NO_LOG=0

usage() {
  cat <<EOF >&2
Usage: $0 [options]

Options:
  --cap <capN|N>              Training cap (default: cap5)
  --lib <id[,id...]>          Lib ids (default: 208,209)
  --method <advi|map|nuts>    Inference method for chem_hier (default: advi)
  --seed <int>                Seed (default: 42)
  --lambda-slopes <float>     Fixed ridge penalty lambda (default: 1e-3)

MAP options:
  --map-maxeval <int>         Max evaluations for find_MAP (default: 20000)

ADVI options:
  --advi-steps <int>          ADVI steps (default: 5000)
  --advi-draws <int>          ADVI draws (default: 200)

NUTS options:
  --max-groups <int>          Limit groups used for training (default: 200 if --method nuts; 0 = all)
  --nuts-draws <int>          NUTS draws (default: 200)
  --nuts-tune <int>           NUTS tune (default: 200)
  --nuts-chains <int>         NUTS chains (default: 2)
  --target-accept <float>     NUTS target_accept (default: 0.9)

Chemistry options:
  --chem-embeddings-path <p>  Embedding parquet (default: resources/metabolites/embeddings_chemberta_pca20.parquet)
  --chem-feature-center <none|global> Center embeddings (default: global)

Other:
  --skip-baseline             Skip baseline collapsed model training/eval
  --skip-eval                 Skip realtest evaluation
  --ppc-draws <int>           Posterior draws for PPC eval (default: 200)
  --ppc-chunk-size <int>      Chunk size for PPC eval (default: 50000)
  --ppc-max-test-rows <int>   Max rows for PPC eval (default: 0 = all)
  --no-log                    Disable logging
EOF
}

find_train_csv() {
  local lib_id=$1
  local cap=$2
  local candidates=(
    "repo_export/lib${lib_id}/${cap}/merged_training_all_lib${lib_id}_${cap}_chemclass_rt_prod.csv"
    "repo_export/lib${lib_id}/${cap}/merged_training_*_lib${lib_id}_${cap}_chemclass_rt_prod.csv"
    "repo_export/merged_training_*_lib${lib_id}_${cap}_chemclass_rt_prod.csv"
  )
  for c in "${candidates[@]}"; do
    for f in $c; do
      echo "$f"
      return 0
    done
  done
  return 1
}

find_realtest_csv() {
  local lib_id=$1
  local candidates=(
    "repo_export/lib${lib_id}/realtest/merged_training_realtest_lib${lib_id}_chemclass_rt_prod.csv"
    "repo_export/lib${lib_id}/realtest/merged_training_*_lib${lib_id}_chemclass_rt_prod.csv"
  )
  for c in "${candidates[@]}"; do
    for f in $c; do
      echo "$f"
      return 0
    done
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cap)
      [[ $# -ge 2 ]] || { echo "Missing value for --cap" >&2; exit 1; }
      CAP="$2"
      shift 2
      ;;
    --lib)
      [[ $# -ge 2 ]] || { echo "Missing value for --lib" >&2; exit 1; }
      IFS=',' read -r -a LIB_SPLIT <<< "$2"
      for lib_val in "${LIB_SPLIT[@]}"; do
        [[ -n "${lib_val}" ]] && LIB_IDS+=("${lib_val}")
      done
      shift 2
      ;;
    --method)
      [[ $# -ge 2 ]] || { echo "Missing value for --method" >&2; exit 1; }
      METHOD="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; exit 1; }
      SEED="$2"
      shift 2
      ;;
    --lambda-slopes)
      [[ $# -ge 2 ]] || { echo "Missing value for --lambda-slopes" >&2; exit 1; }
      LAMBDA_SLOPES="$2"
      shift 2
      ;;
    --map-maxeval)
      [[ $# -ge 2 ]] || { echo "Missing value for --map-maxeval" >&2; exit 1; }
      MAP_MAXEVAL="$2"
      shift 2
      ;;
    --advi-steps)
      [[ $# -ge 2 ]] || { echo "Missing value for --advi-steps" >&2; exit 1; }
      ADVI_STEPS="$2"
      shift 2
      ;;
    --advi-draws)
      [[ $# -ge 2 ]] || { echo "Missing value for --advi-draws" >&2; exit 1; }
      ADVI_DRAWS="$2"
      shift 2
      ;;
    --max-groups)
      [[ $# -ge 2 ]] || { echo "Missing value for --max-groups" >&2; exit 1; }
      MAX_GROUPS="$2"
      shift 2
      ;;
    --nuts-draws)
      [[ $# -ge 2 ]] || { echo "Missing value for --nuts-draws" >&2; exit 1; }
      NUTS_DRAWS="$2"
      shift 2
      ;;
    --nuts-tune)
      [[ $# -ge 2 ]] || { echo "Missing value for --nuts-tune" >&2; exit 1; }
      NUTS_TUNE="$2"
      shift 2
      ;;
    --nuts-chains)
      [[ $# -ge 2 ]] || { echo "Missing value for --nuts-chains" >&2; exit 1; }
      NUTS_CHAINS="$2"
      shift 2
      ;;
    --target-accept)
      [[ $# -ge 2 ]] || { echo "Missing value for --target-accept" >&2; exit 1; }
      TARGET_ACCEPT="$2"
      shift 2
      ;;
    --chem-embeddings-path)
      [[ $# -ge 2 ]] || { echo "Missing value for --chem-embeddings-path" >&2; exit 1; }
      CHEM_EMBEDDINGS_PATH="$2"
      shift 2
      ;;
    --chem-feature-center)
      [[ $# -ge 2 ]] || { echo "Missing value for --chem-feature-center" >&2; exit 1; }
      CHEM_FEATURE_CENTER="$2"
      shift 2
      ;;
    --skip-baseline)
      SKIP_BASELINE=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --ppc-draws)
      [[ $# -ge 2 ]] || { echo "Missing value for --ppc-draws" >&2; exit 1; }
      PPC_DRAWS="$2"
      shift 2
      ;;
    --ppc-chunk-size)
      [[ $# -ge 2 ]] || { echo "Missing value for --ppc-chunk-size" >&2; exit 1; }
      PPC_CHUNK_SIZE="$2"
      shift 2
      ;;
    --ppc-max-test-rows)
      [[ $# -ge 2 ]] || { echo "Missing value for --ppc-max-test-rows" >&2; exit 1; }
      PPC_MAX_TEST_ROWS="$2"
      shift 2
      ;;
    --no-log)
      NO_LOG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

# Default: log with a pseudo-TTY so PyMC progress bars still render.
# If `script` is unavailable, fall back to no logging (to preserve interactivity).
if [[ "${NO_LOG}" -ne 1 && "${COMPASSIGN_TTY_LOGGED:-0}" != "1" ]]; then
  if command -v script >/dev/null 2>&1; then
    LOG_DIR="output/rt_pymc_chem_hier_bench/logs"
    mkdir -p "${LOG_DIR}"
    LOG_TS="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="${LOG_DIR}/run_rt_pymc_chem_hier_bench_${LOG_TS}.log"
    echo "Logging run_rt_pymc_chem_hier_bench.sh output to ${LOG_FILE}"
    echo "Command: $0 $*" >> "${LOG_FILE}"
    # shellcheck disable=SC2016
    exec script -q "${LOG_FILE}" bash -c 'export COMPASSIGN_TTY_LOGGED=1; exec "$0" --no-log "$@"' "$0" "$@"
  else
    echo "Note: 'script' not found; disabling logging to keep progress bars interactive."
  fi
fi

# Default libs 208 and 209.
if [[ ${#LIB_IDS[@]} -eq 0 ]]; then
  LIB_IDS=(208 209)
fi

# Normalize cap (allow numeric like 5 -> cap5).
if [[ "${CAP}" =~ ^[0-9]+$ ]]; then
  CAP="cap${CAP}"
fi

if [[ "${METHOD}" != "map" && "${METHOD}" != "advi" && "${METHOD}" != "nuts" ]]; then
  echo "Unknown --method: ${METHOD} (expected map|advi|nuts)" >&2
  exit 1
fi

if [[ -z "${MAX_GROUPS}" ]]; then
  if [[ "${METHOD}" == "nuts" ]]; then
    MAX_GROUPS="200"
  else
    MAX_GROUPS="0"
  fi
fi

if [[ "${METHOD}" == "nuts" ]]; then
  if [[ "${MAX_GROUPS}" -gt 0 && "${SKIP_EVAL}" -ne 1 ]]; then
    echo "Note: --method nuts uses --max-groups ${MAX_GROUPS}; realtest eval with --require-seen-group will be on a small subset only."
    echo "      Use --skip-eval for sampler-only debugging, or run --method map/advi for full cap${CAP#cap} evaluation."
  elif [[ "${MAX_GROUPS}" == "0" ]]; then
    echo "Warning: --method nuts with --max-groups 0 will be extremely slow and may require large RAM."
  fi
fi

for LIB_ID in "${LIB_IDS[@]}"; do
  TRAIN_CSV=$(find_train_csv "${LIB_ID}" "${CAP}") || {
    echo "Could not find training CSV for lib${LIB_ID} ${CAP}" >&2
    exit 1
  }
  REALTEST_CSV=$(find_realtest_csv "${LIB_ID}") || {
    echo "Could not find realtest CSV for lib${LIB_ID}" >&2
    exit 1
  }

  echo "==== lib${LIB_ID} ${CAP} ===="
  echo "Train CSV: ${TRAIN_CSV}"
  echo "Realtest CSV: ${REALTEST_CSV}"

  BASE_OUT="output/tmp_pymc_collapsed_lib${LIB_ID}_${CAP}"
  CHEM_OUT="output/tmp_pymc_chem_hier_lib${LIB_ID}_${CAP}_${METHOD}"

  EXTRA=()
  if [[ "${METHOD}" == "map" ]]; then
    EXTRA+=(--map-maxeval "${MAP_MAXEVAL}")
  elif [[ "${METHOD}" == "advi" ]]; then
    EXTRA+=(--advi-steps "${ADVI_STEPS}" --advi-draws "${ADVI_DRAWS}")
  else
    EXTRA+=(
      --max-groups "${MAX_GROUPS}"
      --nuts-draws "${NUTS_DRAWS}"
      --nuts-tune "${NUTS_TUNE}"
      --nuts-chains "${NUTS_CHAINS}"
      --target-accept "${TARGET_ACCEPT}"
    )
  fi

  if [[ "${SKIP_BASELINE}" -ne 1 ]]; then
    echo "--- Train baseline (collapsed intercept + collapsed slopes; map) -> ${BASE_OUT}"
    time poetry run python scripts/pipelines/train_rt_pymc_collapsed_ridge.py \
      --data-csv "${TRAIN_CSV}" \
      --include-es-all \
      --output-dir "${BASE_OUT}" \
      --seed "${SEED}" \
      --lambda-mode fixed \
      --lambda-slopes "${LAMBDA_SLOPES}" \
      --intercept-mode collapsed \
      --method map \
      --map-maxeval "${MAP_MAXEVAL}"
  fi

  echo "--- Train chem_hier (explicit intercepts + collapsed slopes; ${METHOD}) -> ${CHEM_OUT}"
  time poetry run python scripts/pipelines/train_rt_pymc_collapsed_ridge.py \
    --data-csv "${TRAIN_CSV}" \
    --include-es-all \
    --output-dir "${CHEM_OUT}" \
    --seed "${SEED}" \
    --lambda-mode fixed \
    --lambda-slopes "${LAMBDA_SLOPES}" \
    --intercept-mode explicit \
    --intercept-prior chem_hier \
    --chem-embeddings-path "${CHEM_EMBEDDINGS_PATH}" \
    --chem-feature-center "${CHEM_FEATURE_CENTER}" \
    --method "${METHOD}" \
    "${EXTRA[@]}"

  if [[ "${SKIP_EVAL}" -ne 1 ]]; then
    if [[ "${SKIP_BASELINE}" -ne 1 ]]; then
      echo "--- Eval baseline on realtest (seen groups only)"
      poetry run python scripts/pipelines/eval_rt_coeff_summaries_by_species_cluster.py \
        --coeff-npz "${BASE_OUT}/models/stage1_coeff_summaries_posterior.npz" \
        --test-csv "${REALTEST_CSV}" \
        --output-dir "${BASE_OUT}/results" \
        --require-seen-group \
        --label "realtest_seen"
    fi

    if [[ "${METHOD}" == "advi" || "${METHOD}" == "nuts" ]]; then
      echo "--- Eval chem_hier on realtest (seen groups only; posterior predictive)"
      poetry run python scripts/pipelines/eval_rt_pymc_posterior_predictive.py \
        --train-output-dir "${CHEM_OUT}" \
        --test-csv "${REALTEST_CSV}" \
        --require-seen-group \
        --max-test-rows "${PPC_MAX_TEST_ROWS}" \
        --chunk-size "${PPC_CHUNK_SIZE}" \
        --n-ppc-draws "${PPC_DRAWS}" \
        --label "realtest_seen"
    else
      echo "--- Eval chem_hier on realtest (seen groups only; Normal approx)"
      poetry run python scripts/pipelines/eval_rt_coeff_summaries_by_species_cluster.py \
        --coeff-npz "${CHEM_OUT}/models/stage1_coeff_summaries_posterior.npz" \
        --test-csv "${REALTEST_CSV}" \
        --output-dir "${CHEM_OUT}/results" \
        --require-seen-group \
        --label "realtest_seen"
    fi
  fi
done

echo "Done."
