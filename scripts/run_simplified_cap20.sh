#!/usr/bin/env bash
# Run cap20 RT experiments for simplified/interaction model variants.

set -euo pipefail

# Paths (adjust if your repo layout differs)
DATA_CSV="repo_export/lib209/cap20/merged_training_all_lib209_cap20_chemclass_rt_prod.csv"
TEST_CSV="repo_export/lib209/realtest/merged_training_realtest_lib209_chemclass_rt_prod.csv"
OUT_BASE="output/rt_prod"

# Sampler settings (NUTS-lite)
CHAINS=2
DRAWS=1000
TUNE=1000
TARGET_ACCEPT=0.9
SEED=42

# Eval settings
CHUNK_SIZE=50000
N_SAMPLES=200

# Use the conda env with JAX/PyMC installed.
PYTHON_BIN="python"

run_variant() {
  local name="$1"
  shift
  local extra_args=("$@")
  local out_dir="${OUT_BASE}/lib209_cap20_${name}"

  echo "=== Training ${name} -> ${out_dir} ==="
  ${PYTHON_BIN} scripts/pipelines/train_rt_prod.py \
    --data-csv "${DATA_CSV}" \
    --output-dir "${out_dir}" \
    --include-es-group1 \
    --chains "${CHAINS}" \
    --draws "${DRAWS}" \
    --tune "${TUNE}" \
    --target-accept "${TARGET_ACCEPT}" \
    --seed "${SEED}" \
    "${extra_args[@]}"

  echo "=== Evaluating ${name} (realtest) ==="
  ${PYTHON_BIN} scripts/pipelines/eval_rt_prod_streaming.py \
    --train-output-dir "${out_dir}" \
    --test-csv "${TEST_CSV}" \
    --chunk-size "${CHUNK_SIZE}" \
    --max-test-rows 0 \
    --n-samples "${N_SAMPLES}" \
    --label realtest
}

# V1: Simplified (no clusters, small species), class-only gamma, mild prior tighten.
run_variant "v1_simplified" \
  --no-clusters \
  --sigma-species-scale 0.05 \
  --class-only-gamma \
  --sigma-gamma-class-scale 0.3 \
  --sigma-y-loc -3.5 \
  --sigma-y-scale 0.4

# V2: As V1 with slightly tighter gamma_class and noise prior.
run_variant "v2_simplified_tighter" \
  --no-clusters \
  --sigma-species-scale 0.05 \
  --class-only-gamma \
  --sigma-gamma-class-scale 0.25 \
  --sigma-y-loc -3.6 \
  --sigma-y-scale 0.35

# V3: V2 plus species×compound interaction intercepts (delta_sc) with shrinkage.
run_variant "v3_simplified_tighter_sc" \
  --no-clusters \
  --sigma-species-scale 0.05 \
  --class-only-gamma \
  --sigma-gamma-class-scale 0.25 \
  --sigma-y-loc -3.6 \
  --sigma-y-scale 0.35 \
  --species-compound-intercept \
  --sigma-sc-scale 0.1

echo "All runs completed. Outputs are under ${OUT_BASE}/lib209_cap20_*."
