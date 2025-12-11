#!/usr/bin/env bash
set -euo pipefail

# End-to-end helper to:
#   1) fetch RT training data from Pachyderm (create_training_data + combine_predictors)
#   2) merge into per-export merged_training Parquets
#   3) split merged Parquet by lib_id
#
# This script assumes:
#   - PROJECT, JOB_208, JOB_209 as in pachyderm_fetch_rt_training.sh
#   - data/pachyderm_rt_training populated (or it will populate it)
#
# It does NOT yet apply the train/test split or build RT CSVs; those are
# separate steps once the merged per-lib Parquets exist.

PROJECT=${PROJECT:-autocuration_platinum}
JOB_208=${JOB_208:-7210c028155f4241b2ad3132c004fc01}
JOB_209=${JOB_209:-95661767ab0e4fe28aaeacca8230449b}
FETCH_DIR=${FETCH_DIR:-data/pachyderm_rt_training}

echo "Project: $PROJECT"
echo "Fetch dir: $FETCH_DIR"
echo "Lib 208 job: $JOB_208"
echo "Lib 209 job: $JOB_209"

mkdir -p "$FETCH_DIR"

echo "== Step 1: Fetch RT training data (create_training_data + combine_predictors) =="

EXPORT_208_ROOT="$FETCH_DIR/job_${JOB_208}"
EXPORT_209_ROOT="$FETCH_DIR/job_${JOB_209}"

if [[ -d "${EXPORT_208_ROOT}/create_training_data" && -d "${EXPORT_208_ROOT}/combine_predictors" && \
      -d "${EXPORT_209_ROOT}/create_training_data" && -d "${EXPORT_209_ROOT}/combine_predictors" ]]; then
  echo "[skip] Existing training exports found under $FETCH_DIR, skipping Pachyderm fetch."
else
  PROJECT="$PROJECT" JOB_208="$JOB_208" JOB_209="$JOB_209" \
    scripts/pachyderm_fetch_rt_training.sh "$FETCH_DIR"
fi

echo "== Step 2: Merge Pachyderm training exports into merged_training =="

# 208 export root
MERGED_208_DIR="repo_export/merged_training_${JOB_208}"
mkdir -p "$MERGED_208_DIR"
python scripts/pipelines/merge_pachyderm_training.py \
  -i "$EXPORT_208_ROOT" \
  -o "$MERGED_208_DIR"

# 209 export root
MERGED_209_DIR="repo_export/merged_training_${JOB_209}"
mkdir -p "$MERGED_209_DIR"
python scripts/pipelines/merge_pachyderm_training.py \
  -i "$EXPORT_209_ROOT" \
  -o "$MERGED_209_DIR"

echo "== Step 3: Split merged Parquet by lib_id =="

python scripts/pipelines/split_merged_by_lib.py \
  "$MERGED_208_DIR/merged_training_all.parquet"

python scripts/pipelines/split_merged_by_lib.py \
  "$MERGED_209_DIR/merged_training_all.parquet"

echo "Done. Per-lib merged Parquets should be under:"
echo "  $MERGED_208_DIR/merged_training_all_lib208.parquet"
echo "  $MERGED_209_DIR/merged_training_all_lib209.parquet"
