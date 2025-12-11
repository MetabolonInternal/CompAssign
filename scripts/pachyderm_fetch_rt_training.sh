#!/usr/bin/env bash
set -euo pipefail

# Fetch RT training data from Pachyderm's create_training_data pipeline
# for specific job/commit IDs and libraries.
#
# This is a focused variant of sally/scripts/pachyderm_fetch_results.sh:
# it only pulls the create_training_data repo for the libs we care about.
#
# Defaults (override via environment variables):
#   PROJECT=autocuration_platinum
#   JOB_208=7210c028155f4241b2ad3132c004fc01
#   JOB_209=95661767ab0e4fe28aaeacca8230449b
# Usage:
#   scripts/pachyderm_fetch_rt_training.sh [DEST_DIR]
#
# Example:
#   PROJECT=autocuration_platinum \
#   JOB_208=7210c028155f4241b2ad3132c004fc01 \
#   JOB_209=95661767ab0e4fe28aaeacca8230449b \
#   scripts/pachyderm_fetch_rt_training.sh data/pachyderm_rt_training

DEST_DIR=${1:-data/pachyderm_rt_training}
PROJECT=${PROJECT:-autocuration_platinum}
JOB_208=${JOB_208:-7210c028155f4241b2ad3132c004fc01}
JOB_209=${JOB_209:-95661767ab0e4fe28aaeacca8230449b}

echo "Project: $PROJECT"
echo "Destination: $DEST_DIR"
echo "Lib 208 job/commit: $JOB_208"
echo "Lib 209 job/commit: $JOB_209"
mkdir -p "$DEST_DIR"

fetch_lib_from_job() {
  local lib_id=$1   # e.g. 208 or 209
  local job_id=$2   # Pachyderm commit ID

  local out_dir="$DEST_DIR/job_${job_id}"
  mkdir -p "$out_dir/create_training_data/lib_id=${lib_id}"
  mkdir -p "$out_dir/combine_predictors/lib_id=${lib_id}"

  echo "[fetch] create_training_data@${job_id}:/lib_id=${lib_id}/ -> $out_dir/create_training_data/lib_id=${lib_id}"
  pachctl get file "create_training_data@${job_id}:/lib_id=${lib_id}/" \
    -r \
    --output "$out_dir/create_training_data/lib_id=${lib_id}" \
    --project "$PROJECT" \
    --progress=false \
    --retry

  echo "[fetch] combine_predictors@${job_id}:/lib_id=${lib_id}/ -> $out_dir/combine_predictors/lib_id=${lib_id}"
  pachctl get file "combine_predictors@${job_id}:/lib_id=${lib_id}/" \
    -r \
    --output "$out_dir/combine_predictors/lib_id=${lib_id}" \
    --project "$PROJECT" \
    --progress=false \
    --retry
}

echo "== Fetching lib 208 training data (create_training_data + combine_predictors) =="
fetch_lib_from_job 208 "$JOB_208"

echo "== Fetching lib 209 training data (create_training_data + combine_predictors) =="
fetch_lib_from_job 209 "$JOB_209"

echo "== Summary =="
du -sh "$DEST_DIR"/* 2>/dev/null || true
