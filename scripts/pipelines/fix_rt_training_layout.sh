#!/usr/bin/env bash
set -euo pipefail

# Fix the on-disk layout of previously downloaded RT training data so that it
# matches what merge_pachyderm_training.py expects.
#
# This script assumes you have already run:
#   scripts/pachyderm_fetch_rt_training.sh data/pachyderm_rt_training
# with the original implementation that wrote:
#   job_<JOB>/create_training_data/group=...
#   job_<JOB>/combine_predictors/species_matrix_type=...
#
# It rearranges those directories into:
#   job_<JOB>/create_training_data/lib_id=<LIB>/group=...
#   job_<JOB>/combine_predictors/lib_id=<LIB>/species_matrix_type=...
#
# so that:
#   python scripts/pipelines/merge_pachyderm_training.py -i job_<JOB> ...
# works without re-downloading from Pachyderm.

ROOT=${1:-data/pachyderm_rt_training}

JOB_208=${JOB_208:-7210c028155f4241b2ad3132c004fc01}
JOB_209=${JOB_209:-95661767ab0e4fe28aaeacca8230449b}

fix_job() {
  local lib_id=$1   # e.g. 208 or 209
  local job_id=$2   # Pachyderm commit ID suffix in job_<JOB>

  local job_dir="${ROOT}/job_${job_id}"
  local create="${job_dir}/create_training_data"
  local combine="${job_dir}/combine_predictors"

  echo "[fix] job=${job_id}, lib_id=${lib_id}"

  if [[ -d "$create" ]]; then
    local create_lib="${create}/lib_id=${lib_id}"
    if [[ ! -d "$create_lib" ]]; then
      echo "[fix]   moving create_training_data groups into lib_id=${lib_id}"
      mkdir -p "$create_lib"
      for d in "$create"/group=*; do
        [[ -d "$d" ]] || continue
        mv "$d" "$create_lib/"
      done
    else
      echo "[fix]   create_training_data already has lib_id=${lib_id}, skipping"
    fi
  else
    echo "[fix]   WARNING: create_training_data not found under $job_dir" >&2
  fi

  if [[ -d "$combine" ]]; then
    local combine_lib="${combine}/lib_id=${lib_id}"
    if [[ ! -d "$combine_lib" ]]; then
      echo "[fix]   moving combine_predictors species_matrix_type dirs into lib_id=${lib_id}"
      mkdir -p "$combine_lib"
      for d in "$combine"/species_matrix_type=*; do
        [[ -d "$d" ]] || continue
        mv "$d" "$combine_lib/"
      done
    else
      echo "[fix]   combine_predictors already has lib_id=${lib_id}, skipping"
    fi
  else
    echo "[fix]   WARNING: combine_predictors not found under $job_dir" >&2
  fi
}

fix_job 208 "$JOB_208"
fix_job 209 "$JOB_209"

echo "[fix] Layout normalisation complete under $ROOT"

