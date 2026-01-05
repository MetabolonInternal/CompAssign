#!/usr/bin/env bash
set -euo pipefail

# One-shot RT data preparation:
#   1) fetch training shards from Pachyderm (create_training_data + combine_predictors)
#   2) merge into a single merged Parquet
#   3) split into per-lib Parquets under repo_export/merged_training/
#   4) regenerate strict species mappings
#   5) build cap100 RT CSVs (train inputs)
#   6) build realtest RT CSVs (eval inputs)
#
# Intended usage: run with no args (defaults hardcoded below).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -ne 0 ]]; then
  echo "ERROR: src/compassign/rt/prep.sh does not take arguments. Run it with no args." >&2
  exit 2
fi

if ! command -v pachctl >/dev/null 2>&1; then
  echo "ERROR: pachctl not found on PATH (required to fetch Pachyderm training data)." >&2
  exit 2
fi

# Pachyderm defaults (edit here when jobs change).
PROJECT="autocuration_platinum"
JOB_208="7210c028155f4241b2ad3132c004fc01"
JOB_209="95661767ab0e4fe28aaeacca8230449b"

LIB_IDS=(208 209)
CAPS="100"

RAW_DIR="${REPO_ROOT}/repo_export/pachyderm_rt_training_raw"
COMBINED_DIR="${REPO_ROOT}/repo_export/pachyderm_rt_training_latest"
MERGED_DIR="${REPO_ROOT}/repo_export/merged_training"

echo "[prep] project=${PROJECT}"
echo "[prep] job_208=${JOB_208}"
echo "[prep] job_209=${JOB_209}"

mkdir -p "${RAW_DIR}"

fetch_lib_from_job() {
  local lib_id="$1"
  local job_id="$2"

  local out_dir="${RAW_DIR}/job_${job_id}"
  mkdir -p "${out_dir}/create_training_data/lib_id=${lib_id}"
  mkdir -p "${out_dir}/combine_predictors/lib_id=${lib_id}"

  echo "[fetch] create_training_data@${job_id} lib${lib_id}"
  pachctl get file "create_training_data@${job_id}:/lib_id=${lib_id}/" \
    -r \
    --output "${out_dir}/create_training_data/lib_id=${lib_id}" \
    --project "${PROJECT}" \
    --progress=false \
    --retry

  echo "[fetch] combine_predictors@${job_id} lib${lib_id}"
  pachctl get file "combine_predictors@${job_id}:/lib_id=${lib_id}/" \
    -r \
    --output "${out_dir}/combine_predictors/lib_id=${lib_id}" \
    --project "${PROJECT}" \
    --progress=false \
    --retry
}

echo "[prep] Step 1/6: fetch Pachyderm training shards"
fetch_lib_from_job 208 "${JOB_208}"
fetch_lib_from_job 209 "${JOB_209}"

echo "[prep] Step 2/6: assemble combined export dir"
rm -rf "${COMBINED_DIR}"
mkdir -p "${COMBINED_DIR}/create_training_data" "${COMBINED_DIR}/combine_predictors"

copy_lib_tree() {
  local lib_id="$1"
  local job_id="$2"
  local job_dir="${RAW_DIR}/job_${job_id}"
  local src_create="${job_dir}/create_training_data/lib_id=${lib_id}"
  local src_pred="${job_dir}/combine_predictors/lib_id=${lib_id}"
  local dst_create="${COMBINED_DIR}/create_training_data/lib_id=${lib_id}"
  local dst_pred="${COMBINED_DIR}/combine_predictors/lib_id=${lib_id}"

  if [[ ! -d "${src_create}" || ! -d "${src_pred}" ]]; then
    echo "ERROR: Missing fetched inputs for lib${lib_id} under ${job_dir}" >&2
    exit 2
  fi

  mkdir -p "${dst_create}" "${dst_pred}"
  cp -R "${src_create}/." "${dst_create}/"
  cp -R "${src_pred}/." "${dst_pred}/"
}

copy_lib_tree 208 "${JOB_208}"
copy_lib_tree 209 "${JOB_209}"

echo "[prep] Step 3/6: merge shards into merged_training_all.parquet"
mkdir -p "${MERGED_DIR}"
poetry run python -u -m compassign.rt.data_prep.merge_pachyderm_training \
  --input-dir "${COMBINED_DIR}" \
  --output-dir "${MERGED_DIR}"

echo "[prep] Step 4/6: split merged_training_all.parquet into per-lib Parquets"
poetry run python -u -m compassign.rt.data_prep.split_merged_by_lib \
  "${MERGED_DIR}/merged_training_all.parquet" \
  --output-dir "${MERGED_DIR}"

echo "[prep] Step 5/6: regenerate strict species mappings"
poetry run python -u -m compassign.rt.data_prep.check_rt_metadata_mapping \
  --libs "208,209" \
  --output-name "merged_training_all_lib{lib}_species_mapping.csv"

echo "[prep] Step 6/6: build cap${CAPS} datasets + realtest CSVs"
bash src/compassign/rt/data_prep/build_rt_cap_datasets.sh --libs "208,209" --caps "${CAPS}"
poetry run python -u -m compassign.rt.data_prep.build_rt_real_test_csvs --out-root repo_export

echo "[prep] Done."
