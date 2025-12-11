#!/usr/bin/env bash
set -euo pipefail

# Reproducible data prep for multilevel RT experiments.
#
# This script regenerates, in order:
#   1) strict species/species_cluster mappings
#   2) capped RT training datasets (e.g. cap100)
#   3) realtest RT CSVs (per lib, under repo_export/lib{lib}/realtest/)
#
# Defaults are chosen to match the expectations of:
#   - scripts/run_rt_prod.sh
#
# Usage:
#   bash scripts/pipelines/run_rt_multilevel_data_prep.sh
#   bash scripts/pipelines/run_rt_multilevel_data_prep.sh --caps 100
#   bash scripts/pipelines/run_rt_multilevel_data_prep.sh --libs 208 --caps 100

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage: run_rt_multilevel_data_prep.sh [--libs 208,209] [--caps 100] [--species-map-208 PATH] [--species-map-209 PATH]

Runs the full multilevel RT data prep:
  1) generate strict species mappings
  2) build capped RT datasets for requested caps
  3) build realtest RT CSVs

Defaults:
  --libs 208,209
  --caps 100
EOF
}

LIBS_CSV="208,209"
CAPS_CSV="100"
SPECIES_MAP_208=""
SPECIES_MAP_209=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --libs)
      LIBS_CSV="${2:-}"
      shift 2
      ;;
    --caps)
      CAPS_CSV="${2:-}"
      shift 2
      ;;
    --species-map-208)
      SPECIES_MAP_208="${2:-}"
      shift 2
      ;;
    --species-map-209)
      SPECIES_MAP_209="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "[prep] libs=${LIBS_CSV}"
echo "[prep] caps=${CAPS_CSV}"

echo "[prep] Step 1/3: generate strict species mappings"
poetry run python -u scripts/pipelines/check_rt_metadata_mapping.py \
  --libs "${LIBS_CSV}" \
  --output-name "merged_training_all_lib{lib}_species_mapping.csv"

echo "[prep] Step 2/3: build capped RT datasets"
CAP_ARGS=(--libs "${LIBS_CSV}" --caps "${CAPS_CSV}")
if [[ -n "${SPECIES_MAP_208}" ]]; then
  CAP_ARGS+=(--species-map-208 "${SPECIES_MAP_208}")
fi
if [[ -n "${SPECIES_MAP_209}" ]]; then
  CAP_ARGS+=(--species-map-209 "${SPECIES_MAP_209}")
fi
bash scripts/pipelines/build_rt_cap_datasets.sh "${CAP_ARGS[@]}"

echo "[prep] Step 3/3: build realtest RT CSVs"
poetry run python -u scripts/pipelines/build_rt_real_test_csvs.py \
  --out-root repo_export

echo "[prep] Done."
