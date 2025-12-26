#!/usr/bin/env bash

set -euo pipefail

# Build capped RT training datasets (cap20/50/100) for libs 208 and 209.
# This mirrors the cap5/cap10 pipeline:
#   1) sample_per_species_compound.py
#   2) attach_chem_classes_and_filter.py
#   3) make_rt_prod_csv_from_merged.py
#
# Inputs (must exist):
#   - repo_export/merged_training/merged_training_all_lib{lib}.parquet          (uncapped per-lib parquet)
#   - repo_export/lib{lib}/mappings/lib_comp_chem_mapping_lib{lib}.csv         (comp -> chem mapping)
#   - repo_export/lib{lib}/species_mapping/merged_training_*_lib{lib}_species_mapping.csv
#   - resources/metabolites/chem_classes_k32.parquet
#
# Outputs (per lib, per cap):
#   - repo_export/lib{lib}/cap{cap}/merged_training_all_lib{lib}_cap{cap}.parquet
#   - repo_export/lib{lib}/cap{cap}/merged_training_all_lib{lib}_cap{cap}_chemclass.parquet
#   - repo_export/lib{lib}/cap{cap}/merged_training_all_lib{lib}_cap{cap}_chemclass_rt_prod.csv
#
# Usage:
#   ./src/compassign/rt/data_prep/build_rt_cap_datasets.sh
#   ./src/compassign/rt/data_prep/build_rt_cap_datasets.sh --libs 208,209 --caps 100
#   ./src/compassign/rt/data_prep/build_rt_cap_datasets.sh --libs 208 --caps 5,10,100

usage() {
  cat <<'EOF'
Usage: build_rt_cap_datasets.sh [--libs 208,209] [--caps 5,10,20,...] [--species-map-208 PATH] [--species-map-209 PATH]

Build capped RT training datasets (Parquet + RT CSV) for one or more libs and caps.

Options:
  --libs <csv>   Comma-separated lib ids (default: 208,209)
  --caps <csv>   Comma-separated caps (default: 5,10,20,50,100,200,500,1000)
  --species-map-208 <path>  Species mapping CSV for lib208 (optional; overrides default path)
  --species-map-209 <path>  Species mapping CSV for lib209 (optional; overrides default path)
  -h, --help     Show this help
EOF
}

LIBS_CSV="208,209"
CAPS_CSV="5,10,20,50,100,200,500,1000"
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

IFS=',' read -r -a LIBS <<< "${LIBS_CSV}"
IFS=',' read -r -a CAPS <<< "${CAPS_CSV}"

CHEM_CLASSES="resources/metabolites/chem_classes_k32.parquet"
if [[ ! -f "$CHEM_CLASSES" ]]; then
  echo "Missing chem classes file: $CHEM_CLASSES" >&2
  exit 1
fi

for lib in "${LIBS[@]}"; do
  input_parq="repo_export/merged_training/merged_training_all_lib${lib}.parquet"
  map_csv="repo_export/lib${lib}/mappings/lib_comp_chem_mapping_lib${lib}.csv"

  if [[ ! -f "$input_parq" ]]; then
    echo "Missing input parquet: $input_parq" >&2
    exit 1
  fi
  if [[ ! -f "$map_csv" ]]; then
    echo "Missing lib mapping: $map_csv" >&2
    exit 1
  fi
  species_map=""
  if [[ "${lib}" == "208" && -n "${SPECIES_MAP_208}" ]]; then
    species_map="${SPECIES_MAP_208}"
  elif [[ "${lib}" == "209" && -n "${SPECIES_MAP_209}" ]]; then
    species_map="${SPECIES_MAP_209}"
  else
    preferred="repo_export/lib${lib}/species_mapping/merged_training_all_lib${lib}_species_mapping.csv"
    species_map="${preferred}"
  fi
  if [[ ! -f "${species_map}" ]]; then
    echo "Missing species mapping CSV: ${species_map}" >&2
    exit 1
  fi

  for cap in "${CAPS[@]}"; do
    out_dir="repo_export/lib${lib}/cap${cap}"
    mkdir -p "$out_dir"

    cap_parq="${out_dir}/merged_training_all_lib${lib}_cap${cap}.parquet"
    cap_parq_chem="${out_dir}/merged_training_all_lib${lib}_cap${cap}_chemclass.parquet"
    cap_csv="${out_dir}/merged_training_all_lib${lib}_cap${cap}_chemclass_rt_prod.csv"

    if [[ -f "$cap_csv" ]]; then
      echo "[lib${lib} cap${cap}] RT CSV exists, skipping: $cap_csv"
      rm -f "$cap_parq" "$cap_parq_chem"
      continue
    fi

    if [[ ! -f "$cap_parq" ]]; then
      echo "[lib${lib} cap${cap}] Sampling to $cap_parq"
      python -m compassign.rt.data_prep.sample_per_species_compound \
        "$input_parq" \
        --cap-per-pair "$cap" \
        --seed 42 \
        --lib-mapping "$map_csv" \
        --classes "$CHEM_CLASSES" \
        --species-mapping "$species_map" \
        --output "$cap_parq"
    else
      echo "[lib${lib} cap${cap}] Sampling output exists, skipping: $cap_parq"
    fi

    if [[ ! -f "$cap_parq_chem" ]]; then
      echo "[lib${lib} cap${cap}] Attaching chem classes to $cap_parq_chem"
      python -m compassign.rt.data_prep.attach_chem_classes_and_filter \
        --input "$cap_parq" \
        --lib-mapping "$map_csv" \
        --classes "$CHEM_CLASSES" \
        --output "$cap_parq_chem"
    else
      echo "[lib${lib} cap${cap}] Chemclass output exists, skipping: $cap_parq_chem"
    fi

    echo "[lib${lib} cap${cap}] Building RT CSV $cap_csv"
    python -m compassign.rt.data_prep.make_rt_prod_csv_from_merged \
      --input "$cap_parq_chem" \
      --species-mapping "$species_map" \
      --output "$cap_csv"

    rm -f "$cap_parq" "$cap_parq_chem"
  done
done

echo "All requested caps built (or skipped if present)."
