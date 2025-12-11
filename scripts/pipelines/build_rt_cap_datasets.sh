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
#   ./scripts/pipelines/build_rt_cap_datasets.sh

LIBS=("208" "209")
CAPS=("5" "10" "20" "50" "100" "200" "500" "1000")

CHEM_CLASSES="resources/metabolites/chem_classes_k32.parquet"
if [[ ! -f "$CHEM_CLASSES" ]]; then
  echo "Missing chem classes file: $CHEM_CLASSES" >&2
  exit 1
fi

for lib in "${LIBS[@]}"; do
  input_parq="repo_export/merged_training/merged_training_all_lib${lib}.parquet"
  map_csv="repo_export/lib${lib}/mappings/lib_comp_chem_mapping_lib${lib}.csv"
  sm_glob=(repo_export/lib${lib}/species_mapping/merged_training_*_lib${lib}_species_mapping.csv)

  if [[ ! -f "$input_parq" ]]; then
    echo "Missing input parquet: $input_parq" >&2
    exit 1
  fi
  if [[ ! -f "$map_csv" ]]; then
    echo "Missing lib mapping: $map_csv" >&2
    exit 1
  fi
  if [[ ${#sm_glob[@]} -eq 0 || ! -f "${sm_glob[0]}" ]]; then
    echo "Missing species mapping CSV under repo_export/lib${lib}/species_mapping/" >&2
    exit 1
  fi
  species_map="${sm_glob[0]}"

  for cap in "${CAPS[@]}"; do
    out_dir="repo_export/lib${lib}/cap${cap}"
    mkdir -p "$out_dir"

    cap_parq="${out_dir}/merged_training_all_lib${lib}_cap${cap}.parquet"
    cap_parq_chem="${out_dir}/merged_training_all_lib${lib}_cap${cap}_chemclass.parquet"
    cap_csv="${out_dir}/merged_training_all_lib${lib}_cap${cap}_chemclass_rt_prod.csv"

    if [[ ! -f "$cap_parq" ]]; then
      echo "[lib${lib} cap${cap}] Sampling to $cap_parq"
      python scripts/pipelines/sample_per_species_compound.py \
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
      python scripts/pipelines/attach_chem_classes_and_filter.py \
        --input "$cap_parq" \
        --lib-mapping "$map_csv" \
        --classes "$CHEM_CLASSES" \
        --output "$cap_parq_chem"
    else
      echo "[lib${lib} cap${cap}] Chemclass output exists, skipping: $cap_parq_chem"
    fi

    if [[ ! -f "$cap_csv" ]]; then
      echo "[lib${lib} cap${cap}] Building RT CSV $cap_csv"
      python scripts/pipelines/make_rt_prod_csv_from_merged.py \
        --input "$cap_parq_chem" \
        --species-mapping "$species_map" \
        --output "$cap_csv"
    else
      echo "[lib${lib} cap${cap}] RT CSV exists, skipping: $cap_csv"
    fi
  done
done

echo "All requested caps built (or skipped if present)."
