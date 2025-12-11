#!/usr/bin/env bash
set -euo pipefail

# Build ChemBERTa 20D embeddings into resources/metabolites
# Shows a tqdm progress bar during encoding.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_TSV="$REPO_ROOT/resources/metabolites/metabolites.tsv"
OUTPUT_PARQUET="$REPO_ROOT/resources/metabolites/embeddings_chemberta_pca20.parquet"

echo "Building ChemBERTa embeddings from: $INPUT_TSV"
echo "Output artifact: $OUTPUT_PARQUET"

poetry run python "$REPO_ROOT/scripts/data_prep/build_chemberta_embeddings.py" \
  --input "$INPUT_TSV" \
  --output "$OUTPUT_PARQUET"

echo "Done."
