#!/usr/bin/env bash

set -euo pipefail

# Upload minimal RT production training data (lib208/lib209, cap-N)
# from the local repo into the remote mirror used for run_rt_prod.sh.
#
# Usage:
#   ./scripts/upload_rt_prod_data_to_remote.sh [--cap <N|capN>]
#
# Environment overrides:
#   REMOTE_USER   (default: ubuntu)
#   REMOTE_HOST   (default: 167.234.209.8)
#   REMOTE_ROOT   (default: $REMOTE_USER@$REMOTE_HOST:~/MyStorage/CompAssign/)

REMOTE_USER=${REMOTE_USER:-ubuntu}
REMOTE_HOST_DEFAULT="167.234.209.8"
REMOTE_HOST=${REMOTE_HOST:-$REMOTE_HOST_DEFAULT}
REMOTE_ROOT=${REMOTE_ROOT:-"${REMOTE_USER}@${REMOTE_HOST}:~/MyStorage/CompAssign/"}

CAP="cap5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --host" >&2
        exit 1
      fi
      REMOTE_HOST="$2"
      REMOTE_ROOT="${REMOTE_USER}@${REMOTE_HOST}:~/MyStorage/CompAssign/"
      shift 2
      ;;
    --cap)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --cap" >&2
        exit 1
      fi
      CAP="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--host <hostname-or-ip>] [--cap <N|capN>]" >&2
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--cap <N|capN>]" >&2
      exit 1
      ;;
  esac
done

# Normalize cap (allow numeric like 5 -> cap5)
if [[ "$CAP" =~ ^[0-9]+$ ]]; then
  CAP="cap${CAP}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

find_local_csv() {
  local lib_id=$1
  local cap=$2
  local candidates=(
    "repo_export/lib${lib_id}/${cap}/merged_training_all_lib${lib_id}_${cap}_chemclass_rt_prod.csv"
    "repo_export/lib${lib_id}/${cap}/merged_training_*_lib${lib_id}_${cap}_chemclass_rt_prod.csv"
    "repo_export/merged_training_*_lib${lib_id}_${cap}_chemclass_rt_prod.csv"
  )
  for c in "${candidates[@]}"; do
    # shellcheck disable=SC2086
    for f in $c; do
      if [[ -f "$f" ]]; then
        echo "$f"
        return 0
      fi
    done
  done
  return 1
}

DATA208=$(find_local_csv 208 "$CAP") || { echo "Could not find local RT CSV for lib208 ${CAP}" >&2; exit 1; }
DATA209=$(find_local_csv 209 "$CAP") || { echo "Could not find local RT CSV for lib209 ${CAP}" >&2; exit 1; }

EMBED_FILE="resources/metabolites/embeddings_chemberta_pca20.parquet"
if [[ ! -f "$EMBED_FILE" ]]; then
  echo "Embedding artifact not found at $EMBED_FILE; build or copy it locally before uploading." >&2
  exit 1
fi

echo "Uploading RT CSVs for cap=${CAP}:"
echo "  lib208: ${DATA208}"
echo "  lib209: ${DATA209}"
echo "Uploading ChemBERTa embeddings:"
echo "  ${EMBED_FILE}"
echo "Remote root: ${REMOTE_ROOT}"

RSYNC_OPTS=(
  -avh
  --relative
  --partial
)

rsync "${RSYNC_OPTS[@]}" \
  "${DATA208}" \
  "${DATA209}" \
  "${EMBED_FILE}" \
  "${REMOTE_ROOT}"

echo "Upload complete. On the remote host you can run, e.g.:"
echo "  cd ~/MyStorage/CompAssign"
echo "  ./scripts/run_rt_prod.sh --cap ${CAP#cap}"
