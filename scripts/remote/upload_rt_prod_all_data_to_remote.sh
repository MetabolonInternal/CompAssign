#!/usr/bin/env bash

set -euo pipefail

# Upload RT production training data (across libs and caps),
# plus ChemBERTa embeddings, to the remote CompAssign mirror.
#
# Files uploaded:
#   - repo_export/**/*cap*_chemclass_rt_prod.csv   (training only; realtest is excluded)
#   - resources/metabolites/embeddings_chemberta_pca20.parquet
#
# Environment overrides:
#   REMOTE_USER   (default: ubuntu)
#   REMOTE_ROOT   (default: $REMOTE_USER@$REMOTE_HOST:~/MyStorage/CompAssign/)

REMOTE_USER=${REMOTE_USER:-ubuntu}
REMOTE_HOST=""
REMOTE_ROOT=""
CAP_FILTER=""

usage() {
  cat <<EOF
Usage: $0 --host <hostname-or-ip> [--cap <cap>]

Options:
  --host HOST   Remote host/IP (required; no default is assumed)
  --cap CAP     Restrict uploads to training CSVs for this cap (e.g. 5, 10, 50).
                Only training CSVs are uploaded; realtest CSVs are always skipped.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --host" >&2
        usage
        exit 1
      fi
      REMOTE_HOST="$2"
      shift 2
      ;;
    --cap)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --cap" >&2
        usage
        exit 1
      fi
      CAP_VALUE="$2"
      if [[ "$CAP_VALUE" == cap* ]]; then
        CAP_FILTER="$CAP_VALUE"
      else
        CAP_FILTER="cap${CAP_VALUE}"
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${REMOTE_HOST}" ]]; then
  echo "ERROR: --host is required and no default host is assumed." >&2
  usage
  exit 1
fi

if [[ -z "${REMOTE_ROOT}" ]]; then
  REMOTE_ROOT="${REMOTE_USER}@${REMOTE_HOST}:~/MyStorage/CompAssign/"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EMBED_FILE="resources/metabolites/embeddings_chemberta_pca20.parquet"
if [[ ! -f "$EMBED_FILE" ]]; then
  echo "Embedding artifact not found at $EMBED_FILE; build or copy it locally before uploading." >&2
  exit 1
fi

CSV_LIST_FILE="$(mktemp)"

if [[ -n "${CAP_FILTER}" ]]; then
  # Restrict uploads to a single cap (e.g., cap5, cap50).
  trap 'rm -f "$CSV_LIST_FILE"' EXIT
  find repo_export -type f -name "*_${CAP_FILTER}_chemclass_rt_prod.csv" | sort > "$CSV_LIST_FILE"
else
  # Prioritise uploads: cap5 first, then cap50, then everything else, skipping realtest.
  CAP5_FILES="$(mktemp)"
  CAP50_FILES="$(mktemp)"
  OTHER_FILES="$(mktemp)"
  trap 'rm -f "$CSV_LIST_FILE" "$CAP5_FILES" "$CAP50_FILES" "$OTHER_FILES"' EXIT

  find repo_export -type f -name '*_cap5_chemclass_rt_prod.csv' | sort > "$CAP5_FILES"
  find repo_export -type f -name '*_cap50_chemclass_rt_prod.csv' | sort > "$CAP50_FILES"
  find repo_export -type f -name '*_chemclass_rt_prod.csv' \
    ! -name '*_cap5_chemclass_rt_prod.csv' \
    ! -name '*_cap50_chemclass_rt_prod.csv' \
    ! -name '*realtest*_chemclass_rt_prod.csv' | sort > "$OTHER_FILES"

  cat "$CAP5_FILES" "$CAP50_FILES" "$OTHER_FILES" > "$CSV_LIST_FILE"
fi

if [[ ! -s "$CSV_LIST_FILE" ]]; then
  echo "No matching *_chemclass_rt_prod.csv files found under repo_export; nothing to upload." >&2
  exit 1
fi

echo "Found $(wc -l < "$CSV_LIST_FILE") RT production CSVs to upload."
echo "Example paths:"
head -n 5 "$CSV_LIST_FILE" | sed 's/^/  /'
echo "Embedding file:"
echo "  ${EMBED_FILE}"
echo "Remote root:"
echo "  ${REMOTE_ROOT}"

RSYNC_OPTS=(
  -avh
  --relative
  --partial
)

rsync "${RSYNC_OPTS[@]}" \
  --files-from="$CSV_LIST_FILE" \
  ./ \
  "${REMOTE_ROOT}"

# Upload embedding file separately to ensure it lands under resources/metabolites/.
rsync "${RSYNC_OPTS[@]}" \
  "${EMBED_FILE}" \
  "${REMOTE_ROOT}"

echo "Upload complete."
echo "On the remote host, data should appear under:"
echo "  ~/MyStorage/CompAssign/repo_export/..."
echo "  ~/MyStorage/CompAssign/resources/metabolites/embeddings_chemberta_pca20.parquet"
