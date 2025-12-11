#!/usr/bin/env bash

set -euo pipefail

# Download RT production results from a remote CompAssign mirror back to this repo.
#
# By default this syncs the entire output/rt_prod directory so that
# subsequent local or remote runs can skip existing traces.
#
# Usage:
#   ./scripts/download_rt_prod_results_from_remote.sh [--host <hostname-or-ip>]
#
# Environment overrides:
#   REMOTE_USER   (default: ubuntu)
#   REMOTE_HOST   (default: 167.234.209.8)
#   REMOTE_ROOT   (default: $REMOTE_USER@$REMOTE_HOST:~/MyStorage/CompAssign/)

REMOTE_USER=${REMOTE_USER:-ubuntu}
REMOTE_HOST_DEFAULT="167.234.209.8"
REMOTE_HOST=${REMOTE_HOST:-$REMOTE_HOST_DEFAULT}
REMOTE_ROOT=${REMOTE_ROOT:-"${REMOTE_USER}@${REMOTE_HOST}:~/MyStorage/CompAssign/"}

usage() {
  cat <<EOF
Usage: $0 [--host <hostname-or-ip>]

Options:
  --host HOST   Remote host/IP (overrides REMOTE_HOST env; default: ${REMOTE_HOST_DEFAULT})
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
      REMOTE_ROOT="${REMOTE_USER}@${REMOTE_HOST}:~/MyStorage/CompAssign/"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOCAL_RT_DIR="output/rt_prod"
REMOTE_RT_DIR="${REMOTE_ROOT}output/rt_prod/"

echo "Syncing RT production results from remote:"
echo "  ${REMOTE_RT_DIR}"
echo "into local:"
echo "  ${REPO_ROOT}/${LOCAL_RT_DIR}"

mkdir -p "${LOCAL_RT_DIR}"

RSYNC_OPTS=(
  -avh
  --partial
)

rsync "${RSYNC_OPTS[@]}" \
  "${REMOTE_RT_DIR}" \
  "${LOCAL_RT_DIR}/"

echo "Download complete. Local RT production results are under:"
echo "  ${REPO_ROOT}/${LOCAL_RT_DIR}"
