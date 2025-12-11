#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/rsync_compassign.sh [push|pull] [--dry-run]

Sync the CompAssign repo with the remote mirror using rsync.
On pull, also sync RT production outputs under output/rt_prod.

Arguments:
  push        Copy from local repo to remote.
  pull        Copy from remote to local.
  --dry-run   Show what would change without modifying files.

Environment overrides:
  REMOTE_ROOT   Remote path (default: joewandy@10.34.1.50:~/CompAssign/)
  LOCAL_ROOT    Local path (default: repo root)
EOF
}

ACTION=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    push|pull)
      ACTION="$1"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ -z "$ACTION" ]]; then
  usage
  exit 1
fi

LOCAL_ROOT=${LOCAL_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
REMOTE_ROOT=${REMOTE_ROOT:-"joewandy@10.34.1.50:~/CompAssign/"}

RSYNC_OPTS=(
  -avh
  --update
  --exclude '.git/'
  --exclude 'output/'
  --exclude '.mypy_cache/'
  --exclude '.pytest_cache/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
  --exclude 'external_repos/'
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

RSYNC_OUTPUT_OPTS=(
  -avh
  --update
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_OUTPUT_OPTS+=(--dry-run)
fi

if [[ "$ACTION" == "pull" ]]; then
  echo "Pulling from $REMOTE_ROOT to $LOCAL_ROOT"
  rsync "${RSYNC_OPTS[@]}" "$REMOTE_ROOT" "$LOCAL_ROOT/"
  # Also pull RT production outputs (models/results) for inspection.
  REMOTE_RT_DIR="${REMOTE_ROOT%/}/output/rt_prod/"
  LOCAL_RT_DIR="${LOCAL_ROOT%/}/output/rt_prod/"
  echo "Pulling RT outputs from $REMOTE_RT_DIR to $LOCAL_RT_DIR"
  rsync "${RSYNC_OUTPUT_OPTS[@]}" "$REMOTE_RT_DIR" "$LOCAL_RT_DIR" || true
else
  echo "Pushing from $LOCAL_ROOT to $REMOTE_ROOT"
  rsync "${RSYNC_OPTS[@]}" "$LOCAL_ROOT/" "$REMOTE_ROOT"
fi
