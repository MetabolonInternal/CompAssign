#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/rsync.sh [push|pull] [options]

Sync the CompAssign repo with the remote mirror using rsync.

Arguments:
  push        Copy from local repo to remote.
  pull        Copy from remote to local.

Options:
  --dry-run                 Show what would change without modifying files.
  -o, --out <name>          Output subdir under output/ to sync (repeatable).
                            Example: --out rt_prod_YYYYMMDD_HHMMSS

Environment overrides:
  REMOTE_ROOT   Remote path (default: joewandy@10.34.1.50:~/CompAssign/)
  LOCAL_ROOT    Local path (default: repo root)
EOF
}

ACTION=""
DRY_RUN=0
OUTPUT_SUBDIRS=()

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
    -o|--out|--output-subdir)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a value" >&2
        usage
        exit 1
      fi
      OUTPUT_SUBDIRS+=("$2")
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

if [[ -z "$ACTION" ]]; then
  usage
  exit 1
fi

LOCAL_ROOT=${LOCAL_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
REMOTE_ROOT=${REMOTE_ROOT:-"joewandy@10.34.1.50:~/CompAssign/"}

REMOTE_SPEC="${REMOTE_ROOT%/}"
REMOTE_HOST=""
REMOTE_PATH=""
if [[ "${REMOTE_SPEC}" == *:* ]]; then
  REMOTE_HOST="${REMOTE_SPEC%%:*}"
  REMOTE_PATH="${REMOTE_SPEC#*:}"
fi

remote_dir_exists() {
  local remote_dir="$1"
  if [[ -z "${REMOTE_HOST}" ]]; then
    return 0
  fi
  if ! command -v ssh >/dev/null 2>&1; then
    return 0
  fi
  # NB: we intentionally do not quote ${remote_dir} so `~` expands on the remote.
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_HOST}" "test -d ${remote_dir}" >/dev/null 2>&1
}

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
  --exclude '.DS_Store'
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_OUTPUT_OPTS+=(--dry-run)
fi

if [[ "$ACTION" == "pull" ]]; then
  echo "Pulling from $REMOTE_ROOT to $LOCAL_ROOT"
  rsync "${RSYNC_OPTS[@]}" "$REMOTE_ROOT" "$LOCAL_ROOT/"

  # Pull outputs (default: all of output/).
  if [[ ${#OUTPUT_SUBDIRS[@]} -gt 0 ]]; then
    for subdir in "${OUTPUT_SUBDIRS[@]}"; do
      subdir_norm="${subdir%/}"
      subdir_norm="${subdir_norm#./output/}"
      subdir_norm="${subdir_norm#output/}"
      if [[ -z "${subdir_norm}" ]]; then
        echo "ERROR: --output-subdir must not be empty" >&2
        usage
        exit 1
      fi
      REMOTE_OUT_DIR="${REMOTE_ROOT%/}/output/${subdir_norm}/"
      LOCAL_OUT_DIR="${LOCAL_ROOT%/}/output/${subdir_norm}/"
      echo "Pulling outputs from $REMOTE_OUT_DIR to $LOCAL_OUT_DIR"
      if remote_dir_exists "${REMOTE_PATH%/}/output/${subdir_norm}"; then
        rsync "${RSYNC_OUTPUT_OPTS[@]}" "$REMOTE_OUT_DIR" "$LOCAL_OUT_DIR" || true
      else
        echo "Skipping outputs (missing on remote): ${REMOTE_OUT_DIR}"
      fi
    done
  else
    REMOTE_OUT_DIR="${REMOTE_ROOT%/}/output/"
    LOCAL_OUT_DIR="${LOCAL_ROOT%/}/output/"
    echo "Pulling outputs from $REMOTE_OUT_DIR to $LOCAL_OUT_DIR"
    if remote_dir_exists "${REMOTE_PATH%/}/output"; then
      rsync "${RSYNC_OUTPUT_OPTS[@]}" "$REMOTE_OUT_DIR" "$LOCAL_OUT_DIR" || true
    else
      echo "Skipping outputs (missing on remote): ${REMOTE_OUT_DIR}"
    fi
  fi
else
  if [[ ${#OUTPUT_SUBDIRS[@]} -gt 0 ]]; then
    echo "ERROR: -o/--out is only supported for pull" >&2
    usage
    exit 1
  fi
  echo "Pushing from $LOCAL_ROOT to $REMOTE_ROOT"
  rsync "${RSYNC_OPTS[@]}" "$LOCAL_ROOT/" "$REMOTE_ROOT"
fi
