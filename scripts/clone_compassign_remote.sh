#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER=${REMOTE_USER:-ubuntu}
REMOTE_HOST_DEFAULT="167.234.209.8"
REMOTE_HOST=${REMOTE_HOST:-$REMOTE_HOST_DEFAULT}
REMOTE_DIR=${REMOTE_DIR:-MyStorage}
REPO_SSH_URL="git@github.com:MetabolonInternal/CompAssign.git"

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

# Ensure an SSH key is loaded locally; default to id_rsa if empty
if ! ssh-add -l >/dev/null 2>&1; then
  DEFAULT_KEY="$HOME/.ssh/id_rsa"
  if [ -f "$DEFAULT_KEY" ]; then
    echo "No SSH identities loaded; adding $DEFAULT_KEY to ssh-agent."
    ssh-add "$DEFAULT_KEY"
  else
    echo "No SSH identities loaded into ssh-agent, and $DEFAULT_KEY not found." >&2
    echo "Add a key with: ssh-add /path/to/your/key" >&2
    exit 1
  fi
fi

REMOTE_ROOT_PATH="${REMOTE_DIR}"

ssh -A -o StrictHostKeyChecking=accept-new "${REMOTE_USER}@${REMOTE_HOST}" \
  REMOTE_ROOT_PATH="${REMOTE_ROOT_PATH}" \
  bash -s <<'EOF'
set -euo pipefail

export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"

cd "$HOME/$REMOTE_ROOT_PATH"

if [ -d CompAssign ]; then
  echo "CompAssign already exists in $PWD; skipping clone."
else
  git clone git@github.com:MetabolonInternal/CompAssign.git
fi
EOF
