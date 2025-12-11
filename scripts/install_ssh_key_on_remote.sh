#!/usr/bin/env bash

set -euo pipefail

# Copy your local SSH key pair to the remote Ubuntu instance so it can
# authenticate directly with GitHub (git@github.com).
# SECURITY NOTE: this copies a private key to the remote machine.
# Use only on ephemeral hosts you trust.
#
# Usage:
#   ./scripts/install_ssh_key_on_remote.sh [path/to/id_rsa]
#
# Environment overrides:
#   REMOTE_USER   (default: ubuntu)
#   REMOTE_HOST   (default: 167.234.209.8)

REMOTE_USER=${REMOTE_USER:-ubuntu}
REMOTE_HOST_DEFAULT="167.234.209.8"
REMOTE_HOST=${REMOTE_HOST:-$REMOTE_HOST_DEFAULT}

usage() {
  cat <<EOF
Usage: $0 [--host <hostname-or-ip>] [path/to/id_rsa]

Options:
  --host HOST   Remote host/IP (overrides REMOTE_HOST env; default: ${REMOTE_HOST_DEFAULT})

If no key path is provided, defaults to \$HOME/.ssh/id_rsa.
EOF
}

KEY_PATH=""

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
      if [[ -z "$KEY_PATH" ]]; then
        KEY_PATH="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

KEY_PATH="${KEY_PATH:-$HOME/.ssh/id_rsa}"
PUB_PATH="${KEY_PATH}.pub"

if [[ ! -f "$KEY_PATH" ]]; then
  echo "Private key not found at ${KEY_PATH}" >&2
  exit 1
fi

if [[ ! -f "$PUB_PATH" ]]; then
  echo "Public key not found at ${PUB_PATH}" >&2
  exit 1
fi

echo "Copying SSH key pair to ${REMOTE} (~/.ssh)..."

ssh "${REMOTE}" 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'

scp "${KEY_PATH}" "${PUB_PATH}" "${REMOTE}:~/.ssh/"

ssh "${REMOTE}" "chmod 600 ~/.ssh/$(basename "${KEY_PATH}") ~/.ssh/$(basename "${PUB_PATH}")"

echo "Done. On the remote host, GitHub SSH should now work, e.g.:"
echo "  ssh -T git@github.com"
echo "  cd ~/MyStorage/CompAssign && git pull"
