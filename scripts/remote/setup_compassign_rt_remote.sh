#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER=${REMOTE_USER:-ubuntu}
REMOTE_HOST_DEFAULT="167.234.209.8"
REMOTE_HOST=${REMOTE_HOST:-$REMOTE_HOST_DEFAULT}
REMOTE_DIR=${REMOTE_DIR:-MyStorage}
PROJECT_SUBDIR="CompAssign"
ENV_YML="environment-rt-inference.yml"
ENV_NAME="compassign-rt"

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

REMOTE_ROOT_PATH="${REMOTE_DIR}/${PROJECT_SUBDIR}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" \
  REMOTE_ROOT_PATH="${REMOTE_ROOT_PATH}" \
  ENV_YML="${ENV_YML}" \
  ENV_NAME="${ENV_NAME}" \
  bash -s <<'EOF'
set -euo pipefail

MINICONDA_DIR="$HOME/miniconda3"

cd "$HOME/$REMOTE_ROOT_PATH"

if [ ! -f "$ENV_YML" ]; then
  echo "Environment file $ENV_YML not found in $PWD." >&2
  exit 1
fi

# Ensure conda is available on PATH (reuse existing Miniconda if present, otherwise install)
if command -v conda >/dev/null 2>&1; then
  :
elif [ -x "$MINICONDA_DIR/bin/conda" ]; then
  export PATH="$MINICONDA_DIR/bin:$PATH"
else
  echo "conda is not available on the remote host; installing Miniconda in $MINICONDA_DIR."
  INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  INSTALLER_PATH="$HOME/miniconda3-installer.sh"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$INSTALLER_URL" -o "$INSTALLER_PATH"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$INSTALLER_PATH" "$INSTALLER_URL"
  else
    echo "Neither curl nor wget is available to download Miniconda." >&2
    exit 1
  fi

  bash "$INSTALLER_PATH" -b -p "$MINICONDA_DIR"
  rm -f "$INSTALLER_PATH"

  export PATH="$MINICONDA_DIR/bin:$PATH"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is still not available after attempted Miniconda installation." >&2
  exit 1
fi

# Ensure conda is initialised in future interactive shells.
BASHRC="$HOME/.bashrc"
if [ -f "$BASHRC" ]; then
  if ! grep -q "miniconda3/etc/profile.d/conda.sh" "$BASHRC" >/dev/null 2>&1; then
    {
      echo ""
      echo "# >>> miniconda3 >>>"
      echo "if [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then"
      echo "  . \"\$HOME/miniconda3/etc/profile.d/conda.sh\""
      echo "fi"
      echo "# <<< miniconda3 <<<"
    } >> "$BASHRC"
  fi
fi

echo "Ensuring Conda channel Terms of Service are accepted (if required)."
if conda tos --help >/dev/null 2>&1; then
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
fi

echo "Using environment file: $PWD/$ENV_YML"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists; updating from $ENV_YML."
  conda env update -f "$ENV_YML"
else
  echo "Creating conda environment '$ENV_NAME' from $ENV_YML."
  conda env create -f "$ENV_YML"
fi

echo
echo "To use the environment on the remote host, run:"
echo "  source \"$MINICONDA_DIR/etc/profile.d/conda.sh\""
echo "  conda activate $ENV_NAME"
EOF
