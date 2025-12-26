#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
  exec poetry run python -u scripts/run_rt_report_all.py --skip-existing --build-pdf
fi

exec poetry run python -u scripts/run_rt_report_all.py "$@"
