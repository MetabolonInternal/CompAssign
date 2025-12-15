#!/usr/bin/env bash
set -euo pipefail

# Legacy wrapper retained for compatibility.
#
# The production RT pipeline has moved to ridge coefficient-summary models and is now
# driven by the repo-level entrypoints:
#   - ./scripts/run_rt_prod.sh
#   - ./scripts/run_rt_prod_eval.sh
#
# This wrapper simply forwards to ./scripts/run_rt_prod.sh.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

bash "${REPO_ROOT}/scripts/run_rt_prod.sh" "$@"
