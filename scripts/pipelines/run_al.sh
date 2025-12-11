#!/usr/bin/env bash
set -euo pipefail

echo "scripts/pipelines/run_al.sh is disabled." >&2
echo "It depended on the removed legacy hierarchical RT model." >&2
echo "TODO: port the active-learning pipeline to the Stage1CoeffSummaries-based ridge RT models." >&2
exit 2

