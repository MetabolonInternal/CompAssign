#!/usr/bin/env bash
set -euo pipefail

echo "scripts/run_simplified_cap20.sh is disabled." >&2
echo "It depended on the removed legacy hierarchical RT model and its eval scripts." >&2
echo "TODO: port this experiment to the Stage1CoeffSummaries-based ridge RT pipeline." >&2
echo "Use: ./scripts/run_rt_prod.sh and ./scripts/run_rt_prod_eval.sh" >&2
exit 2
