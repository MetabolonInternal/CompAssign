#!/usr/bin/env bash
set -euo pipefail

echo "scripts/pipelines/run_training.sh is disabled." >&2
echo "It depended on the removed legacy hierarchical RT model via scripts/pipelines/train.py." >&2
echo "TODO: port the synthetic end-to-end training pipeline to use ridge RT predictions." >&2
exit 2

