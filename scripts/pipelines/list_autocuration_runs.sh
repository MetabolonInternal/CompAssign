#!/usr/bin/env bash

# List recent Pachyderm runs for a given pipeline/project.
# Defaults are tuned for the autocuration_platinum evaluate_models pipeline.

set -euo pipefail

PROJECT="autocuration_platinum"
PIPELINE="evaluate_models"
COUNT=3

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

List the last N Pachyderm runs for a pipeline in a project.

Options:
  -n, --count N        Number of recent runs to show (default: ${COUNT})
  -p, --project NAME   Pachyderm project name (default: ${PROJECT})
      --pipeline NAME  Pachyderm pipeline name (default: ${PIPELINE})
  -h, --help           Show this help message

Examples:
  $0
  $0 -n 5
  $0 --project autocuration_platinum --pipeline evaluate_models -n 10
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--count)
            COUNT="${2:-}"
            shift 2
            ;;
        -p|--project)
            PROJECT="${2:-}"
            shift 2
            ;;
        --pipeline)
            PIPELINE="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if ! command -v pachctl >/dev/null 2>&1; then
    echo "Error: pachctl not found on PATH. Please install/configure Pachyderm CLI." >&2
    exit 1
fi

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -le 0 ]]; then
    echo "Error: count must be a positive integer (got: $COUNT)" >&2
    exit 1
fi

echo "Listing last ${COUNT} runs for pipeline '${PIPELINE}' in project '${PROJECT}'"
echo

# Add 1 line for the header row
LINES=$((COUNT + 1))

pachctl list job \
    --project "${PROJECT}" \
    --pipeline "${PIPELINE}" \
    --full-timestamps \
    --no-color \
    --no-pager \
    | head -n "${LINES}"

