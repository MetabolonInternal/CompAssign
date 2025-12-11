#!/usr/bin/env bash
set -euo pipefail

# Run every *.sh under scripts/experiments/rt (recursively).
# For each script, try with --quick first; if that fails, retry without.
# Logs are written to output/rt_all_runs/<script_basename>.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
OUT_DIR="$REPO_ROOT/output/rt_all_runs"
mkdir -p "$OUT_DIR"

# Collect target scripts in a portable way (macOS Bash 3 compatible)
SH_SCRIPTS=$(find "$REPO_ROOT/scripts/experiments/rt" -type f -name "*.sh" | sort)

declare -a OK
declare -a FAIL

NUM_SCRIPTS=0
if [[ -n "${SH_SCRIPTS:-}" ]]; then
  NUM_SCRIPTS=$(printf "%s\n" "$SH_SCRIPTS" | sed -e '/^$/d' | wc -l | tr -d ' ')
fi
echo "[run_all_rt_experiments] Found ${NUM_SCRIPTS} scripts under scripts/experiments/rt"

OLDIFS=$IFS
IFS=$'\n'
for shpath in $SH_SCRIPTS; do
  # Skip this runner itself if it’s within the tree
  if [[ "$(basename "$shpath")" == "run_all_rt_experiments.sh" ]]; then
    continue
  fi

  bname="$(basename "$shpath")"
  log="$OUT_DIR/${bname}.log"

  printf "\n>>> Running %s with --quick (logging to %s)\n" "$bname" "$log"
  set +e
  bash "$shpath" --quick >"$log" 2>&1
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "    --quick failed (exit=$status); retrying without --quick"
    bash "$shpath" >>"$log" 2>&1
    status=$?
  fi
  set -e

  if [[ $status -eq 0 ]]; then
    echo "    OK: $bname"
    OK+=("$bname")
  else
    echo "    FAIL: $bname (exit=$status). See $log"
    FAIL+=("$bname")
  fi
done
IFS=$OLDIFS

printf "\n[run_all_rt_experiments] Summary\n"
echo "  OK   : ${#OK[@]}"
for s in "${OK[@]:-}"; do echo "    - $s"; done
echo "  FAIL : ${#FAIL[@]}"
for s in "${FAIL[@]:-}"; do echo "    - $s"; done

if [[ ${#FAIL[@]} -gt 0 ]]; then
  echo "One or more scripts failed. Inspect logs under $OUT_DIR"
  exit 1
fi

echo "All scripts completed successfully. Logs at $OUT_DIR"
