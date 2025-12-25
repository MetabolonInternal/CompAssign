#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Regenerate RT multilevel plots/ summaries for a run directory.

By default, uses the run directory pointed to by:
  output/rt_prod_latest.txt

Usage:
  ./scripts/plot_rt_multilevel.sh [options]

Options:
  --run-dir <path>   Run directory under output/ (default: latest pointer).
  --out-dir <path>   Output directory (default: <run-dir>/plots).
  --cap <capN|N>     Cap label (default: cap100).
  --libs <ids>       Comma-separated lib ids (default: infer from run-dir/lib*).
  --dry-run          Print the command and exit.
  -h, --help         Show this help text.

This generates BOTH plot sets into the same output directory using filename tags:
  - tag=full       (candidates + lasso baselines)
  - tag=candidates (candidates only)
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_DIR=""
OUT_DIR=""
CAP="cap100"
LIBS=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --cap)
      CAP="$2"
      shift 2
      ;;
    --libs)
      LIBS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$CAP" =~ ^[0-9]+$ ]]; then
  CAP="cap${CAP}"
fi

if [[ -z "$RUN_DIR" ]]; then
  LATEST_PATH="${REPO_ROOT}/output/rt_prod_latest.txt"
  if [[ ! -f "${LATEST_PATH}" ]]; then
    echo "ERROR: Missing latest pointer: ${LATEST_PATH}" >&2
    echo "Pass --run-dir explicitly, or create the pointer file." >&2
    exit 2
  fi
  RUN_DIR="$(cat "${LATEST_PATH}" | tr -d '[:space:]')"
fi
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${REPO_ROOT}/${RUN_DIR}"
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: Run directory does not exist: ${RUN_DIR}" >&2
  exit 2
fi

CANDIDATE_MODELS=(
  "pymc_collapsed_group_species_cluster@none"
  "sklearn_ridge_species_cluster"
  "pymc_pooled_species_comp_hier_supercat_cluster_supercat@none"
)
FULL_MODELS=("${CANDIDATE_MODELS[@]}" "lasso_eslasso_species_cluster" "lasso_eslasso_local")

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${RUN_DIR}/plots"
fi

if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${REPO_ROOT}/${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}"

echo "[plot] RUN_DIR=${RUN_DIR}"
echo "[plot] OUT_DIR=${OUT_DIR}"
echo "[plot] CAP=${CAP}"
if [[ -n "${LIBS}" ]]; then
  echo "[plot] LIBS=${LIBS}"
fi

mk_cmd() {
  local tag="$1"
  local models_csv="$2"

  local -a cmd=(
    poetry run python -u scripts/pipelines/plot_rt_multilevel_results.py
    --run-dir "${RUN_DIR}"
    --out-dir "${OUT_DIR}"
    --cap "${CAP}"
    --anchor none
    --models "${models_csv}"
    --tag "${tag}"
  )
  if [[ -n "${LIBS}" ]]; then
    cmd+=(--libs "${LIBS}")
  fi
  printf "%q " "${cmd[@]}"
}

FULL_MODELS_CSV="$(IFS=,; echo "${FULL_MODELS[*]}")"
CANDIDATE_MODELS_CSV="$(IFS=,; echo "${CANDIDATE_MODELS[*]}")"

echo "[plot] FULL_MODELS=${FULL_MODELS_CSV}"
echo "[plot] CANDIDATE_MODELS=${CANDIDATE_MODELS_CSV}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[plot] DRY_RUN CMD (full): $(mk_cmd full "${FULL_MODELS_CSV}")"
  echo "[plot] DRY_RUN CMD (candidates): $(mk_cmd candidates "${CANDIDATE_MODELS_CSV}")"
  exit 0
fi

cd "${REPO_ROOT}"

CMD_FULL=(
  poetry run python -u scripts/pipelines/plot_rt_multilevel_results.py
  --run-dir "${RUN_DIR}"
  --out-dir "${OUT_DIR}"
  --cap "${CAP}"
  --anchor none
  --models "${FULL_MODELS_CSV}"
  --tag full
)
if [[ -n "${LIBS}" ]]; then
  CMD_FULL+=(--libs "${LIBS}")
fi
"${CMD_FULL[@]}"

CMD_CANDIDATES=(
  poetry run python -u scripts/pipelines/plot_rt_multilevel_results.py
  --run-dir "${RUN_DIR}"
  --out-dir "${OUT_DIR}"
  --cap "${CAP}"
  --anchor none
  --models "${CANDIDATE_MODELS_CSV}"
  --tag candidates
)
if [[ -n "${LIBS}" ]]; then
  CMD_CANDIDATES+=(--libs "${LIBS}")
fi
"${CMD_CANDIDATES[@]}"
