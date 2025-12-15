#!/bin/bash

# Active Learning Assessment Runner
# Reproduces the primary realistic experiment by default

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'


# Defaults (realistic case)
# Default output directory (descriptive)
OUTPUT_DIR="output/al_decoy_heavy_fp_control"
THRESHOLD=0.30
BATCH_SIZE=25
ROUNDS=12
ACQUISITION="hybrid"
LAMBDA_FP=0.95
DIVERSITY_K=100
SEED=42
QUICK=false

# Inference defaults (robust by default)
# Use 4 chains with 1000 tune and 1000 draws for stability and diagnostics
DRAWS=1000
TUNE=1000
CHAINS=4

# Data generator and task knobs
N_SPECIES=40
N_COMPOUNDS=60
N_PEAKS_PER_COMPOUND=5
MASS_ERROR_PPM=12
DECOY_FRACTION=0.9
INITIAL_LABELED=0.0
ISOMER_FRACTION=0.4
NEAR_ISOBAR_FRACTION=0.3
NOISE_MULTIPLIER=2.0
PRESENCE_PROB=0.10

# Candidate matching windows in the assignment model (realistic defaults)
CAND_MASS_PPM=20
CAND_RT_K=2.5

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --output DIR                   Output directory (default: ${OUTPUT_DIR})
  --threshold VALUE              Assignment threshold (default: ${THRESHOLD})
  --batch-size N                 Batch size per AL round (default: ${BATCH_SIZE})
  --rounds N                     Maximum AL rounds (default: ${ROUNDS})
  --acquisition NAME             Acquisition: hybrid|entropy|fp|lc|margin (default: ${ACQUISITION})
  --lambda-fp VALUE              False-positive weight for hybrid (default: ${LAMBDA_FP})
  --diversity-k N                Top-k pool for diverse sampling (default: ${DIVERSITY_K})
  --seed N                       Random seed (default: ${SEED})
  --draws N                      Posterior draws per chain (default: ${DRAWS})
  --tune N                       Tuning steps per chain (default: ${TUNE})
  --chains N                     Number of chains (default: ${CHAINS})
  --n-species N                  Number of species/samples (default: ${N_SPECIES})
  --n-compounds N                Number of compounds in the library (default: ${N_COMPOUNDS})
  --n-peaks-per-compound N       Primary peaks per compound (default: ${N_PEAKS_PER_COMPOUND})
  --mass-error-ppm X             Generator mass error in ppm (default: ${MASS_ERROR_PPM})
  --decoy-fraction X             Fraction of decoy compounds (default: ${DECOY_FRACTION})
  --initial-labeled-fraction X   Seeded label fraction (default: ${INITIAL_LABELED})
  --isomer-fraction X            Fraction of isomers (default: ${ISOMER_FRACTION})
  --near-isobar-fraction X       Fraction of near-isobars (default: ${NEAR_ISOBAR_FRACTION})
  --noise-multiplier X           Scale factor for noise peaks (default: ${NOISE_MULTIPLIER})
  --presence-prob X              Probability a real compound appears per species (default: ${PRESENCE_PROB})
  --cand-mass-ppm X              Candidate matching mass tolerance ppm (default: ${CAND_MASS_PPM})
  --cand-rt-k X                  Candidate matching RT window (SD multiplier, default: ${CAND_RT_K})
  --target-recall-ratio X        Fraction of full-review recall to use as target (default: 0.95)
  --quick                        Use a small 15×15 dataset (smoke test)
  --help                         Show this message
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --acquisition) ACQUISITION="$2"; shift 2 ;;
        --lambda-fp) LAMBDA_FP="$2"; shift 2 ;;
        --diversity-k) DIVERSITY_K="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --draws) DRAWS="$2"; shift 2 ;;
        --tune) TUNE="$2"; shift 2 ;;
        --chains) CHAINS="$2"; shift 2 ;;
        # no --robust flag; robust is the default. Use --quick for a smaller dataset and lighter sampling.
        --n-species) N_SPECIES="$2"; shift 2 ;;
        --n-compounds) N_COMPOUNDS="$2"; shift 2 ;;
        --n-peaks-per-compound) N_PEAKS_PER_COMPOUND="$2"; shift 2 ;;
        --mass-error-ppm) MASS_ERROR_PPM="$2"; shift 2 ;;
        --decoy-fraction) DECOY_FRACTION="$2"; shift 2 ;;
        --initial-labeled-fraction) INITIAL_LABELED="$2"; shift 2 ;;
        --isomer-fraction) ISOMER_FRACTION="$2"; shift 2 ;;
        --near-isobar-fraction) NEAR_ISOBAR_FRACTION="$2"; shift 2 ;;
        --noise-multiplier) NOISE_MULTIPLIER="$2"; shift 2 ;;
        --presence-prob) PRESENCE_PROB="$2"; shift 2 ;;
        --cand-mass-ppm) CAND_MASS_PPM="$2"; shift 2 ;;
        --cand-rt-k) CAND_RT_K="$2"; shift 2 ;;
        --target-recall-ratio) TARGET_RECALL_RATIO="$2"; shift 2 ;;
        --quick) QUICK=true; shift ;;
        --help) usage; exit 0 ;;
        *) echo -e "${YELLOW}Unknown option: $1${NC}" >&2; usage; exit 1 ;;
    esac
done

if [[ "${QUICK}" == "true" ]]; then
    echo -e "${YELLOW}Quick mode (15×15 dataset). Using lighter sampling (4×500×500).${NC}"
    DRAWS=500
    TUNE=500
    CHAINS=4
fi

if [[ ! -f "scripts/experiments/active_learning/assess_active_learning.py" ]]; then
    echo -e "${YELLOW}Cannot find scripts/experiments/active_learning/assess_active_learning.py${NC}" >&2
    exit 1
fi

CMD=("python" "scripts/experiments/active_learning/assess_active_learning.py")
CMD+=("--output-dir" "${OUTPUT_DIR}")
CMD+=("--threshold" "${THRESHOLD}")
CMD+=("--batch-size" "${BATCH_SIZE}")
CMD+=("--rounds" "${ROUNDS}")
CMD+=("--acquisition" "${ACQUISITION}")
CMD+=("--lambda-fp" "${LAMBDA_FP}")
CMD+=("--diversity-k" "${DIVERSITY_K}")
CMD+=("--seed" "${SEED}")
CMD+=("--n-species" "${N_SPECIES}")
CMD+=("--n-compounds" "${N_COMPOUNDS}")
CMD+=("--n-peaks-per-compound" "${N_PEAKS_PER_COMPOUND}")
CMD+=("--draws" "${DRAWS}")
CMD+=("--tune" "${TUNE}")
CMD+=("--chains" "${CHAINS}")
CMD+=("--mass-error-ppm" "${MASS_ERROR_PPM}")
CMD+=("--decoy-fraction" "${DECOY_FRACTION}")
CMD+=("--initial-labeled-fraction" "${INITIAL_LABELED}")
CMD+=("--isomer-fraction" "${ISOMER_FRACTION}")
CMD+=("--near-isobar-fraction" "${NEAR_ISOBAR_FRACTION}")
CMD+=("--noise-multiplier" "${NOISE_MULTIPLIER}")
CMD+=("--presence-prob" "${PRESENCE_PROB}")
CMD+=("--cand-mass-ppm" "${CAND_MASS_PPM}")
CMD+=("--cand-rt-k" "${CAND_RT_K}")
if [[ -n "${TARGET_RECALL_RATIO:-}" ]]; then
  CMD+=("--target-recall-ratio" "${TARGET_RECALL_RATIO}")
fi
if [[ "${QUICK}" == "true" ]]; then CMD+=("--quick"); fi

echo -e "${CYAN}Running active learning assessment${NC}"
echo -e "${YELLOW}Command:${NC} PYTHONPATH=. ${CMD[*]}"
PYTHONPATH=. "${CMD[@]}"

  PLOT_CMD=("python" "-m" "src.compassign.active_learning.plots" "--results-dir" "${OUTPUT_DIR}/results")
PYTHONPATH=. "${PLOT_CMD[@]}"

echo -e "\n${GREEN}✅ Active learning assessment complete${NC}"
