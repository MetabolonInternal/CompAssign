#!/bin/bash

# CompAssign Training Script
# Trains the hierarchical Bayesian model with minimal features

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Default parameters
N_COMPOUNDS=30  # Increased for more overlaps
N_SPECIES=40
N_SAMPLES=1000  # Normal mode draws per chain
N_TUNE=1000     # Normal mode tuning steps per chain
N_CHAINS=4
MASS_TOLERANCE_PPM=25.0  # Instrument-level mass tolerance (ppm)
RT_WINDOW_K=2.0  # Wider window for harder data
PROBABILITY_THRESHOLD=0.5  # Lower threshold for more realistic tradeoff
TARGET_ACCEPT=0.99  # Use 0.99 in all cases
OUTPUT_DIR="output"
SEED=42
MASS_ERROR_PPM=5.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            echo -e "${YELLOW}Running in quick mode (500 samples/tune, 15 compounds/species)${NC}"
            N_SAMPLES=500
            N_TUNE=500
            N_COMPOUNDS=15
            N_SPECIES=15
            N_CHAINS=4
            TARGET_ACCEPT=0.99
            MASS_TOLERANCE_PPM=25.0
            shift
            ;;
        --samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --tune)
            N_TUNE="$2"
            shift 2
            ;;
        --compounds)
            N_COMPOUNDS="$2"
            shift 2
            ;;
        --species)
            N_SPECIES="$2"
            shift 2
            ;;
        --chains)
            N_CHAINS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick           Run small sanity check (150 samples, 15x15 dataset)"
            echo "  --samples N       Number of MCMC samples (default: 1000)"
            echo "  --tune N          Number of tuning steps (default: 1000)"
            echo "  --compounds N     Number of compounds (default: 30)"
            echo "  --species N       Number of species (default: 40)"
            echo "  --output DIR      Output directory (default: output)"
            echo "  --help            Show this help message"
            echo ""
            echo "Model:"
            echo "  Hierarchical Bayesian with enhanced minimal features"
            echo "  Core: mass_err_ppm, rt_z, log_intensity, log_rt_uncertainty"
            echo "  Context: has_isotope, isotope_score, n_adducts, rt_cluster_size,"
            echo "           n_correlated plus adduct indicators"
            echo ""
            echo "Examples:"
            echo "  $0                        # Run with default parameters (challenging data)"
            echo "  $0 --quick                # Quick test with small dataset"
            echo "  $0 --compounds 30 --species 50  # Even harder dataset"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to print section headers (removed - no longer needed)

# Function to check if script exists
check_script() {
    if [ ! -f "scripts/pipelines/train.py" ]; then
        echo -e "${RED}Error: pipelines/train.py not found!${NC}"
        echo "Please ensure you're running from the CompAssign root directory"
        exit 1
    fi
}

# Function to format time
format_time() {
    local seconds=$1
    local minutes=$((seconds / 60))
    local remaining_seconds=$((seconds % 60))
    echo "${minutes} minutes ${remaining_seconds} seconds"
}

# Main execution
main() {
    # Record start time
    START_TIME=$(date +%s)
    
    # Print configuration concisely
    echo -e "${CYAN}CompAssign Training${NC}"
    echo "Compounds: ${N_COMPOUNDS}, Species: ${N_SPECIES}, Samples: ${N_SAMPLES}"
    echo ""
    
    # Check if script exists
    check_script
    
    CMD="python scripts/pipelines/train.py"
    CMD="${CMD} --n-compounds ${N_COMPOUNDS}"
    CMD="${CMD} --n-species ${N_SPECIES}"
    CMD="${CMD} --n-samples ${N_SAMPLES}"
    CMD="${CMD} --n-tune ${N_TUNE}"
    CMD="${CMD} --n-chains ${N_CHAINS}"
    CMD="${CMD} --mass-tolerance-ppm ${MASS_TOLERANCE_PPM}"
    CMD="${CMD} --mass-error-ppm ${MASS_ERROR_PPM}"
    CMD="${CMD} --rt-window-k ${RT_WINDOW_K}"
    CMD="${CMD} --probability-threshold ${PROBABILITY_THRESHOLD}"
    CMD="${CMD} --target-accept ${TARGET_ACCEPT}"
    CMD="${CMD} --output-dir ${OUTPUT_DIR}"
    CMD="${CMD} --seed ${SEED}"
    
    # Print command for reproducibility
    echo -e "${YELLOW}Starting training...${NC}"
    echo "Command: PYTHONPATH=. ${CMD}"
    echo ""
    
    # Run the training with PYTHONPATH set
    if PYTHONPATH=. $CMD; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        echo ""
        echo -e "${GREEN}✅ Training completed in $(format_time $DURATION)${NC}"
        
    else
        # Preserve Python's stderr/stdout and exit silently with failure code
        exit 1
    fi
}

# Run main entrypoint
main
