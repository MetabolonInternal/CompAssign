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
N_SAMPLES=1000  # Default to 1000 samples for reasonable runtime
MASS_TOLERANCE=0.01  # Increased tolerance for harder data
RT_WINDOW_K=2.0  # Wider window for harder data
PROBABILITY_THRESHOLD=0.5  # Lower threshold for more realistic tradeoff
TARGET_ACCEPT=0.95
OUTPUT_DIR="output"
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            echo -e "${YELLOW}Running in quick mode (500 samples)${NC}"
            N_SAMPLES=500
            shift
            ;;
        --samples)
            N_SAMPLES="$2"
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
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick           Run with 500 samples (faster)"
            echo "  --samples N       Number of MCMC samples (default: 1000)"
            echo "  --compounds N     Number of compounds (default: 20)"
            echo "  --species N       Number of species (default: 40)"
            echo "  --output DIR      Output directory (default: output)"
            echo "  --help            Show this help message"
            echo ""
            echo "Model:"
            echo "  Hierarchical Bayesian with 4 minimal features"
            echo "  Features: mass_err_ppm, rt_z, log_intensity, log_rt_uncertainty"
            echo ""
            echo "Examples:"
            echo "  $0                        # Run with default parameters (challenging data)"
            echo "  $0 --quick                # Quick test with 500 samples"
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
    if [ ! -f "scripts/train.py" ]; then
        echo -e "${RED}Error: train.py not found!${NC}"
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
    
    CMD="python scripts/train.py"
    CMD="${CMD} --n-compounds ${N_COMPOUNDS}"
    CMD="${CMD} --n-species ${N_SPECIES}"
    CMD="${CMD} --n-samples ${N_SAMPLES}"
    CMD="${CMD} --mass-tolerance ${MASS_TOLERANCE}"
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
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo ""
        echo -e "${RED}============================================${NC}"
        echo -e "${RED}❌ Training failed${NC}"
        echo -e "${RED}============================================${NC}"
        echo ""
        echo "Duration before failure: $(format_time $DURATION)"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Check if conda environment is activated: conda activate compassign"
        echo "  2. Verify all dependencies are installed: conda env update -f environment.yml"
        echo "  3. Check error messages above for specific issues"
        echo ""
        echo "For quick testing, try: $0 --quick"
        exit 1
    fi
}

# Run main function
main
