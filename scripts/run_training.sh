#!/bin/bash

# CompAssign Training Script
# Wrapper for train.py with commonly used parameters

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
N_COMPOUNDS=10
N_SPECIES=40
N_SAMPLES=1000
MASS_TOLERANCE=0.005
RT_WINDOW_K=1.5
PROBABILITY_THRESHOLD=0.7
MATCHING="greedy"
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
        --full)
            echo -e "${YELLOW}Running in full mode (2000 samples)${NC}"
            N_SAMPLES=2000
            shift
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
            echo "  --full            Run with 2000 samples (slower, more accurate)"
            echo "  --compounds N     Number of compounds (default: 10)"
            echo "  --species N       Number of species (default: 40)"
            echo "  --output DIR      Output directory (default: output)"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                # Run with defaults"
            echo "  $0 --quick        # Quick test run"
            echo "  $0 --compounds 20 --species 100  # Larger dataset"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to print section headers
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}       CompAssign Training Pipeline         ${NC}"
    echo -e "${BLUE}============================================${NC}"
}

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
    
    # Print header
    print_header
    
    echo ""
    echo "Configuration:"
    echo "  • Compounds: ${N_COMPOUNDS}"
    echo "  • Species: ${N_SPECIES}"
    echo "  • Samples: ${N_SAMPLES}"
    echo "  • Mass tolerance: ${MASS_TOLERANCE} Da"
    echo "  • RT window: ±${RT_WINDOW_K}σ"
    echo "  • Matching: ${MATCHING}"
    echo "  • Probability threshold: ${PROBABILITY_THRESHOLD}"
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo ""
    
    # Check if script exists
    check_script
    
    # Build the command
    CMD="python scripts/train.py"
    CMD="${CMD} --n-compounds ${N_COMPOUNDS}"
    CMD="${CMD} --n-species ${N_SPECIES}"
    CMD="${CMD} --n-samples ${N_SAMPLES}"
    CMD="${CMD} --mass-tolerance ${MASS_TOLERANCE}"
    CMD="${CMD} --rt-window-k ${RT_WINDOW_K}"
    CMD="${CMD} --probability-threshold ${PROBABILITY_THRESHOLD}"
    CMD="${CMD} --matching ${MATCHING}"
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
        echo ""
        echo -e "${GREEN}============================================${NC}"
        
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