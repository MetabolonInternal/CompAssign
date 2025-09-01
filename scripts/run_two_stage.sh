#!/bin/bash

# Two-stage training wrapper for CompAssign
# Allows running stages separately or together

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default parameters (matching run_training.sh)
N_COMPOUNDS=10
N_SPECIES=40
N_SAMPLES=1000
MASS_TOLERANCE=0.005
RT_WINDOW_K=1.5
PROBABILITY_THRESHOLD=0.7
OUTPUT_DIR="two_stage_output"
SEED=42
STAGE="both"
FEATURES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --quick)
            N_SAMPLES=500
            echo -e "${YELLOW}Quick mode: 500 samples${NC}"
            shift
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Two-Stage CompAssign Training"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stage [both|1|2]  Which stage to run (default: both)"
            echo "  --quick             Use 500 samples for quick testing"
            echo "  --features LIST     Comma-separated features to use"
            echo "  --output DIR        Output directory"
            echo "  --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  # Run both stages (default)"
            echo "  $0"
            echo ""
            echo "  # Run stage 1 only (RT model)"
            echo "  $0 --stage 1"
            echo ""
            echo "  # Run stage 2 with saved RT model"
            echo "  $0 --stage 2"
            echo ""
            echo "  # Test with minimal features"
            echo "  $0 --features mass_err_ppm,rt_z,log_intensity --quick"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     CompAssign Two-Stage Training Pipeline   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# Show configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  • Stage: $STAGE"
echo "  • Samples: $N_SAMPLES"
echo "  • Output: $OUTPUT_DIR"
if [ -n "$FEATURES" ]; then
    echo "  • Features: $FEATURES"
fi
echo ""

# Build command
CMD="python scripts/train_two_stage.py"
CMD="$CMD --n-compounds $N_COMPOUNDS"
CMD="$CMD --n-species $N_SPECIES"
CMD="$CMD --n-samples $N_SAMPLES"
CMD="$CMD --mass-tolerance $MASS_TOLERANCE"
CMD="$CMD --rt-window-k $RT_WINDOW_K"
CMD="$CMD --probability-threshold $PROBABILITY_THRESHOLD"
CMD="$CMD --seed $SEED"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --stage $STAGE"

if [ -n "$FEATURES" ]; then
    CMD="$CMD --features $FEATURES"
fi

# Run the command
echo -e "${YELLOW}Starting two-stage training...${NC}"
echo "Command: PYTHONPATH=. $CMD"
echo ""

if PYTHONPATH=. $CMD; then
    echo ""
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    
    # Show next steps based on stage
    if [ "$STAGE" = "1" ]; then
        echo ""
        echo -e "${BLUE}Next steps:${NC}"
        echo "  Run stage 2 with: $0 --stage 2"
    elif [ "$STAGE" = "2" ]; then
        echo ""
        echo -e "${BLUE}Stage 2 complete!${NC}"
        echo "  Check results in: $OUTPUT_DIR/"
    fi
else
    echo -e "${RED}❌ Training failed${NC}"
    exit 1
fi