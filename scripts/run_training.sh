#!/bin/bash
#
# CompAssign Training Runner
# Runs training with recommended parameters (99.5% precision)
#
# Usage:
#   ./scripts/run_training.sh              # Run with default 1000 samples
#   ./scripts/run_training.sh 500          # Run with custom sample count
#   ./scripts/run_training.sh 1000 0.8     # Custom samples and threshold
#   ./scripts/run_training.sh --quick      # Quick test with 100 samples
#   ./scripts/run_training.sh --test       # Test thresholds
#

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values (recommended based on testing)
N_SAMPLES=1000
N_COMPOUNDS=10     # Fast testing (use 60 for realistic evaluation)
N_SPECIES=40       # 40 species across 8 clusters (~5 per cluster)
MASS_TOL=0.005
RT_WINDOW=1.5
MATCHING="greedy"  # Greedy performs better than Hungarian
THRESHOLD=0.5      # Balanced precision-recall (70% precision, 43% recall)
TARGET_ACCEPT=0.95  # Standard target accept
TEST_THRESHOLDS=false

# Parse arguments
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "CompAssign Training Runner"
    echo ""
    echo "Usage:"
    echo "  ./scripts/run_training.sh              # Standard training (10 compounds, 1000 samples)"
    echo "  ./scripts/run_training.sh 500          # Custom sample count"
    echo "  ./scripts/run_training.sh 1000 0.8     # Custom samples and threshold"
    echo "  ./scripts/run_training.sh --quick      # Quick test (100 samples)"
    echo "  ./scripts/run_training.sh --compounds 30  # Custom compound count"
    echo "  ./scripts/run_training.sh --species 80    # Custom species count"
    echo "  ./scripts/run_training.sh --test       # Test multiple thresholds"
    echo "  ./scripts/run_training.sh --hungarian  # Use Hungarian matching (optimal 1-to-1)"
    echo "  ./scripts/run_training.sh --no-matching # No one-to-one constraint (high recall)"
    echo ""
    echo "Default parameters (recommended):"
    echo "  • Compounds: 10 (fast testing, use --compounds 60 for realistic)"
    echo "  • Species: 40 (across 8 clusters, ~5 per cluster)"
    echo "  • Mass tolerance: 0.005 Da"
    echo "  • RT window: ±1.5σ"
    echo "  • Matching: Greedy (better performance than Hungarian)"
    echo "  • Probability threshold: 0.5 (balanced precision-recall)"
    echo ""
    echo "Examples:"
    echo "  # Production training"
    echo "  ./scripts/run_training.sh"
    echo ""
    echo "  # Quick development test"
    echo "  ./scripts/run_training.sh --quick"
    echo ""
    echo "  # Explore precision-recall tradeoff"
    echo "  ./scripts/run_training.sh --test"
    echo ""
    echo "  # Realistic evaluation (60 compounds)"
    echo "  ./scripts/run_training.sh --compounds 60"
    echo ""
    echo "  # Custom threshold for more recall"
    echo "  ./scripts/run_training.sh 1000 0.8"
    exit 0
fi

# Quick mode
if [[ "$1" == "--quick" ]]; then
    N_SAMPLES=100
    echo -e "${YELLOW}Quick test mode: Using only 100 samples${NC}"
    shift
fi

# Custom compound count
if [[ "$1" == "--compounds" ]]; then
    shift
    if [[ -n "$1" ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
        N_COMPOUNDS=$1
        echo -e "${YELLOW}Using $N_COMPOUNDS compounds${NC}"
        shift
    else
        echo -e "${YELLOW}Warning: Invalid compound count, using default (10)${NC}"
    fi
fi

# Custom species count
if [[ "$1" == "--species" ]]; then
    shift
    if [[ -n "$1" ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
        N_SPECIES=$1
        echo -e "${YELLOW}Using $N_SPECIES species${NC}"
        shift
    else
        echo -e "${YELLOW}Warning: Invalid species count, using default (40)${NC}"
    fi
fi

# Test thresholds mode
if [[ "$1" == "--test" ]]; then
    TEST_THRESHOLDS=true
    echo -e "${YELLOW}Testing multiple thresholds for precision-recall analysis${NC}"
    shift
fi

# Greedy matching mode (already default, but kept for clarity)
if [[ "$1" == "--greedy" ]]; then
    MATCHING="greedy"
    echo -e "${YELLOW}Using greedy matching algorithm${NC}"
    shift
fi

# Hungarian matching mode (optimal one-to-one assignment)
if [[ "$1" == "--hungarian" ]]; then
    MATCHING="hungarian"
    echo -e "${YELLOW}Using Hungarian matching algorithm (optimal one-to-one)${NC}"
    shift
fi

# None matching mode (no one-to-one constraint)
if [[ "$1" == "--no-matching" ]]; then
    MATCHING="none"
    echo -e "${YELLOW}Using no matching constraint (allows multiple assignments per compound)${NC}"
    shift
fi

# Target accept parameter
if [[ "$1" == "--target-accept" ]]; then
    shift
    if [[ -n "$1" ]] && [[ "$1" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        TARGET_ACCEPT=$1
        echo -e "${YELLOW}Using target_accept=$TARGET_ACCEPT for MCMC${NC}"
        shift
    else
        echo -e "${YELLOW}Warning: Invalid target-accept value, using default${NC}"
    fi
fi

# Custom sample count
if [[ -n "$1" ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
    N_SAMPLES=$1
    shift
fi

# Custom threshold
if [[ -n "$1" ]] && [[ "$1" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    THRESHOLD=$1
    echo -e "${YELLOW}⚠️  Using custom threshold: $THRESHOLD (default: 0.5)${NC}"
    shift
fi

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Create output directories
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"/{data,models,results,plots}

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}       CompAssign Training Pipeline         ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Configuration:"
echo "  • Compounds: $N_COMPOUNDS"
echo "  • Species: $N_SPECIES"
echo "  • Samples: $N_SAMPLES"
echo "  • Mass tolerance: $MASS_TOL Da"
echo "  • RT window: ±${RT_WINDOW}σ"
echo "  • Matching: $MATCHING"
echo "  • Probability threshold: $THRESHOLD"

echo ""
echo -e "${BLUE}============================================${NC}"
echo ""

# Build the training command
CMD="PYTHONPATH=. python scripts/train.py"
CMD="$CMD --n-compounds $N_COMPOUNDS"
CMD="$CMD --n-species $N_SPECIES"
CMD="$CMD --n-samples $N_SAMPLES"
CMD="$CMD --mass-tolerance $MASS_TOL"
CMD="$CMD --rt-window-k $RT_WINDOW"
CMD="$CMD --matching $MATCHING"
CMD="$CMD --probability-threshold $THRESHOLD"
CMD="$CMD --target-accept $TARGET_ACCEPT"

if [[ "$TEST_THRESHOLDS" == true ]]; then
    CMD="$CMD --test-thresholds"
fi

CMD="$CMD --output-dir $OUTPUT_DIR"

# Run training
echo -e "${YELLOW}Starting training...${NC}"
echo "Command: $CMD"
echo ""

# Execute with timing
start_time=$(date +%s)

# Run the command and capture exit code
if eval $CMD; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo -e "${GREEN}✅ Training completed in $((duration / 60))m $((duration % 60))s${NC}"
    echo ""
    # The Python script already printed detailed results above
    
    if [[ "$TEST_THRESHOLDS" == true ]] && [[ -f "$OUTPUT_DIR/results/threshold_analysis.csv" ]]; then
        echo ""
        echo "Threshold analysis saved to: $OUTPUT_DIR/results/threshold_analysis.csv"
    fi
    
    echo -e "${GREEN}============================================${NC}"
else
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo -e "\033[0;31m============================================${NC}"
    echo -e "\033[0;31m❌ Training failed${NC}"
    echo -e "\033[0;31m============================================${NC}"
    echo ""
    echo "Duration before failure: $((duration / 60)) minutes $((duration % 60)) seconds"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if conda environment is activated: conda activate compassign"
    echo "  2. Verify all dependencies are installed: conda env update -f environment.yml"
    echo "  3. Check error messages above for specific issues"
    echo ""
    echo "For quick testing, try: ./scripts/run_training.sh --quick"
    exit 1
fi