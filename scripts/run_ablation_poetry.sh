#!/bin/bash
#
# run_ablation_poetry.sh - Run ablation study using Poetry
#
# This is a Poetry-compatible version of run_ablation.sh
#

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default configuration
N_SAMPLES=1000
QUICK_MODE=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="output/ablation"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            N_SAMPLES=500
            shift
            ;;
        --samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --help|-h)
            echo "CompAssign Ablation Study Runner (Poetry Version)"
            echo ""
            echo "Usage:"
            echo "  $0              # Full study with 1000 samples"
            echo "  $0 --quick      # Quick test with 500 samples, 4 configs"
            echo "  $0 --samples N  # Use N samples per configuration"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set output directory
if [ "$QUICK_MODE" = true ]; then
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_quick_${TIMESTAMP}"
else
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_full_${TIMESTAMP}"
fi

# Ensure we're in the project root
cd "$(dirname "$0")/.."

echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}     CompAssign Ablation Study (Poetry)${NC}"
echo -e "${YELLOW}================================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Mode: ${YELLOW}$([ "$QUICK_MODE" = true ] && echo "QUICK (4 configs)" || echo "FULL (12 configs)")${NC}"
echo -e "  Samples per config: ${BLUE}${N_SAMPLES}${NC}"
echo -e "  Output directory: ${BLUE}${OUTPUT_DIR}${NC}"
echo ""

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: Poetry not found!${NC}"
    echo -e "Please run: ${BLUE}./scripts/setup_poetry.sh${NC}"
    exit 1
fi

# Check if environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Poetry environment not found!${NC}"
    echo -e "Please run: ${BLUE}./scripts/setup_poetry.sh${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/ablation.log"

# Estimate runtime
if [ "$QUICK_MODE" = true ]; then
    ESTIMATED_TIME="10-15 minutes"
    N_CONFIGS=4
else
    ESTIMATED_TIME="40-50 minutes"
    N_CONFIGS=12
fi

echo -e "${YELLOW}Starting Ablation Study${NC}"
echo "Estimated runtime: ${BLUE}${ESTIMATED_TIME}${NC}"
echo "Configurations to test: ${BLUE}${N_CONFIGS}${NC}"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the ablation study using Poetry
echo -e "${YELLOW}Running ablation study...${NC}"
echo ""

if [ "$QUICK_MODE" = true ]; then
    poetry run python scripts/ablation_study.py \
        --quick \
        --n-samples "$N_SAMPLES" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"
else
    poetry run python scripts/ablation_study.py \
        --n-samples "$N_SAMPLES" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"
fi

# Check if successful
if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}✅ ABLATION STUDY COMPLETE!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "Runtime: ${BLUE}${MINUTES}m ${SECONDS}s${NC}"
    echo ""
    echo "Results saved to:"
    echo "  • Table: ${BLUE}${OUTPUT_DIR}/ablation_results.csv${NC}"
    echo "  • Plot: ${BLUE}${OUTPUT_DIR}/ablation_results.png${NC}"
    echo "  • Summary: ${BLUE}${OUTPUT_DIR}/ablation_summary.json${NC}"
    echo ""
    
    # Extract key findings
    echo -e "${YELLOW}Key Findings:${NC}"
    if grep -q "Minimal configuration for >95% precision" "$LOG_FILE"; then
        BEST_CONFIG=$(grep "Minimal configuration for >95% precision" "$LOG_FILE" | tail -1 | cut -d: -f2)
        echo -e "${GREEN}✓ Target achieved with:${BEST_CONFIG}${NC}"
    else
        echo -e "${RED}✗ No configuration achieved >95% precision${NC}"
    fi
    
else
    echo -e "${RED}❌ ABLATION STUDY FAILED${NC}"
    echo "Check the log file: ${BLUE}${LOG_FILE}${NC}"
    exit 1
fi