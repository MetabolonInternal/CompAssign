#!/bin/bash
#
# CompAssign Model Verification Runner
# Runs all verification tests SEQUENTIALLY for better stability
#
# Usage:
#   ./scripts/run_verification.sh              # Run all models (1000 samples)
#   ./scripts/run_verification.sh 100          # Run with custom sample count
#   ./scripts/run_verification.sh --report-only   # Generate report only
#   ./scripts/run_verification.sh --step N     # Run from step N (1-4)
#

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for special modes
if [[ "$1" == "--report-only" ]]; then
    echo -e "${YELLOW}===========================================${NC}"
    echo "Generating Benchmark Report Only"
    echo -e "${YELLOW}===========================================${NC}"
    cd "$(dirname "$0")/.."
    PYTHONPATH=. python scripts/generate_benchmark_report.py
    echo -e "${GREEN}✅ Report generation complete!${NC}"
    exit 0
fi

# Check for step mode
START_STEP=1
if [[ "$1" == "--step" ]]; then
    START_STEP=$2
    N_SAMPLES=${3:-1000}
    echo -e "${YELLOW}Starting from step $START_STEP with $N_SAMPLES samples${NC}"
else
    N_SAMPLES=${1:-1000}  # Can override with first argument
fi

echo -e "${YELLOW}===========================================${NC}"
echo "CompAssign Model Verification Suite"
echo "Running SEQUENTIALLY for stability"
echo -e "${YELLOW}===========================================${NC}"

# Configuration
BASE_DIR="output/verification"

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Create directories
mkdir -p "$BASE_DIR"/{standard,enhanced_production,enhanced_ultra,reports}

# Step 1: Standard Model
if [ $START_STEP -le 1 ]; then
    echo ""
    echo -e "${YELLOW}1/4: Training STANDARD model...${NC}"
    echo "--------------------------------"
    PYTHONPATH=. python scripts/train.py \
        --model standard \
        --n-samples "$N_SAMPLES" \
        --n-chains 4 \
        --output-dir "$BASE_DIR/standard" \
        2>&1 | tee "$BASE_DIR/standard_training.log"
    echo -e "${GREEN}✅ Standard model complete${NC}"
fi

# Step 2: Enhanced Production Model
if [ $START_STEP -le 2 ]; then
    echo ""
    echo -e "${YELLOW}2/4: Training ENHANCED PRODUCTION model...${NC}"
    echo "-------------------------------------------"
    PYTHONPATH=. python scripts/train.py \
        --model enhanced \
        --n-samples "$N_SAMPLES" \
        --n-chains 4 \
        --test-thresholds \
        --mass-tolerance 0.005 \
        --fp-penalty 5.0 \
        --high-precision-threshold 0.9 \
        --output-dir "$BASE_DIR/enhanced_production" \
        2>&1 | tee "$BASE_DIR/enhanced_production_training.log"
    echo -e "${GREEN}✅ Enhanced production model complete${NC}"
fi

# Step 3: Enhanced Ultra-High Precision Model
if [ $START_STEP -le 3 ]; then
    echo ""
    echo -e "${YELLOW}3/4: Training ENHANCED ULTRA-HIGH PRECISION model...${NC}"
    echo "-----------------------------------------------------"
    PYTHONPATH=. python scripts/train.py \
        --model enhanced \
        --n-samples "$((N_SAMPLES * 2))" \
        --n-chains 4 \
        --test-thresholds \
        --mass-tolerance 0.003 \
        --fp-penalty 10.0 \
        --high-precision-threshold 0.95 \
        --output-dir "$BASE_DIR/enhanced_ultra" \
        2>&1 | tee "$BASE_DIR/enhanced_ultra_training.log"
    echo -e "${GREEN}✅ Enhanced ultra-high precision model complete${NC}"
fi

# Step 4: Generate Report
if [ $START_STEP -le 4 ]; then
    echo ""
    echo -e "${YELLOW}4/4: Generating benchmark report...${NC}"
    echo "------------------------------------"
    PYTHONPATH=. python scripts/generate_benchmark_report.py
    echo -e "${GREEN}✅ Report generation complete${NC}"
fi

echo ""
echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}✅ VERIFICATION COMPLETE!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo "View results:"
echo "  - Report: $BASE_DIR/reports/dashboard.html"
echo "  - Summary: $BASE_DIR/reports/executive_summary.md"
echo "  - Plots: $BASE_DIR/reports/plots/"
echo -e "${GREEN}===========================================${NC}"

# Help message
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo ""
    echo "Usage:"
    echo "  ./scripts/run_verification.sh              # Run all steps"
    echo "  ./scripts/run_verification.sh 100          # Run with 100 samples"
    echo "  ./scripts/run_verification.sh --report-only   # Generate report only"
    echo "  ./scripts/run_verification.sh --step 4     # Run from step 4 (report only)"
    echo "  ./scripts/run_verification.sh --step 2 500 # Run from step 2 with 500 samples"
    echo ""
    echo "Steps:"
    echo "  1. Standard model training"
    echo "  2. Enhanced production model training"
    echo "  3. Enhanced ultra-high precision model training"
    echo "  4. Generate benchmark report"
fi