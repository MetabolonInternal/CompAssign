#!/bin/bash
#
# run_ablation.sh - Run CompAssign Ablation Study
#
# This script runs systematic tests to isolate the impact of each model
# component on precision and recall.
#
# Usage:
#   ./scripts/run_ablation.sh              # Full study (1000 samples)
#   ./scripts/run_ablation.sh --quick      # Quick test (500 samples, 4 configs)
#   ./scripts/run_ablation.sh --samples N  # Custom sample count
#   ./scripts/run_ablation.sh --resume     # Resume from last checkpoint
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
RESUME_MODE=false
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
        --resume)
            RESUME_MODE=true
            shift
            ;;
        --help|-h)
            echo "CompAssign Ablation Study Runner"
            echo ""
            echo "Usage:"
            echo "  $0              # Full study with 1000 samples"
            echo "  $0 --quick      # Quick test with 500 samples, 4 configs"
            echo "  $0 --samples N  # Use N samples per configuration"
            echo "  $0 --resume     # Resume from last run (not implemented yet)"
            echo ""
            echo "The study tests 12 configurations (4 in quick mode) to isolate:"
            echo "  • Impact of threshold changes (0.5 → 0.9)"
            echo "  • Impact of mass tolerance (0.01 → 0.005 Da)"
            echo "  • Impact of absolute value features"
            echo "  • Impact of FP penalty (1.0, 3.0, 5.0)"
            echo "  • Impact of RT uncertainty features"
            echo ""
            echo "Results are saved to: output/ablation_TIMESTAMP/"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
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
echo -e "${YELLOW}     CompAssign Ablation Study Runner${NC}"
echo -e "${YELLOW}================================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Mode: ${YELLOW}$([ "$QUICK_MODE" = true ] && echo "QUICK (4 configs)" || echo "FULL (12 configs)")${NC}"
echo -e "  Samples per config: ${BLUE}${N_SAMPLES}${NC}"
echo -e "  Output directory: ${BLUE}${OUTPUT_DIR}${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file setup
LOG_FILE="$OUTPUT_DIR/ablation.log"
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"

# Function to print and log
log_message() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Check environment
log_message "${YELLOW}Checking environment...${NC}"
if ! conda env list | grep -q "compassign"; then
    log_message "${RED}ERROR: compassign environment not found!${NC}"
    log_message "Please create it with: conda env create -f environment.yml"
    exit 1
fi

# Activate environment
log_message "${GREEN}✓ Environment found${NC}"
log_message "${YELLOW}Activating compassign environment...${NC}"
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate compassign

# Estimate runtime
if [ "$QUICK_MODE" = true ]; then
    ESTIMATED_TIME="10-15 minutes"
    N_CONFIGS=4
else
    ESTIMATED_TIME="40-50 minutes"
    N_CONFIGS=12
fi

log_message ""
log_message "${YELLOW}================================================${NC}"
log_message "${YELLOW}Starting Ablation Study${NC}"
log_message "${YELLOW}================================================${NC}"
log_message "Estimated runtime: ${BLUE}${ESTIMATED_TIME}${NC}"
log_message "Configurations to test: ${BLUE}${N_CONFIGS}${NC}"
log_message ""
log_message "The study will test:"
if [ "$QUICK_MODE" = true ]; then
    log_message "  1. S-Base: Standard baseline"
    log_message "  2. S-Threshold: Only threshold change (0.5→0.9)"
    log_message "  3. S-Both: Threshold + mass tolerance"
    log_message "  4. E-Weight5: Full enhanced model"
else
    log_message "  Phase 1: Baseline isolation (4 configs)"
    log_message "  Phase 2: Feature engineering (2 configs)"
    log_message "  Phase 3: FP penalty impact (3 configs)"
    log_message "  Phase 4: Feature ablation (3 configs)"
fi
log_message ""
log_message "${YELLOW}Progress will be shown below...${NC}"
log_message "${YELLOW}================================================${NC}"
log_message ""

# Record start time
START_TIME=$(date +%s)

# Run the ablation study
if [ "$QUICK_MODE" = true ]; then
    PYTHONPATH=. python scripts/ablation_study.py \
        --quick \
        --n-samples "$N_SAMPLES" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"
else
    PYTHONPATH=. python scripts/ablation_study.py \
        --n-samples "$N_SAMPLES" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"
fi

# Check if successful
if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    log_message ""
    log_message "${GREEN}================================================${NC}"
    log_message "${GREEN}✅ ABLATION STUDY COMPLETE!${NC}"
    log_message "${GREEN}================================================${NC}"
    log_message ""
    log_message "Runtime: ${BLUE}${MINUTES}m ${SECONDS}s${NC}"
    log_message ""
    log_message "Results saved to:"
    log_message "  • Table: ${BLUE}${OUTPUT_DIR}/ablation_results.csv${NC}"
    log_message "  • Plot: ${BLUE}${OUTPUT_DIR}/ablation_results.png${NC}"
    log_message "  • Summary: ${BLUE}${OUTPUT_DIR}/ablation_summary.json${NC}"
    log_message "  • Log: ${BLUE}${OUTPUT_DIR}/ablation.log${NC}"
    log_message ""
    
    # Extract key findings
    log_message "${YELLOW}Key Findings:${NC}"
    
    # Check if any configuration achieved >95% precision
    if grep -q "Minimal configuration for >95% precision" "$LOG_FILE"; then
        BEST_CONFIG=$(grep "Minimal configuration for >95% precision" "$LOG_FILE" | tail -1 | cut -d: -f2)
        log_message "${GREEN}✓ Target achieved with:${BEST_CONFIG}${NC}"
    else
        log_message "${RED}✗ No configuration achieved >95% precision${NC}"
    fi
    
    # Show top 3 configurations by precision
    log_message ""
    log_message "${YELLOW}Top 3 configurations by precision:${NC}"
    if [ -f "$OUTPUT_DIR/ablation_results.csv" ]; then
        # Skip header and sort by precision column
        tail -n +2 "$OUTPUT_DIR/ablation_results.csv" | \
            sort -t',' -k3 -r | \
            head -3 | \
            awk -F',' '{printf "  • %-15s Precision: %s Recall: %s FP: %s\n", $1, $3, $4, $6}' | \
            tee -a "$SUMMARY_FILE"
    fi
    
    log_message ""
    log_message "${YELLOW}Next Steps:${NC}"
    log_message "1. Review the CSV table for detailed comparisons"
    log_message "2. Check the visualization plot for trends"
    log_message "3. Use the minimal >95% precision config for production"
    
else
    log_message ""
    log_message "${RED}================================================${NC}"
    log_message "${RED}❌ ABLATION STUDY FAILED${NC}"
    log_message "${RED}================================================${NC}"
    log_message ""
    log_message "Check the log file for errors: ${BLUE}${LOG_FILE}${NC}"
    exit 1
fi

# Generate quick summary report
echo "================================================" > "$SUMMARY_FILE"
echo "CompAssign Ablation Study Summary" >> "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Configuration:" >> "$SUMMARY_FILE"
echo "  Mode: $([ "$QUICK_MODE" = true ] && echo "QUICK" || echo "FULL")" >> "$SUMMARY_FILE"
echo "  Samples: $N_SAMPLES" >> "$SUMMARY_FILE"
echo "  Runtime: ${MINUTES}m ${SECONDS}s" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

if [ -f "$OUTPUT_DIR/ablation_results.csv" ]; then
    echo "Results Summary:" >> "$SUMMARY_FILE"
    echo "----------------" >> "$SUMMARY_FILE"
    cat "$OUTPUT_DIR/ablation_results.csv" | column -t -s',' >> "$SUMMARY_FILE"
fi

log_message ""
log_message "${GREEN}Summary report saved to: ${BLUE}${SUMMARY_FILE}${NC}"
log_message "${GREEN}================================================${NC}"