#!/bin/bash
#
# CompAssign Model Comparison Runner
# Compares standard vs enhanced models on identical datasets
#

set -e  # Exit on error

# Configuration
N_SAMPLES=${1:-1000}  # Default 1000 samples, can override with first argument
OUTPUT_DIR="output/comparison"
LOG_FILE="comparison.log"

echo "=========================================="
echo "CompAssign Model Comparison"
echo "=========================================="
echo "Samples per chain: $N_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "=========================================="

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

# Clean previous results if they exist
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning previous results..."
    rm -rf "$OUTPUT_DIR"
fi

# Run the comparison in foreground
echo "Starting model comparison (running in foreground)..."
echo "Press Ctrl+C to interrupt if needed"
echo ""

# Use tee to both display output and save to log
PYTHONPATH=. python scripts/compare_models.py \
    --n-samples "$N_SAMPLES" \
    --enhanced-fp-penalty 5.0 \
    --enhanced-mass-tolerance 0.005 \
    --enhanced-high-threshold 0.9 \
    --standard-mass-tolerance 0.01 \
    --standard-threshold 0.5 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

# Check if it completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Comparison completed successfully!"
    echo "Results saved in: $OUTPUT_DIR/"
    echo "Log saved in: $LOG_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Comparison failed or was interrupted"
    echo "Check log file: $LOG_FILE"
    echo "=========================================="
fi