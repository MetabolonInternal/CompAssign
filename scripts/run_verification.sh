#!/bin/bash
#
# CompAssign Model Verification Runner
# Runs all verification tests SEQUENTIALLY for better stability
#

set -e  # Exit on error

echo "=========================================="
echo "CompAssign Model Verification Suite"
echo "Running SEQUENTIALLY for stability"
echo "=========================================="

# Configuration
N_SAMPLES=${1:-1000}  # Can override with first argument
BASE_DIR="output/verification"

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Create directories
mkdir -p "$BASE_DIR"/{standard,enhanced_production,enhanced_ultra,reports}

echo ""
echo "1/3: Training STANDARD model..."
echo "--------------------------------"
PYTHONPATH=. python scripts/train.py \
    --model standard \
    --n-samples "$N_SAMPLES" \
    --n-chains 4 \
    --test-thresholds \
    --output-dir "$BASE_DIR/standard" \
    2>&1 | tee "$BASE_DIR/standard_training.log"

echo ""
echo "2/3: Training ENHANCED PRODUCTION model..."
echo "-------------------------------------------"
PYTHONPATH=. python scripts/train.py \
    --model enhanced \
    --n-samples "$((N_SAMPLES * 2))" \
    --n-chains 4 \
    --test-thresholds \
    --mass-tolerance 0.005 \
    --fp-penalty 5.0 \
    --high-precision-threshold 0.9 \
    --output-dir "$BASE_DIR/enhanced_production" \
    2>&1 | tee "$BASE_DIR/enhanced_production_training.log"

echo ""
echo "3/3: Training ENHANCED ULTRA-HIGH PRECISION model..."
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

echo ""
echo "4/4: Generating benchmark report..."
echo "------------------------------------"
PYTHONPATH=. python scripts/generate_benchmark_report.py

echo ""
echo "=========================================="
echo "✅ VERIFICATION COMPLETE!"
echo "=========================================="
echo "View results:"
echo "  - Report: $BASE_DIR/reports/dashboard.html"
echo "  - Summary: $BASE_DIR/reports/executive_summary.md"
echo "  - Plots: $BASE_DIR/reports/plots/"
echo "=========================================="