#!/bin/bash
#
# clear_verification.sh - Clean up all output files before running verification
#
# This script removes all training outputs and temporary files to ensure
# a clean slate for testing the CompAssign training pipeline.
#
# Usage:
#   ./scripts/clear_verification.sh           # Interactive mode (asks for confirmation)
#   ./scripts/clear_verification.sh --force   # Force mode (no confirmation)

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}CompAssign Output Cleanup${NC}"
echo "========================================="

# Check for force flag
FORCE_MODE=false
if [[ "$1" == "--force" ]] || [[ "$1" == "-f" ]]; then
    FORCE_MODE=true
fi

# List what will be deleted
echo -e "\nThe following will be ${RED}DELETED${NC}:"
echo "  • output/ directory (all training outputs)"
echo "  • training_debug.txt"
echo "  • training_*.log files"
echo "  • test_*.py temporary test scripts"
echo "  • Any .pyc and __pycache__ directories"

# Count existing files
OUTPUT_COUNT=$(find output -type f 2>/dev/null | wc -l || echo "0")
LOG_COUNT=$(ls -1 training_*.log 2>/dev/null | wc -l || echo "0")
TEST_COUNT=$(ls -1 test_*.py 2>/dev/null | wc -l || echo "0")

echo -e "\nCurrent file counts:"
echo "  • Output files: $OUTPUT_COUNT"
echo "  • Log files: $LOG_COUNT"
echo "  • Test scripts: $TEST_COUNT"

# Ask for confirmation unless in force mode
if [ "$FORCE_MODE" = false ]; then
    echo -e "\n${YELLOW}Are you sure you want to delete all outputs?${NC}"
    read -p "Type 'yes' to confirm: " -r
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        echo -e "${RED}Cleanup cancelled.${NC}"
        exit 1
    fi
fi

echo -e "\n${YELLOW}Cleaning up...${NC}"

# Remove output directory
if [ -d "output" ]; then
    echo "  Removing output/ directory..."
    rm -rf output/
    echo -e "  ${GREEN}✓${NC} output/ removed"
else
    echo "  • output/ directory not found (skipping)"
fi

# Remove debug and log files
echo "  Removing log files..."
rm -f training_debug.txt
rm -f training_*.log
rm -f *.log
echo -e "  ${GREEN}✓${NC} Log files removed"

# Remove temporary test scripts
echo "  Removing temporary test scripts..."
rm -f test_*.py
echo -e "  ${GREEN}✓${NC} Test scripts removed"

# Remove Python cache files
echo "  Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Python cache removed"

# Create fresh output directory structure
echo -e "\n${YELLOW}Creating fresh output directory structure...${NC}"
mkdir -p output
echo -e "  ${GREEN}✓${NC} output/ directory created"

echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}Cleanup complete!${NC} Ready for fresh verification run."
echo ""
echo "Next steps:"
echo "  1. Run quick test:      ./scripts/run_verification.sh --quick"
echo "  2. Run full suite:      ./scripts/run_verification.sh"
echo "  3. Run specific model:  PYTHONPATH=. python scripts/train.py --model standard"
echo ""