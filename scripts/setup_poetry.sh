#!/bin/bash
#
# setup_poetry.sh - Setup CompAssign using Poetry (no conda required)
#
# This script installs Poetry if needed and sets up the CompAssign environment.
# Works on any system with Python 3.10+
#
# Usage:
#   ./scripts/setup_poetry.sh        # Install Poetry and dependencies
#   ./scripts/setup_poetry.sh check  # Check environment status
#   ./scripts/setup_poetry.sh shell  # Activate poetry shell
#

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print header
print_header() {
    echo -e "${YELLOW}================================================${NC}"
    echo -e "${YELLOW}     CompAssign Poetry Setup${NC}"
    echo -e "${YELLOW}================================================${NC}"
    echo ""
}

# Check if Python is installed and version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Error: Python not found!${NC}"
        echo "Please install Python 3.10 or higher"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${GREEN}✓ Found Python $PYTHON_VERSION${NC}"
    
    # Check if version is adequate
    if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l) -eq 1 ]]; then
        echo -e "${RED}Error: Python 3.10+ required (found $PYTHON_VERSION)${NC}"
        exit 1
    fi
}

# Install Poetry if not present
install_poetry() {
    if command -v poetry &> /dev/null; then
        echo -e "${GREEN}✓ Poetry is already installed${NC}"
        poetry --version
        return 0
    fi
    
    echo -e "${YELLOW}Poetry not found. Installing...${NC}"
    echo ""
    
    # Install Poetry using official installer
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Add to shell profile if not already there
    SHELL_PROFILE=""
    if [ -f "$HOME/.bashrc" ]; then
        SHELL_PROFILE="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_PROFILE="$HOME/.bash_profile"
    elif [ -f "$HOME/.zshrc" ]; then
        SHELL_PROFILE="$HOME/.zshrc"
    fi
    
    if [ -n "$SHELL_PROFILE" ]; then
        if ! grep -q "/.local/bin" "$SHELL_PROFILE"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_PROFILE"
            echo -e "${GREEN}✓ Added Poetry to $SHELL_PROFILE${NC}"
        fi
    fi
    
    # Verify installation
    if command -v poetry &> /dev/null; then
        echo -e "${GREEN}✓ Poetry installed successfully!${NC}"
        poetry --version
    else
        echo -e "${RED}Error: Poetry installation failed${NC}"
        echo "Try manual installation: https://python-poetry.org/docs/#installation"
        exit 1
    fi
}

# Install project dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing CompAssign dependencies...${NC}"
    echo ""
    
    # Configure Poetry to create virtual environment in project
    poetry config virtualenvs.in-project true
    
    # Install dependencies
    echo "This will install:"
    echo "  • PyMC 5.10+ (Bayesian modeling)"
    echo "  • ArviZ (MCMC diagnostics)"
    echo "  • NumPy, Pandas, Matplotlib"
    echo "  • Scikit-learn (for calibration)"
    echo ""
    
    poetry install --no-root
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dependencies installed successfully!${NC}"
        return 0
    else
        echo -e "${RED}✗ Dependency installation failed${NC}"
        return 1
    fi
}

# Check environment status
check_environment() {
    echo -e "${YELLOW}Checking environment status...${NC}"
    echo ""
    
    if [ ! -f "pyproject.toml" ]; then
        echo -e "${RED}✗ pyproject.toml not found${NC}"
        echo "Please run this script from the CompAssign root directory"
        return 1
    fi
    
    if [ -d ".venv" ]; then
        echo -e "${GREEN}✓ Virtual environment exists${NC}"
        
        # Check if activated
        if [[ "$VIRTUAL_ENV" == *".venv"* ]]; then
            echo -e "${GREEN}✓ Virtual environment is activated${NC}"
        else
            echo -e "${YELLOW}○ Virtual environment exists but not activated${NC}"
            echo -e "  To activate: ${BLUE}poetry shell${NC}"
        fi
        
        # Show key packages
        echo ""
        echo -e "${YELLOW}Key packages:${NC}"
        poetry show | grep -E "^(pymc|arviz|pandas|numpy|matplotlib|scikit-learn)" || true
        
    else
        echo -e "${RED}✗ Virtual environment does not exist${NC}"
        echo -e "  To create: ${BLUE}./scripts/setup_poetry.sh${NC}"
    fi
}

# Main execution
main() {
    local action="${1:-install}"
    
    # Change to project root
    cd "$(dirname "$0")/.." 2>/dev/null || true
    
    print_header
    
    case "$action" in
        install|"")
            check_python
            install_poetry
            install_dependencies
            echo ""
            echo -e "${GREEN}================================================${NC}"
            echo -e "${GREEN}✅ Setup Complete!${NC}"
            echo -e "${GREEN}================================================${NC}"
            echo ""
            echo "Next steps:"
            echo -e "  1. Activate environment: ${BLUE}poetry shell${NC}"
            echo -e "  2. Run ablation study: ${BLUE}python scripts/ablation_study.py --quick${NC}"
            echo -e "  3. Or use wrapper: ${BLUE}./scripts/run_ablation_poetry.sh${NC}"
            ;;
            
        check)
            check_environment
            ;;
            
        shell)
            echo -e "${YELLOW}Activating Poetry shell...${NC}"
            poetry shell
            ;;
            
        update)
            echo -e "${YELLOW}Updating dependencies...${NC}"
            poetry update
            echo -e "${GREEN}✓ Dependencies updated${NC}"
            ;;
            
        help|--help|-h)
            echo "Usage: ./scripts/setup_poetry.sh [action]"
            echo ""
            echo "Actions:"
            echo "  install  Install Poetry and dependencies (default)"
            echo "  check    Check environment status"
            echo "  shell    Activate Poetry shell"
            echo "  update   Update all dependencies"
            echo "  help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./scripts/setup_poetry.sh          # Initial setup"
            echo "  ./scripts/setup_poetry.sh check    # Check status"
            echo "  poetry shell                        # Activate environment"
            echo "  poetry run python scripts/train.py  # Run without activating"
            ;;
            
        *)
            echo -e "${RED}Unknown action: $action${NC}"
            echo "Use 'help' to see available actions"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"