#!/bin/bash
#
# setup_environment.sh - Setup and manage CompAssign conda environment
#
# This script handles environment creation, updates, and activation for CompAssign.
# It works with both Anaconda and Miniconda installations.
#
# Usage:
#   source scripts/setup_environment.sh        # Create and activate environment
#   source scripts/setup_environment.sh update # Update existing environment
#   source scripts/setup_environment.sh check  # Check environment status
#

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Environment name
ENV_NAME="compassign"
ENV_FILE="environment.yml"

# Detect conda installation
detect_conda() {
    if command -v conda &> /dev/null; then
        echo "system"
        return 0
    elif [ -f ~/anaconda3/bin/conda ]; then
        echo ~/anaconda3/bin/conda
        return 0
    elif [ -f ~/miniconda3/bin/conda ]; then
        echo ~/miniconda3/bin/conda
        return 0
    elif [ -f ~/miniforge3/bin/conda ]; then
        echo ~/miniforge3/bin/conda
        return 0
    elif [ -f /opt/anaconda3/bin/conda ]; then
        echo /opt/anaconda3/bin/conda
        return 0
    elif [ -f /opt/miniconda3/bin/conda ]; then
        echo /opt/miniconda3/bin/conda
        return 0
    else
        return 1
    fi
}

# Initialize conda for the current shell
init_conda() {
    local conda_path=$1
    
    if [ "$conda_path" = "system" ]; then
        # Conda is already in PATH
        return 0
    fi
    
    # Get conda base directory
    local conda_dir=$(dirname $(dirname "$conda_path"))
    
    # Source conda initialization
    if [ -f "$conda_dir/etc/profile.d/conda.sh" ]; then
        source "$conda_dir/etc/profile.d/conda.sh"
    else
        echo -e "${RED}Error: Could not find conda initialization script${NC}"
        return 1
    fi
}

# Function to print header
print_header() {
    echo -e "${YELLOW}================================================${NC}"
    echo -e "${YELLOW}     CompAssign Environment Setup${NC}"
    echo -e "${YELLOW}================================================${NC}"
    echo ""
}

# Function to check if environment exists
env_exists() {
    conda env list | grep -q "^${ENV_NAME} "
    return $?
}

# Function to check environment status
check_environment() {
    echo -e "${YELLOW}Checking environment status...${NC}"
    echo ""
    
    if env_exists; then
        echo -e "${GREEN}✓ Environment '${ENV_NAME}' exists${NC}"
        
        # Check if it's currently activated
        if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
            echo -e "${GREEN}✓ Environment is currently activated${NC}"
        else
            echo -e "${YELLOW}○ Environment exists but is not activated${NC}"
            echo -e "  To activate: ${BLUE}conda activate ${ENV_NAME}${NC}"
        fi
        
        # List key packages
        echo ""
        echo -e "${YELLOW}Key packages installed:${NC}"
        conda list -n "$ENV_NAME" | grep -E "^(pymc|arviz|pandas|numpy|matplotlib|scikit-learn)" | head -10
        
        return 0
    else
        echo -e "${RED}✗ Environment '${ENV_NAME}' does not exist${NC}"
        echo -e "  To create: ${BLUE}source scripts/setup_environment.sh${NC}"
        return 1
    fi
}

# Function to create environment
create_environment() {
    echo -e "${YELLOW}Creating new environment '${ENV_NAME}'...${NC}"
    echo ""
    
    # Check if environment.yml exists
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${RED}Error: ${ENV_FILE} not found!${NC}"
        echo "Please run this script from the CompAssign project root directory."
        return 1
    fi
    
    # Show what will be installed
    echo -e "${BLUE}This will install:${NC}"
    echo "  • Python 3.11"
    echo "  • PyMC 5.25+ (Bayesian modeling)"
    echo "  • ArviZ (MCMC diagnostics)"
    echo "  • NumPy, Pandas, Matplotlib"
    echo "  • Scikit-learn (for calibration)"
    echo "  • Jupyter (optional, for notebooks)"
    echo ""
    
    # Ask for confirmation
    read -p "Proceed with installation? (y/n) " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Installation cancelled${NC}"
        return 1
    fi
    
    echo ""
    echo -e "${YELLOW}Installing... This may take 5-10 minutes${NC}"
    
    # Create environment from file
    conda env create -f "$ENV_FILE"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Environment created successfully!${NC}"
        return 0
    else
        echo -e "${RED}✗ Environment creation failed${NC}"
        return 1
    fi
}

# Function to update environment
update_environment() {
    echo -e "${YELLOW}Updating environment '${ENV_NAME}'...${NC}"
    echo ""
    
    if ! env_exists; then
        echo -e "${RED}Error: Environment does not exist${NC}"
        echo -e "Create it first with: ${BLUE}source scripts/setup_environment.sh${NC}"
        return 1
    fi
    
    # Update environment from file
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Environment updated successfully!${NC}"
        return 0
    else
        echo -e "${RED}✗ Environment update failed${NC}"
        return 1
    fi
}

# Function to activate environment
activate_environment() {
    echo -e "${YELLOW}Activating environment '${ENV_NAME}'...${NC}"
    
    conda activate "$ENV_NAME"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Environment activated!${NC}"
        echo ""
        echo -e "${BLUE}Python version:${NC}"
        python --version
        echo ""
        echo -e "${BLUE}PyMC version:${NC}"
        python -c "import pymc; print(f'PyMC {pymc.__version__}')" 2>/dev/null || echo "PyMC not found"
        return 0
    else
        echo -e "${RED}✗ Failed to activate environment${NC}"
        return 1
    fi
}

# Main script logic
main() {
    local action="${1:-create}"
    
    # Change to project root (parent of scripts directory)
    cd "$(dirname "$0")/.." 2>/dev/null || true
    
    print_header
    
    # Detect conda
    echo -e "${YELLOW}Detecting conda installation...${NC}"
    CONDA_PATH=$(detect_conda)
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Conda not found!${NC}"
        echo ""
        echo "Please install Anaconda or Miniconda first:"
        echo "  • Anaconda: https://www.anaconda.com/products/distribution"
        echo "  • Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
    
    echo -e "${GREEN}✓ Found conda at: ${CONDA_PATH}${NC}"
    echo ""
    
    # Initialize conda
    init_conda "$CONDA_PATH"
    
    # Handle different actions
    case "$action" in
        check)
            check_environment
            ;;
        update)
            update_environment
            if [ $? -eq 0 ]; then
                activate_environment
            fi
            ;;
        create|"")
            if env_exists; then
                echo -e "${YELLOW}Environment already exists${NC}"
                check_environment
                echo ""
                read -p "Activate existing environment? (y/n) " -r
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    activate_environment
                fi
            else
                create_environment
                if [ $? -eq 0 ]; then
                    echo ""
                    activate_environment
                fi
            fi
            ;;
        remove)
            echo -e "${YELLOW}Removing environment '${ENV_NAME}'...${NC}"
            read -p "Are you sure? This cannot be undone. (y/n) " -r
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                conda env remove -n "$ENV_NAME"
                echo -e "${GREEN}✓ Environment removed${NC}"
            else
                echo -e "${YELLOW}Removal cancelled${NC}"
            fi
            ;;
        help|--help|-h)
            echo "Usage: source scripts/setup_environment.sh [action]"
            echo ""
            echo "Actions:"
            echo "  (none)   Create and activate environment (default)"
            echo "  check    Check environment status"
            echo "  update   Update existing environment"
            echo "  remove   Remove environment completely"
            echo "  help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  source scripts/setup_environment.sh        # Create/activate"
            echo "  source scripts/setup_environment.sh check  # Check status"
            echo "  source scripts/setup_environment.sh update # Update packages"
            ;;
        *)
            echo -e "${RED}Unknown action: $action${NC}"
            echo "Use 'help' to see available actions"
            return 1
            ;;
    esac
    
    echo ""
    echo -e "${YELLOW}================================================${NC}"
    
    # Show next steps if environment is active
    if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
        echo -e "${GREEN}Ready to use CompAssign!${NC}"
        echo ""
        echo "Next steps:"
        echo "  • Run tests: ${BLUE}PYTHONPATH=. python scripts/train.py --model standard --n-samples 100${NC}"
        echo "  • Run ablation: ${BLUE}./scripts/run_ablation.sh --quick${NC}"
        echo "  • Run verification: ${BLUE}./scripts/run_verification.sh${NC}"
    fi
}

# Check if script is being sourced (required for activation to work)
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo -e "${RED}Error: This script must be sourced, not executed${NC}"
    echo -e "Usage: ${BLUE}source scripts/setup_environment.sh${NC}"
    exit 1
fi

# Run main function
main "$@"