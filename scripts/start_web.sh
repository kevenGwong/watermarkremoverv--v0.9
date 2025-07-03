#!/bin/bash

# Web Application Launcher with Environment Check
# 检查并启动 Web 应用

set -e

ENV_NAME="py310aiwatermark"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              AI Watermark Remover - Web Interface           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}❌ Conda not found${NC}"
        echo "Please install Conda/Miniconda first"
        exit 1
    fi
    echo -e "${GREEN}✅ Conda available${NC}"
}

# Check environment
check_environment() {
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        echo -e "${RED}❌ No conda environment activated${NC}"
        echo -e "${YELLOW}Activating environment: $ENV_NAME${NC}"
        
        # Try to activate environment
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Failed to activate environment: $ENV_NAME${NC}"
            echo "Please check if the environment exists:"
            echo "conda env list"
            exit 1
        fi
    fi
    
    if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
        echo -e "${YELLOW}⚠️  Current environment: $CONDA_DEFAULT_ENV${NC}"
        echo -e "${YELLOW}Expected environment: $ENV_NAME${NC}"
        echo -e "${YELLOW}Switching to correct environment...${NC}"
        
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME
    fi
    
    echo -e "${GREEN}✅ Environment '$CONDA_DEFAULT_ENV' is active${NC}"
}

# Start web application
start_web_app() {
    echo -e "${BLUE}🚀 Starting Web Application...${NC}"
    echo ""
    echo -e "${YELLOW}Available options:${NC}"
    echo -e "  1. Use existing debug app (recommended)"
    echo -e "  2. Use new modular interface"
    echo ""
    
    case "${1:-1}" in
        "1"|"existing"|"debug")
            echo -e "${CYAN}Using existing debug app...${NC}"
            python app.py web --use-existing --port 8501 --host 0.0.0.0
            ;;
        "2"|"new"|"modular")
            echo -e "${CYAN}Using new modular interface...${NC}"
            python app.py web --port 8501 --host 0.0.0.0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [1|existing|debug|2|new|modular]"
            exit 1
            ;;
    esac
}

# Main execution
main() {
    check_conda
    check_environment
    
    echo -e "${YELLOW}Web UI will be available at: http://localhost:8501${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    start_web_app "$1"
}

# Handle script arguments
case "${1:-help}" in
    "start"|"1"|"existing"|"debug")
        main "1"
        ;;
    "new"|"2"|"modular")
        main "2"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  start, 1, existing, debug  - Use existing debug app (default)"
        echo "  new, 2, modular           - Use new modular interface"
        echo "  help, -h, --help          - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0                        # Use existing debug app"
        echo "  $0 existing               # Use existing debug app"
        echo "  $0 new                    # Use new modular interface"
        ;;
    *)
        main "$1"
        ;;
esac