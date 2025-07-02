#!/bin/bash

# Web App Launcher for AI Watermark Remover
# Designed to run in conda py310aiwatermark environment

set -e

ENV_NAME="py310aiwatermark"
CONFIG_FILE="web_config.yaml"
APP_FILE="watermark_web_app.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AI Watermark Remover Web App ===${NC}"

# Function to check if conda environment exists and is activated
check_environment() {
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        echo -e "${RED}❌ No conda environment activated${NC}"
        echo -e "${YELLOW}Please activate the environment first:${NC}"
        echo "conda activate $ENV_NAME"
        exit 1
    fi
    
    if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
        echo -e "${YELLOW}⚠️  Current environment: $CONDA_DEFAULT_ENV${NC}"
        echo -e "${YELLOW}Expected environment: $ENV_NAME${NC}"
        echo -e "${YELLOW}Please activate the correct environment:${NC}"
        echo "conda activate $ENV_NAME"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Environment '$ENV_NAME' is active${NC}"
}

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}🔍 Checking dependencies...${NC}"
    
    # Check if streamlit is installed
    if ! python -c "import streamlit" 2>/dev/null; then
        echo -e "${RED}❌ Streamlit not found${NC}"
        echo -e "${YELLOW}Installing web dependencies...${NC}"
        pip install -r requirements_web.txt
    fi
    
    # Check if key modules can be imported
    if ! python -c "import torch, transformers, iopaint" 2>/dev/null; then
        echo -e "${RED}❌ Missing key dependencies${NC}"
        echo -e "${YELLOW}Please ensure all requirements are installed:${NC}"
        echo "pip install -r requirements_web.txt"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Dependencies verified${NC}"
}

# Function to check model files
check_models() {
    echo -e "${BLUE}🔍 Checking model files...${NC}"
    
    # Check custom model
    CUSTOM_MODEL="/home/duolaameng/SAM_Remove/Watermark_sam/output/checkpoints/epoch=071-valid_iou=0.7267.ckpt"
    if [ ! -f "$CUSTOM_MODEL" ]; then
        echo -e "${RED}❌ Custom model not found: $CUSTOM_MODEL${NC}"
        echo -e "${YELLOW}Please ensure the custom model checkpoint exists${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Custom model found${NC}"
    
    # Check if LaMA model is downloaded
    echo -e "${YELLOW}ℹ️  LaMA model will be downloaded automatically on first use${NC}"
}

# Function to create necessary directories
setup_directories() {
    echo -e "${BLUE}📁 Setting up directories...${NC}"
    
    mkdir -p temp
    mkdir -p output
    
    echo -e "${GREEN}✅ Directories ready${NC}"
}

# Function to start the web app
start_app() {
    echo -e "${BLUE}🚀 Starting web application...${NC}"
    echo -e "${YELLOW}Web UI will be available at: http://localhost:8501${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
    echo ""
    
    # Start streamlit app
    streamlit run $APP_FILE \
        --server.address 0.0.0.0 \
        --server.port 8501 \
        --server.headless true \
        --theme.base dark \
        --theme.primaryColor "#ff6b6b" \
        --theme.backgroundColor "#1e1e1e" \
        --theme.secondaryBackgroundColor "#2d2d2d" \
        --theme.textColor "#ffffff"
}

# Main execution
main() {
    check_environment
    check_dependencies
    check_models
    setup_directories
    start_app
}

# Parse command line arguments
case "${1:-start}" in
    "start")
        main
        ;;
    "check")
        check_environment
        check_dependencies
        check_models
        echo -e "${GREEN}✅ All checks passed - ready to start!${NC}"
        ;;
    "deps")
        check_environment
        echo -e "${YELLOW}Installing dependencies...${NC}"
        pip install -r requirements_web.txt
        echo -e "${GREEN}✅ Dependencies installed${NC}"
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  start  - Start the web application (default)"
        echo "  check  - Check environment and dependencies"
        echo "  deps   - Install dependencies"
        echo "  help   - Show this help"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac