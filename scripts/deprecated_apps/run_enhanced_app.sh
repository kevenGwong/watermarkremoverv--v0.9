#!/bin/bash

# Enhanced Web App Launcher with debugging features
# Enhanced version with custom mask upload and transparent debugging

set -e

ENV_NAME="py310aiwatermark"
APP_FILE="watermark_web_app_enhanced.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AI Watermark Remover Enhanced Web App ===${NC}"
echo -e "${YELLOW}🚀 Features: Debug Mode + Custom Mask Upload + Transparent Fix${NC}"

# Function to check environment
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

# Function to run debug test
run_debug_test() {
    echo -e "${BLUE}🔧 Running transparent function debug test...${NC}"
    
    if python debug_transparent_issue.py; then
        echo -e "${GREEN}✅ Debug test completed successfully${NC}"
        echo -e "${YELLOW}Check debug_output/ folder for detailed analysis${NC}"
    else
        echo -e "${RED}❌ Debug test failed${NC}"
        return 1
    fi
}

# Function to start enhanced app
start_enhanced_app() {
    echo -e "${BLUE}🚀 Starting Enhanced Web Application...${NC}"
    echo -e "${YELLOW}Enhanced features:${NC}"
    echo -e "  • 🔧 Debug Mode - visualize masks and processing steps"
    echo -e "  • 🎯 Custom Mask Upload - upload your own masks"
    echo -e "  • 🎨 Transparent Preview - multiple background options"
    echo -e "  • 🎮 Demo Mode - test with generated images"
    echo ""
    echo -e "${YELLOW}Web UI will be available at: http://localhost:8502${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
    echo ""
    
    # Start streamlit app on different port to avoid conflicts
    streamlit run $APP_FILE \
        --server.address 0.0.0.0 \
        --server.port 8502 \
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
    
    # Ask user if they want to run debug test first
    echo -e "${YELLOW}Do you want to run the debug test first? (y/n)${NC}"
    read -r response
    
    if [[ $response =~ ^([yY][eE][sS]|[yY])$ ]]; then
        run_debug_test
        echo ""
        echo -e "${YELLOW}Press Enter to continue to web app...${NC}"
        read -r
    fi
    
    start_enhanced_app
}

# Parse command line arguments
case "${1:-start}" in
    "start")
        main
        ;;
    "debug")
        check_environment
        run_debug_test
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  start  - Start the enhanced web application (default)"
        echo "  debug  - Run debug test only"
        echo "  help   - Show this help"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac