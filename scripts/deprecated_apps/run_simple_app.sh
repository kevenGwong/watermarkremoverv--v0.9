#!/bin/bash

# Simple & Reliable Web App Launcher
# 基于原始稳定版本的简化启动器

set -e

ENV_NAME="py310aiwatermark"
APP_FILE="watermark_web_app_simple.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        AI Watermark Remover - Simple & Reliable Edition     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}🎯 Features:${NC}"
echo -e "  ✅ Based on stable original version"
echo -e "  🎯 Clear model selection (Custom vs Florence-2)"
echo -e "  📋 Simple step-by-step workflow"
echo -e "  🔧 No complex parameters that cause errors"
echo -e "  ⚡ Fast and reliable processing"
echo ""

# Check environment
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

# Start app
start_simple_app() {
    echo -e "${BLUE}🚀 Starting Simple & Reliable Web App...${NC}"
    echo -e "${YELLOW}Key advantages:${NC}"
    echo -e "  🎯 Clear model selection interface"
    echo -e "  📋 Step-by-step processing workflow"
    echo -e "  🔧 No problematic advanced parameters"
    echo -e "  ⚡ Fast, stable, and reliable"
    echo ""
    echo -e "${YELLOW}Web UI available at: http://localhost:8505${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    streamlit run $APP_FILE \
        --server.address 0.0.0.0 \
        --server.port 8505 \
        --server.headless true \
        --theme.base dark \
        --theme.primaryColor "#4CAF50" \
        --theme.backgroundColor "#1e1e1e" \
        --theme.secondaryBackgroundColor "#2d2d2d" \
        --theme.textColor="#ffffff"
}

# Test backend
test_backend() {
    echo -e "${BLUE}🧪 Testing backend functionality...${NC}"
    
    python -c "
from web_backend import WatermarkProcessor
try:
    processor = WatermarkProcessor('web_config.yaml')
    print('✅ Backend loaded successfully')
except Exception as e:
    print(f'❌ Backend test failed: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Backend test passed${NC}"
    else
        echo -e "${RED}❌ Backend test failed${NC}"
        exit 1
    fi
}

# Main
case "${1:-start}" in
    "start")
        check_environment
        start_simple_app
        ;;
    "test")
        check_environment
        test_backend
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  start  - Start the simple web app (default)"
        echo "  test   - Test backend functionality"
        echo "  help   - Show this help"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac