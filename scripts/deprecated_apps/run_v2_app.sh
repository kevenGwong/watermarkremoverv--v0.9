#!/bin/bash

# Web App v2 Launcher - Clear Workflow Edition
# 清晰工作流程版本启动器

set -e

ENV_NAME="py310aiwatermark"
APP_FILE="watermark_web_app_v2.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          AI Watermark Remover v2 - Clear Workflow Edition   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}🎯 Key Improvements:${NC}"
echo -e "  ✅ Fixed dpm_solver_pp sampler error"
echo -e "  🎯 Clear model selection (Custom vs Florence-2)"
echo -e "  📋 Step-by-step workflow display"
echo -e "  💾 Custom settings save/load"
echo -e "  💡 Integrated parameter tooltips"
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
start_v2_app() {
    echo -e "${BLUE}🚀 Starting v2 Web Application...${NC}"
    echo -e "${YELLOW}New features in v2:${NC}"
    echo -e "  🎯 Clear model selection interface"
    echo -e "  📋 Step-by-step processing workflow"
    echo -e "  💾 Custom parameter presets"
    echo -e "  🔧 Fixed all known issues"
    echo ""
    echo -e "${YELLOW}Web UI available at: http://localhost:8504${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    streamlit run $APP_FILE \
        --server.address 0.0.0.0 \
        --server.port 8504 \
        --server.headless true \
        --theme.base dark \
        --theme.primaryColor "#4CAF50" \
        --theme.backgroundColor "#1e1e1e" \
        --theme.secondaryBackgroundColor "#2d2d2d" \
        --theme.textColor="#ffffff"
}

# Main
check_environment
start_v2_app