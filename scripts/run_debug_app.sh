#!/bin/bash

# Debug Edition Web App Launcher
# 调试版本启动器 - 参数控制和实时对比

set -e

ENV_NAME="py310aiwatermark"
APP_FILE="watermark_web_app_debug.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        AI Watermark Remover - Debug Edition                 ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}🔬 Debug Features:${NC}"
echo -e "  📊 Left panel: Complete parameter control"
echo -e "  🔄 Right panel: Interactive before/after comparison"
echo -e "  🎯 Mask model selection: Custom vs Florence-2"
echo -e "  ⚙️ Full inpainting parameter control"
echo -e "  ⚡ Performance options and monitoring"
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

# Check streamlit-image-comparison
check_dependencies() {
    echo -e "${BLUE}🔍 Checking debug dependencies...${NC}"
    
    if ! python -c "import streamlit_image_comparison" 2>/dev/null; then
        echo -e "${YELLOW}📦 Installing streamlit-image-comparison...${NC}"
        pip install streamlit-image-comparison
    fi
    
    echo -e "${GREEN}✅ Dependencies ready${NC}"
}

# Test debug backend
test_debug_backend() {
    echo -e "${BLUE}🧪 Testing debug backend functionality...${NC}"
    
    python -c "
from web_backend import WatermarkProcessor
try:
    processor = WatermarkProcessor('web_config.yaml')
    print('✅ Debug backend loaded successfully')
    
    # Test enhanced functionality
    import streamlit_image_comparison
    print('✅ Image comparison component available')
    
except Exception as e:
    print(f'❌ Debug backend test failed: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Debug backend test passed${NC}"
    else
        echo -e "${RED}❌ Debug backend test failed${NC}"
        exit 1
    fi
}

# Start debug app
start_debug_app() {
    echo -e "${BLUE}🚀 Starting Debug Edition Web App...${NC}"
    echo -e "${YELLOW}Debug capabilities:${NC}"
    echo -e "  🎯 Real-time parameter adjustment"
    echo -e "  🔄 Interactive before/after comparison"
    echo -e "  📊 Detailed processing metrics"
    echo -e "  🎭 Mask visualization and analysis"
    echo -e "  💾 Debug-optimized downloads"
    echo ""
    echo -e "${YELLOW}Web UI available at: http://localhost:8506${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    streamlit run $APP_FILE \
        --server.address 0.0.0.0 \
        --server.port 8506 \
        --server.headless true \
        --theme.base dark \
        --theme.primaryColor "#FF6B35" \
        --theme.backgroundColor "#1e1e1e" \
        --theme.secondaryBackgroundColor "#2d2d2d" \
        --theme.textColor="#ffffff"
}

# Main
case "${1:-start}" in
    "start")
        check_environment
        check_dependencies
        start_debug_app
        ;;
    "test")
        check_environment
        check_dependencies
        test_debug_backend
        ;;
    "deps")
        check_environment
        echo -e "${YELLOW}📦 Installing debug dependencies...${NC}"
        pip install streamlit-image-comparison
        echo -e "${GREEN}✅ Debug dependencies installed${NC}"
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  start  - Start the debug web app (default)"
        echo "  test   - Test debug backend functionality"
        echo "  deps   - Install debug dependencies"
        echo "  help   - Show this help"
        echo ""
        echo "Debug Features:"
        echo "  🔬 Parameter Control Panel"
        echo "  🔄 Interactive Image Comparison"
        echo "  🎯 Real-time Mask Visualization"
        echo "  📊 Performance Monitoring"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac