#!/bin/bash

# Professional Web App Launcher with Full Parameter Control
# 专业版Web应用启动器 - 包含所有高级参数控制

set -e

ENV_NAME="py310aiwatermark"
APP_FILE="watermark_web_app_pro.py"
CONFIG_FILE="web_config_advanced.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          AI Watermark Remover Pro - Professional Edition     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}🎯 Advanced Features:${NC}"
echo -e "  🔧 Full Parameter Control (30+ settings)"
echo -e "  🎨 Custom Prompts & Detection Settings"
echo -e "  ⚡ Performance Presets (Fast/Balanced/Quality/Ultra)"
echo -e "  📊 Real-time Memory & GPU Monitoring"
echo -e "  🔍 Debug Mode with Intermediate Results"
echo -e "  💾 Advanced Download Options"
echo ""

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

# Function to check advanced config
check_advanced_config() {
    echo -e "${BLUE}🔍 Checking advanced configuration...${NC}"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}❌ Advanced config file not found: $CONFIG_FILE${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Advanced configuration found${NC}"
    
    # Show config summary
    echo -e "${YELLOW}📋 Configuration Summary:${NC}"
    echo "  - Mask Generator: $(grep 'model_type:' $CONFIG_FILE | cut -d':' -f2 | tr -d ' \"')"
    echo "  - LDM Steps: $(grep 'ldm_steps:' $CONFIG_FILE | head -1 | cut -d':' -f2 | tr -d ' ')"
    echo "  - HD Strategy: $(grep 'hd_strategy:' $CONFIG_FILE | head -1 | cut -d':' -f2 | tr -d ' \"')"
    echo "  - Parameter Presets: $(grep -A1 'parameter_presets:' $CONFIG_FILE | grep -v 'parameter_presets:' | wc -l) available"
    return 0
}

# Function to show parameter guide
show_parameter_guide() {
    echo -e "${CYAN}📚 Quick Parameter Guide:${NC}"
    echo ""
    echo -e "${YELLOW}🎯 Mask Generation:${NC}"
    echo "  • Mask Threshold (0.0-1.0): Controls detection sensitivity"
    echo "  • Dilate Kernel Size (1-15): Expands detected regions"
    echo "  • Custom Prompts: 'watermark', 'logo', 'text overlay', etc."
    echo ""
    echo -e "${YELLOW}🎨 LaMA Inpainting:${NC}"
    echo "  • LDM Steps (10-200): More steps = higher quality"
    echo "  • Sampler: ddim (stable), plms (fast), dpm_solver++ (quality)"
    echo "  • HD Strategy: CROP (detailed), RESIZE (fast), ORIGINAL (exact)"
    echo ""
    echo -e "${YELLOW}⚡ Performance Presets:${NC}"
    echo "  • Fast: Quick processing (20 steps, 1024px)"
    echo "  • Balanced: Recommended (50 steps, 1600px)" 
    echo "  • Quality: High quality (100 steps, 2048px)"
    echo "  • Ultra: Maximum quality (200 steps, 4096px)"
    echo ""
}

# Function to start professional app
start_pro_app() {
    echo -e "${BLUE}🚀 Starting Professional Web Application...${NC}"
    echo -e "${YELLOW}Professional features enabled:${NC}"
    echo -e "  🔧 Advanced Mode - Full parameter control"
    echo -e "  🔍 Debug Mode - Intermediate result visualization"
    echo -e "  📊 System Monitoring - Real-time resource usage"
    echo -e "  🎯 Custom Prompts - Flexible detection targets"
    echo ""
    echo -e "${YELLOW}Web UI will be available at: http://localhost:8503${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
    echo ""
    
    # Start streamlit app
    streamlit run $APP_FILE \
        --server.address 0.0.0.0 \
        --server.port 8503 \
        --server.headless true \
        --theme.base dark \
        --theme.primaryColor "#ff6b6b" \
        --theme.backgroundColor "#1e1e1e" \
        --theme.secondaryBackgroundColor "#2d2d2d" \
        --theme.textColor="#ffffff"
}

# Function to test advanced backend
test_advanced_backend() {
    echo -e "${BLUE}🧪 Testing advanced backend...${NC}"
    
    python -c "
from web_backend_advanced import AdvancedWatermarkProcessor
try:
    processor = AdvancedWatermarkProcessor('$CONFIG_FILE')
    print('✅ Advanced backend loaded successfully')
    
    # Test system info
    info = processor.get_advanced_system_info()
    print(f'✅ System info: {info[\"device\"]}, RAM: {info[\"ram_usage\"]}')
    
    # Test presets
    presets = processor.get_parameter_presets()
    print(f'✅ Parameter presets: {list(presets.keys())}')
    
    # Test prompts
    prompts = processor.get_available_prompts()
    print(f'✅ Available prompts: {len(prompts)} options')
    
except Exception as e:
    print(f'❌ Backend test failed: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Advanced backend test passed${NC}"
        return 0
    else
        echo -e "${RED}❌ Advanced backend test failed${NC}"
        return 1
    fi
}

# Main execution
main() {
    check_environment
    
    if ! check_advanced_config; then
        echo -e "${RED}Please ensure web_config_advanced.yaml exists${NC}"
        exit 1
    fi
    
    # Ask user for options
    echo -e "${YELLOW}Choose an option:${NC}"
    echo "1. Start Professional Web App"
    echo "2. Test Advanced Backend"
    echo "3. Show Parameter Guide"
    echo "4. All of the above"
    echo ""
    read -p "Enter choice (1-4): " choice
    
    case $choice in
        1)
            start_pro_app
            ;;
        2)
            test_advanced_backend
            ;;
        3)
            show_parameter_guide
            echo ""
            echo -e "${YELLOW}Press Enter to continue to web app...${NC}"
            read -r
            start_pro_app
            ;;
        4)
            test_advanced_backend
            echo ""
            show_parameter_guide
            echo ""
            echo -e "${YELLOW}Press Enter to continue to web app...${NC}"
            read -r
            start_pro_app
            ;;
        *)
            echo -e "${RED}Invalid choice. Starting web app...${NC}"
            start_pro_app
            ;;
    esac
}

# Parse command line arguments
case "${1:-start}" in
    "start")
        main
        ;;
    "test")
        check_environment
        check_advanced_config
        test_advanced_backend
        ;;
    "guide")
        show_parameter_guide
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  start  - Start the professional web application (default)"
        echo "  test   - Test advanced backend only"
        echo "  guide  - Show parameter guide"
        echo "  help   - Show this help"
        echo ""
        echo "Features:"
        echo "  🔧 30+ Advanced Parameters"
        echo "  🎯 Custom Detection Prompts"
        echo "  ⚡ Performance Presets"
        echo "  📊 System Monitoring"
        echo "  🔍 Debug Mode"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac