"""
AI Watermark Remover - Modular Web Interface
模块化Web界面主入口
"""

import streamlit as st
import logging
from pathlib import Path
import sys

# 添加包路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# 导入组件
from watermark_remover_ai.interfaces.web.frontend.components.parameter_panel import ParameterPanel
from watermark_remover_ai.interfaces.web.frontend.components.image_comparison import ImageComparison
from watermark_remover_ai.interfaces.web.frontend.components.download_buttons import DownloadButtons
from watermark_remover_ai.interfaces.web.frontend.layouts.main_layout import MainLayout
from watermark_remover_ai.interfaces.web.services.processing_service import ProcessingService

# 配置页面
st.set_page_config(
    page_title="AI Watermark Remover - Modular",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args=None, config=None):
    """主函数"""
    # 初始化服务
    processing_service = ProcessingService(config)
    
    # 初始化组件
    parameter_panel = ParameterPanel()
    image_comparison = ImageComparison()
    download_buttons = DownloadButtons()
    main_layout = MainLayout()
    
    # 渲染主布局
    main_layout.render(
        parameter_panel=parameter_panel,
        image_comparison=image_comparison,
        download_buttons=download_buttons,
        processing_service=processing_service
    )

if __name__ == "__main__":
    main() 