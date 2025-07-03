"""
AI Watermark Remover - 模块化主入口
整合配置、UI、推理和图像处理模块
"""

import streamlit as st
import logging
from PIL import Image

# 导入模块
from config.config import ConfigManager
from interfaces.web.ui import MainInterface
from core.inference import InferenceManager

# 配置页面
def setup_page_config(config_manager):
    """设置页面配置"""
    app_config = config_manager.app_config
    st.set_page_config(
        page_title=app_config.page_title,
        page_icon=app_config.page_icon,
        layout=app_config.layout,
        initial_sidebar_state=app_config.initial_sidebar_state
    )

# 配置日志
def setup_logging():
    """配置日志"""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

# 初始化session state
def init_session_state():
    """初始化session state"""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'processing_result' not in st.session_state:
        st.session_state.processing_result = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None

# 加载处理器
def load_processor(inference_manager):
    """加载处理器"""
    if st.session_state.processor is None:
        with st.spinner("Loading AI models..."):
            try:
                if inference_manager.load_processor():
                    st.session_state.processor = inference_manager
                    st.success("✅ AI models loaded successfully!")
                    return True
                else:
                    st.error("❌ Failed to load models")
                    return False
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                return False
    return True

# 获取处理器
def get_processor():
    """获取处理器"""
    return st.session_state.processor

# 主应用函数
def main():
    """主应用函数"""
    # 设置日志
    logger = setup_logging()
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    # 设置页面配置
    setup_page_config(config_manager)
    
    # 初始化session state
    init_session_state()
    
    # 初始化推理管理器
    inference_manager = InferenceManager(config_manager)
    
    # 初始化主界面
    main_interface = MainInterface(config_manager)
    
    # 加载处理器
    if not load_processor(inference_manager):
        st.error("Failed to initialize application. Please check the logs.")
        return
    
    # 渲染主界面
    main_interface.render(
        inference_manager=get_processor() or inference_manager,
        processing_result=st.session_state.processing_result
    )
    
    # 页脚信息
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔬 Debug Edition")
    with col2:
        st.caption("⚡ Real-time Parameters")  
    with col3:
        st.caption("🔄 Interactive Comparison")

if __name__ == "__main__":
    main() 