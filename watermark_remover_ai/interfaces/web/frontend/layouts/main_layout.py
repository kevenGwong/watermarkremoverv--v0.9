"""
主布局组件
"""

import streamlit as st
from PIL import Image
from typing import Dict, Any, Optional

class MainLayout:
    """主布局组件"""
    
    def render(self, parameter_panel, image_comparison, download_buttons, processing_service):
        """渲染主布局"""
        # 页面标题
        st.title("🎨 AI Watermark Remover - 模块化版本")
        st.markdown("---")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择图像文件",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="支持PNG、JPG、JPEG、WEBP格式"
        )
        
        if uploaded_file is not None:
            # 加载图像
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            
            # 获取参数
            params = parameter_panel.render()
            
            # 如果有参数（点击了处理按钮）
            if params:
                # 处理图像
                with st.spinner("正在处理图像..."):
                    result = processing_service.process_image(
                        image=image,
                        **params
                    )
                
                if result.success:
                    st.session_state.processing_result = result
                    st.success("✅ 处理完成！")
                else:
                    st.error(f"❌ 处理失败: {result.error_message}")
            
            # 渲染图像对比
            processed_image = None
            if hasattr(st.session_state, 'processing_result') and st.session_state.processing_result:
                processed_image = st.session_state.processing_result.result_image
            
            image_comparison.render(
                original_image=st.session_state.original_image,
                processed_image=processed_image
            )
            
            # 渲染下载按钮
            if processed_image:
                download_buttons.render(
                    image=processed_image,
                    filename_base=f"watermark_removed_{uploaded_file.name.split('.')[0]}"
                )
        
        # 系统信息
        with st.sidebar:
            st.markdown("---")
            st.subheader("ℹ️ 系统信息")
            st.write(f"Streamlit版本: {st.__version__}")
            st.write("模块化架构 v2.0") 