"""
图像对比组件
"""

import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison
import numpy as np
from typing import Optional

class ImageComparison:
    """图像对比组件"""
    
    def render(self, original_image: Optional[Image.Image] = None, 
               processed_image: Optional[Image.Image] = None):
        """渲染图像对比"""
        st.title("🔄 图像对比")
        
        if original_image is None:
            st.info("请先上传图像")
            return
        
        # 创建两列布局
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 原始图像")
            st.image(original_image, caption="原始图像", use_column_width=True)
        
        with col2:
            if processed_image is not None:
                st.subheader("✨ 处理结果")
                st.image(processed_image, caption="处理结果", use_column_width=True)
            else:
                st.subheader("✨ 处理结果")
                st.info("等待处理...")
        
        # 如果有处理结果，显示对比
        if processed_image is not None:
            st.subheader("🔄 滑动对比")
            try:
                image_comparison(
                    img1=original_image,
                    img2=processed_image,
                    label1="原始",
                    label2="处理后"
                )
            except Exception as e:
                st.error(f"对比显示失败: {e}")
                # 降级显示
                st.image(original_image, caption="原始图像")
                st.image(processed_image, caption="处理结果")
    
    def add_background(self, rgba_image: Image.Image, bg_type: str) -> Image.Image:
        """为透明图像添加背景"""
        if rgba_image.mode != 'RGBA':
            return rgba_image
        
        # 创建背景
        if bg_type == "white":
            background = Image.new('RGBA', rgba_image.size, (255, 255, 255, 255))
        elif bg_type == "black":
            background = Image.new('RGBA', rgba_image.size, (0, 0, 0, 255))
        elif bg_type == "checkerboard":
            # 创建棋盘格背景
            size = rgba_image.size
            background = Image.new('RGBA', size, (255, 255, 255, 255))
            # 这里可以添加棋盘格逻辑
        else:
            return rgba_image
        
        # 合成图像
        result = Image.alpha_composite(background, rgba_image)
        return result.convert('RGB') 