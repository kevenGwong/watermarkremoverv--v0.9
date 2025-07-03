"""
下载按钮组件
"""

import streamlit as st
from PIL import Image
import io
from typing import Optional

class DownloadButtons:
    """下载按钮组件"""
    
    def render(self, image: Optional[Image.Image] = None, filename_base: str = "watermark_removed"):
        """渲染下载按钮"""
        if image is None:
            return
        
        st.subheader("💾 下载结果")
        
        # 创建三列布局
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._create_download_button(image, f"{filename_base}.png", "PNG格式")
        
        with col2:
            # 转换为JPEG
            jpeg_image = image.convert('RGB') if image.mode in ['RGBA', 'LA'] else image
            self._create_download_button(jpeg_image, f"{filename_base}.jpg", "JPEG格式")
        
        with col3:
            # 转换为WEBP
            webp_image = image.convert('RGB') if image.mode in ['RGBA', 'LA'] else image
            self._create_download_button(webp_image, f"{filename_base}.webp", "WEBP格式")
    
    def _create_download_button(self, image: Image.Image, filename: str, label: str):
        """创建下载按钮"""
        # 转换图像为字节流
        img_byte_arr = io.BytesIO()
        
        if filename.endswith('.png'):
            image.save(img_byte_arr, format='PNG')
        elif filename.endswith('.jpg'):
            image.save(img_byte_arr, format='JPEG', quality=95)
        elif filename.endswith('.webp'):
            image.save(img_byte_arr, format='WEBP', quality=95)
        
        img_byte_arr.seek(0)
        
        # 创建下载按钮
        st.download_button(
            label=label,
            data=img_byte_arr.getvalue(),
            file_name=filename,
            mime="image/png" if filename.endswith('.png') else "image/jpeg"
        ) 