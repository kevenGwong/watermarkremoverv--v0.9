"""
Streamlit Web UI for AI Watermark Remover
A web interface that replicates the functionality of remwmgui.py
"""
import streamlit as st
import time
import io
import zipfile
from pathlib import Path
from PIL import Image
import logging
import traceback
from typing import List, Optional

# 配置页面
st.set_page_config(
    page_title="AI Watermark Remover",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入后端模块
try:
    from web_backend import WatermarkProcessor, ProcessingResult
except ImportError as e:
    st.error(f"Failed to import backend modules: {e}")
    st.error("Please ensure all dependencies are installed and models are available.")
    st.stop()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []

def load_processor():
    """加载处理器（只加载一次）"""
    if st.session_state.processor is None:
        with st.spinner("Loading AI models... This may take a few minutes on first run."):
            try:
                st.session_state.processor = WatermarkProcessor("web_config.yaml")
                st.success("✅ AI models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                st.error("Please check your model paths and dependencies.")
                return False
    return True

def process_uploaded_image(uploaded_file, settings: dict) -> Optional[ProcessingResult]:
    """处理上传的图片"""
    try:
        # 读取图片
        image = Image.open(uploaded_file)
        
        # 处理图片
        result = st.session_state.processor.process_image(
            image=image,
            transparent=settings['transparent'],
            max_bbox_percent=settings['max_bbox_percent'],
            force_format=settings['force_format']
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return ProcessingResult(success=False, error_message=str(e))

def create_download_link(image: Image.Image, filename: str, format_type: str = "PNG"):
    """创建图片下载链接"""
    img_buffer = io.BytesIO()
    
    # 根据格式和透明度调整
    if format_type.upper() == "PNG":
        image.save(img_buffer, format="PNG")
    elif format_type.upper() == "WEBP":
        image.save(img_buffer, format="WEBP", quality=95)
    elif format_type.upper() in ["JPG", "JPEG"]:
        # JPG不支持透明度，转换为RGB
        if image.mode == "RGBA":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if len(image.split()) == 4 else None)
            image = rgb_image
        image.save(img_buffer, format="JPEG", quality=95)
    
    img_buffer.seek(0)
    
    return st.download_button(
        label=f"📥 Download {filename}",
        data=img_buffer.getvalue(),
        file_name=filename,
        mime=f"image/{format_type.lower()}"
    )

def create_batch_download(results: List[ProcessingResult], format_type: str = "PNG"):
    """创建批量下载压缩包"""
    if not results:
        return
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, result in enumerate(results):
            if result.success and result.result_image:
                img_buffer = io.BytesIO()
                
                if format_type.upper() == "PNG":
                    result.result_image.save(img_buffer, format="PNG")
                    filename = f"watermark_removed_{i+1}.png"
                elif format_type.upper() == "WEBP":
                    result.result_image.save(img_buffer, format="WEBP", quality=95)
                    filename = f"watermark_removed_{i+1}.webp"
                elif format_type.upper() in ["JPG", "JPEG"]:
                    img = result.result_image
                    if img.mode == "RGBA":
                        rgb_image = Image.new("RGB", img.size, (255, 255, 255))
                        rgb_image.paste(img, mask=img.split()[-1] if len(img.split()) == 4 else None)
                        img = rgb_image
                    img.save(img_buffer, format="JPEG", quality=95)
                    filename = f"watermark_removed_{i+1}.jpg"
                
                zip_file.writestr(filename, img_buffer.getvalue())
    
    zip_buffer.seek(0)
    
    return st.download_button(
        label=f"📦 Download All ({len([r for r in results if r.success])} files)",
        data=zip_buffer.getvalue(),
        file_name=f"watermark_removed_batch.zip",
        mime="application/zip"
    )

def main():
    """主应用函数"""
    
    # 标题
    st.title("🎨 AI Watermark Remover")
    st.markdown("**Remove watermarks from images using advanced AI technology**")
    
    # 加载处理器
    if not load_processor():
        return
    
    # 侧边栏设置
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # 处理模式
        mode = st.radio(
            "Processing Mode",
            ["Single Image", "Batch Processing"],
            help="Choose between processing one image or multiple images"
        )
        
        st.divider()
        
        # 处理选项
        st.subheader("Processing Options")
        
        transparent = st.checkbox(
            "Make watermark transparent",
            value=False,
            help="Make watermark regions transparent instead of removing them"
        )
        
        max_bbox_percent = st.slider(
            "Max watermark size (%)",
            min_value=1,
            max_value=100,
            value=10,
            help="Maximum percentage of image that a watermark can cover"
        )
        
        force_format = st.selectbox(
            "Output format",
            ["Auto", "PNG", "WEBP", "JPG"],
            help="Force specific output format or keep original"
        )
        
        st.divider()
        
        # 高级选项
        with st.expander("🔧 Advanced Options"):
            mask_type = st.selectbox(
                "Mask Generation Method",
                ["Custom Model", "Florence-2"],
                help="Choose between custom trained model or Florence-2"
            )
            
            show_masks = st.checkbox(
                "Show generated masks",
                value=True,
                help="Display the masks used for watermark detection"
            )
        
        st.divider()
        
        # 系统信息
        if st.button("🔄 Refresh System Info"):
            system_info = st.session_state.processor.get_system_info()
            st.subheader("💻 System Status")
            st.text(f"Device: {system_info['device']}")
            st.text(f"CUDA: {'Available' if system_info['cuda_available'] else 'Not Available'}")
            st.text(f"RAM: {system_info['ram_usage']}")
            st.text(f"VRAM: {system_info['vram_usage']}")
            st.text(f"CPU: {system_info['cpu_usage']}")
    
    # 主内容区域
    if mode == "Single Image":
        st.header("📸 Single Image Processing")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload an image with watermarks to remove"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_column_width=True)
                st.text(f"Size: {original_image.size[0]} × {original_image.size[1]}")
            
            # 处理按钮
            if st.button("🚀 Remove Watermark", type="primary"):
                settings = {
                    'transparent': transparent,
                    'max_bbox_percent': max_bbox_percent,
                    'force_format': None if force_format == "Auto" else force_format
                }
                
                with st.spinner("Processing image... This may take a few seconds."):
                    result = process_uploaded_image(uploaded_file, settings)
                
                if result.success:
                    with col2:
                        st.subheader("Processed Image")
                        st.image(result.result_image, use_column_width=True)
                        st.text(f"Processing time: {result.processing_time:.2f}s")
                        
                        # 下载按钮
                        output_format = force_format if force_format != "Auto" else "PNG"
                        filename = f"watermark_removed.{output_format.lower()}"
                        create_download_link(result.result_image, filename, output_format)
                    
                    # 显示mask（如果启用）
                    if show_masks and result.mask_image:
                        st.subheader("Generated Mask")
                        st.image(result.mask_image, use_column_width=True, caption="White areas show detected watermarks")
                
                else:
                    st.error(f"❌ Processing failed: {result.error_message}")
    
    else:  # Batch Processing
        st.header("📁 Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.info(f"📊 {len(uploaded_files)} files uploaded")
            
            # 显示预览
            if len(uploaded_files) <= 6:  # 限制预览数量
                cols = st.columns(min(3, len(uploaded_files)))
                for i, file in enumerate(uploaded_files[:6]):
                    with cols[i % 3]:
                        image = Image.open(file)
                        st.image(image, caption=file.name, use_column_width=True)
            
            # 批处理按钮
            if st.button("🚀 Process All Images", type="primary"):
                settings = {
                    'transparent': transparent,
                    'max_bbox_percent': max_bbox_percent,
                    'force_format': None if force_format == "Auto" else force_format
                }
                
                # 进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                    
                    result = process_uploaded_image(uploaded_file, settings)
                    results.append(result)
                    
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                
                status_text.text("✅ Batch processing completed!")
                st.session_state.processing_results = results
                
                # 显示结果统计
                successful = len([r for r in results if r.success])
                failed = len(results) - successful
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Processed", len(results))
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", failed)
                
                # 批量下载
                if successful > 0:
                    output_format = force_format if force_format != "Auto" else "PNG"
                    create_batch_download(results, output_format)
                
                # 显示详细结果
                st.subheader("📋 Detailed Results")
                for i, (file, result) in enumerate(zip(uploaded_files, results)):
                    with st.expander(f"{file.name} - {'✅' if result.success else '❌'}"):
                        if result.success:
                            col1, col2 = st.columns(2)
                            with col1:
                                original = Image.open(file)
                                st.image(original, caption="Original", use_column_width=True)
                            with col2:
                                st.image(result.result_image, caption="Processed", use_column_width=True)
                            
                            st.text(f"Processing time: {result.processing_time:.2f}s")
                            
                            # 单独下载
                            output_format = force_format if force_format != "Auto" else "PNG"
                            filename = f"{Path(file.name).stem}_processed.{output_format.lower()}"
                            create_download_link(result.result_image, filename, output_format)
                        else:
                            st.error(f"Error: {result.error_message}")
    
    # 页脚
    st.divider()
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🎨 AI Watermark Remover | Built with Streamlit</p>
        <p>Powered by Florence-2 + LaMA & Custom Segmentation Models</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()