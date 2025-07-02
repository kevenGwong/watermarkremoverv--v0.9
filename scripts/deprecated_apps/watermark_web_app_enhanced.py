"""
Enhanced Streamlit Web UI with debugging features and custom mask upload
"""
import streamlit as st
import time
import io
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import logging
import traceback
from typing import List, Optional

# 配置页面
st.set_page_config(
    page_title="AI Watermark Remover Enhanced",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入后端模块
try:
    from web_backend import WatermarkProcessor, ProcessingResult
except ImportError as e:
    st.error(f"Failed to import backend modules: {e}")
    st.stop()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

def load_processor():
    """加载处理器"""
    if st.session_state.processor is None:
        with st.spinner("Loading AI models..."):
            try:
                st.session_state.processor = WatermarkProcessor("web_config.yaml")
                st.success("✅ AI models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                return False
    return True

def create_demo_image_with_watermark():
    """创建带水印的演示图片"""
    # 创建基础图片
    img = Image.new('RGB', (600, 400), 'lightblue')
    draw = ImageDraw.Draw(img)
    
    # 绘制一些内容
    draw.rectangle([50, 50, 250, 150], fill='lightgreen', outline='darkgreen', width=3)
    draw.rectangle([350, 250, 550, 350], fill='lightyellow', outline='orange', width=3)
    
    # 添加"水印" - 半透明黑色矩形
    watermark = Image.new('RGBA', (200, 60), (0, 0, 0, 128))
    img_rgba = img.convert('RGBA')
    img_rgba.paste(watermark, (300, 50), watermark)
    
    # 添加文字水印
    draw = ImageDraw.Draw(img_rgba)
    try:
        draw.text((310, 65), "WATERMARK", fill=(255, 255, 255, 200))
    except:
        draw.text((310, 65), "WATERMARK", fill=(255, 255, 255))
    
    return img_rgba.convert('RGB')

def visualize_mask_effect(original_image: Image.Image, mask: Image.Image, transparent_result: Image.Image):
    """可视化mask效果"""
    st.subheader("🔍 Mask效果分析")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.text("原图")
        st.image(original_image, use_column_width=True)
    
    with col2:
        st.text("生成的Mask")
        st.image(mask, use_column_width=True)
        
        # 显示mask统计信息
        mask_array = np.array(mask)
        white_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        st.text(f"白色像素: {white_pixels}")
        st.text(f"占比: {white_pixels/total_pixels*100:.1f}%")
    
    with col3:
        st.text("透明效果")
        st.image(transparent_result, use_column_width=True)
    
    with col4:
        st.text("Mask叠加")
        # 创建mask叠加效果
        overlay = original_image.copy().convert('RGBA')
        mask_colored = Image.new('RGBA', mask.size, (255, 0, 0, 100))  # 红色半透明
        mask_array = np.array(mask)
        mask_rgba = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        mask_rgba[mask_array > 128] = [255, 0, 0, 100]  # 红色标记水印区域
        mask_overlay = Image.fromarray(mask_rgba, 'RGBA')
        overlay.paste(mask_overlay, (0, 0), mask_overlay)
        st.image(overlay, use_column_width=True)

def fix_transparent_display_issue(transparent_image: Image.Image, background_color='white'):
    """修复透明显示问题 - 为透明图片添加背景色用于预览"""
    if transparent_image.mode != 'RGBA':
        return transparent_image
    
    # 创建指定颜色的背景
    if background_color == 'white':
        bg_color = (255, 255, 255, 255)
    elif background_color == 'checkered':
        # 创建棋盘格背景
        bg = Image.new('RGBA', transparent_image.size, (255, 255, 255, 255))
        for x in range(0, transparent_image.width, 20):
            for y in range(0, transparent_image.height, 20):
                if (x//20 + y//20) % 2:
                    for i in range(min(20, transparent_image.width - x)):
                        for j in range(min(20, transparent_image.height - y)):
                            bg.putpixel((x + i, y + j), (200, 200, 200, 255))
        bg.paste(transparent_image, (0, 0), transparent_image)
        return bg
    else:
        bg_color = (0, 0, 0, 255)  # 黑色
    
    background = Image.new('RGBA', transparent_image.size, bg_color)
    background.paste(transparent_image, (0, 0), transparent_image)
    return background

def process_with_custom_mask(image: Image.Image, mask: Image.Image) -> ProcessingResult:
    """使用自定义mask处理图片"""
    try:
        start_time = time.time()
        
        # 确保mask格式正确
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # 应用透明处理
        result_image = apply_transparent_effect(image, mask)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            success=True,
            result_image=result_image,
            mask_image=mask,
            processing_time=processing_time
        )
    except Exception as e:
        return ProcessingResult(
            success=False,
            error_message=str(e),
            processing_time=time.time() - start_time
        )

def apply_transparent_effect(image: Image.Image, mask: Image.Image) -> Image.Image:
    """应用透明效果 - 优化版本"""
    # 转换格式
    image = image.convert("RGBA")
    mask = mask.convert("L")
    
    # 转换为numpy数组进行快速处理
    img_array = np.array(image)
    mask_array = np.array(mask)
    
    # 创建透明结果
    result_array = img_array.copy()
    
    # 水印区域设为透明 (mask中白色区域 > 128)
    transparent_mask = mask_array > 128
    result_array[transparent_mask, 3] = 0  # 设置alpha通道为0 (透明)
    
    return Image.fromarray(result_array, 'RGBA')

def create_download_link_enhanced(image: Image.Image, filename: str, format_type: str = "PNG", show_preview: bool = True):
    """增强的下载链接，支持透明图片"""
    img_buffer = io.BytesIO()
    
    # 根据格式处理透明度
    if format_type.upper() == "PNG":
        image.save(img_buffer, format="PNG")
        mime_type = "image/png"
    elif format_type.upper() == "WEBP":
        image.save(img_buffer, format="WEBP", quality=95)
        mime_type = "image/webp"
    elif format_type.upper() in ["JPG", "JPEG"]:
        # JPG不支持透明度，需要合成背景
        if image.mode == "RGBA":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if len(image.split()) == 4 else None)
            image = rgb_image
        image.save(img_buffer, format="JPEG", quality=95)
        mime_type = "image/jpeg"
    
    img_buffer.seek(0)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if show_preview and format_type.upper() in ["JPG", "JPEG"] and image.mode == "RGBA":
            st.info("💡 JPG格式不支持透明度，透明区域将显示为白色背景")
    
    with col2:
        st.download_button(
            label=f"📥 Download {filename}",
            data=img_buffer.getvalue(),
            file_name=filename,
            mime=mime_type
        )

def main():
    """主应用函数"""
    
    # 标题
    st.title("🎨 AI Watermark Remover Enhanced")
    st.markdown("**Enhanced version with debugging features and custom mask upload**")
    
    # 加载处理器
    if not load_processor():
        return
    
    # 侧边栏设置
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # 调试模式
        st.session_state.debug_mode = st.checkbox(
            "🔧 Debug Mode",
            value=st.session_state.debug_mode,
            help="Show mask visualization and detailed information"
        )
        
        # 处理模式
        processing_mode = st.radio(
            "Processing Mode",
            ["Demo", "Single Image", "Custom Mask", "Batch Processing"],
            help="Choose processing mode"
        )
        
        st.divider()
        
        # 处理选项
        st.subheader("Processing Options")
        
        transparent = st.checkbox(
            "Make watermark transparent",
            value=True,
            help="Make watermark regions transparent instead of removing them"
        )
        
        if transparent:
            preview_bg = st.selectbox(
                "Preview background",
                ["white", "black", "checkered"],
                help="Background for transparent preview"
            )
        
        max_bbox_percent = st.slider(
            "Max watermark size (%)",
            min_value=1,
            max_value=100,
            value=15,
            help="Maximum percentage of image that a watermark can cover"
        )
        
        force_format = st.selectbox(
            "Output format",
            ["PNG", "WEBP", "JPG"],
            help="PNG recommended for transparent images"
        )
        
        st.divider()
        
        # 高级选项
        with st.expander("🔧 Advanced Options"):
            mask_type = st.selectbox(
                "Mask Generation Method",
                ["Custom Model", "Florence-2"],
                help="Choose mask generation method"
            )
    
    # 主内容区域
    if processing_mode == "Demo":
        st.header("🎮 Demo Mode - Test with Generated Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🎨 Generate Demo Image"):
                demo_image = create_demo_image_with_watermark()
                st.session_state.demo_image = demo_image
            
            if 'demo_image' in st.session_state:
                st.subheader("Demo Image with Watermark")
                st.image(st.session_state.demo_image, use_column_width=True)
                
                if st.button("🚀 Process Demo Image"):
                    with st.spinner("Processing demo image..."):
                        if mask_type == "Custom Model":
                            result = st.session_state.processor.process_image(
                                image=st.session_state.demo_image,
                                transparent=transparent,
                                max_bbox_percent=max_bbox_percent
                            )
                        else:
                            # 使用Florence-2
                            st.session_state.processor.config['mask_generator']['model_type'] = 'florence'
                            result = st.session_state.processor.process_image(
                                image=st.session_state.demo_image,
                                transparent=transparent,
                                max_bbox_percent=max_bbox_percent
                            )
                        
                        if result.success:
                            with col2:
                                st.subheader("Processed Result")
                                
                                if transparent:
                                    # 显示透明效果预览
                                    preview_img = fix_transparent_display_issue(
                                        result.result_image, preview_bg
                                    )
                                    st.image(preview_img, use_column_width=True)
                                    st.text(f"Processing time: {result.processing_time:.2f}s")
                                    
                                    # 下载选项
                                    filename = f"demo_transparent.{force_format.lower()}"
                                    create_download_link_enhanced(
                                        result.result_image, filename, force_format
                                    )
                                    
                                    # 调试模式显示详细信息
                                    if st.session_state.debug_mode and result.mask_image:
                                        visualize_mask_effect(
                                            st.session_state.demo_image,
                                            result.mask_image,
                                            result.result_image
                                        )
                                else:
                                    st.image(result.result_image, use_column_width=True)
                                    filename = f"demo_inpainted.{force_format.lower()}"
                                    create_download_link_enhanced(
                                        result.result_image, filename, force_format
                                    )
                        else:
                            st.error(f"❌ Processing failed: {result.error_message}")
    
    elif processing_mode == "Custom Mask":
        st.header("🎯 Custom Mask Upload")
        st.info("💡 Upload your own image and corresponding mask")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            uploaded_image = st.file_uploader(
                "Choose image file",
                type=['jpg', 'jpeg', 'png', 'webp'],
                key="custom_image"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, use_column_width=True)
                st.text(f"Size: {image.size[0]} × {image.size[1]}")
        
        with col2:
            st.subheader("Upload Mask")
            uploaded_mask = st.file_uploader(
                "Choose mask file (white=watermark, black=background)",
                type=['jpg', 'jpeg', 'png', 'webp'],
                key="custom_mask"
            )
            
            if uploaded_mask:
                mask = Image.open(uploaded_mask).convert('L')
                st.image(mask, use_column_width=True)
                
                # 显示mask统计
                mask_array = np.array(mask)
                white_pixels = np.sum(mask_array > 128)
                total_pixels = mask_array.size
                st.text(f"White pixels: {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
        
        if uploaded_image and uploaded_mask:
            if st.button("🚀 Process with Custom Mask", type="primary"):
                image = Image.open(uploaded_image)
                mask = Image.open(uploaded_mask)
                
                # 确保尺寸匹配
                if image.size != mask.size:
                    mask = mask.resize(image.size, Image.LANCZOS)
                    st.warning(f"⚠️ Mask resized to match image size: {image.size}")
                
                result = process_with_custom_mask(image, mask)
                
                if result.success:
                    st.subheader("🎉 Result")
                    
                    if transparent:
                        preview_img = fix_transparent_display_issue(
                            result.result_image, preview_bg
                        )
                        st.image(preview_img, use_column_width=True)
                        
                        filename = f"custom_transparent.{force_format.lower()}"
                        create_download_link_enhanced(
                            result.result_image, filename, force_format
                        )
                        
                        if st.session_state.debug_mode:
                            visualize_mask_effect(image, mask, result.result_image)
                    
                    st.success(f"✅ Processing completed in {result.processing_time:.2f}s")
                else:
                    st.error(f"❌ Processing failed: {result.error_message}")
    
    elif processing_mode == "Single Image":
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
                with st.spinner("Processing image..."):
                    result = st.session_state.processor.process_image(
                        image=original_image,
                        transparent=transparent,
                        max_bbox_percent=max_bbox_percent,
                        force_format=force_format if force_format != "Auto" else None
                    )
                
                if result.success:
                    with col2:
                        st.subheader("Processed Image")
                        
                        if transparent:
                            preview_img = fix_transparent_display_issue(
                                result.result_image, preview_bg
                            )
                            st.image(preview_img, use_column_width=True)
                        else:
                            st.image(result.result_image, use_column_width=True)
                        
                        st.text(f"Processing time: {result.processing_time:.2f}s")
                        
                        # 下载按钮
                        filename = f"watermark_removed.{force_format.lower()}"
                        create_download_link_enhanced(
                            result.result_image, filename, force_format
                        )
                    
                    # 调试模式显示详细信息
                    if st.session_state.debug_mode and result.mask_image:
                        visualize_mask_effect(
                            original_image, result.mask_image, result.result_image
                        )
                
                else:
                    st.error(f"❌ Processing failed: {result.error_message}")
    
    else:  # Batch Processing
        st.header("📁 Batch Processing")
        st.info("Upload multiple images for batch processing")
        # ... 保持原有的批处理逻辑
    
    # 页脚
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🎨 AI Watermark Remover Enhanced | Built with Streamlit</p>
        <p>🔧 Enhanced with debugging features and custom mask support</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()