"""
AI Watermark Remover - Simple & Reliable Version
基于原始工作版本，添加清晰的工作流程和模型选择
"""
import streamlit as st
import time
import io
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import Optional, Dict, Any

# 配置页面
st.set_page_config(
    page_title="AI Watermark Remover - Simple",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入后端模块 - 使用稳定的原始版本
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
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

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

def render_model_selection():
    """模型选择步骤"""
    st.header("🎯 Step 1: Select Detection Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Custom Watermark Detection")
        st.info("专门训练的FPN+MIT-B5模型，专注于水印检测")
        
        with st.container():
            st.write("**适用场景：**")
            st.write("• 标准水印图片")
            st.write("• 需要高精度检测")
            st.write("• 已知水印类型")
        
        if st.button("🎯 Use Custom Model", type="primary", use_container_width=True):
            st.session_state.selected_model = "custom"
            st.rerun()
    
    with col2:
        st.subheader("🔍 Florence-2 Detection")
        st.info("Microsoft多模态模型，支持文本描述检测")
        
        with st.container():
            st.write("**适用场景：**")
            st.write("• 多样化水印类型")
            st.write("• 文字水印")
            st.write("• 不规则水印")
        
        if st.button("🔍 Use Florence-2 Model", use_container_width=True):
            st.session_state.selected_model = "florence"
            st.rerun()
    
    # 显示当前选择
    if st.session_state.selected_model:
        model_name = "Custom FPN+MIT-B5" if st.session_state.selected_model == "custom" else "Florence-2"
        st.success(f"✅ Selected Model: **{model_name}**")
        return st.session_state.selected_model
    
    return None

def render_basic_settings(model_type: str):
    """基础设置"""
    st.header("⚙️ Step 2: Configure Settings")
    
    settings = {}
    
    # 基础设置
    col1, col2 = st.columns(2)
    
    with col1:
        settings['transparent'] = st.checkbox(
            "Make transparent instead of remove",
            help="创建透明区域而不是填充修复"
        )
        
        settings['max_bbox_percent'] = st.slider(
            "Max detection area (%)", 1.0, 50.0, 10.0, 1.0,
            help="限制检测区域的最大百分比"
        )
    
    with col2:
        settings['force_format'] = st.selectbox(
            "Output format", ["PNG", "WEBP", "JPG"],
            help="输出图片格式"
        )
        
        if model_type == "florence":
            settings['detection_prompt'] = st.selectbox(
                "Detection target", 
                ["watermark", "logo", "text overlay", "signature"],
                help="Florence-2检测目标类型"
            )
    
    # 质量设置
    with st.expander("🔧 Quality Settings", expanded=False):
        settings['quality_mode'] = st.selectbox(
            "Processing quality", 
            ["Fast", "Balanced", "High Quality"],
            index=1,
            help="处理质量模式"
        )
    
    return settings

def process_image_step(image: Image.Image, model_type: str, settings: Dict[str, Any]):
    """图像处理步骤"""
    st.header("🚀 Step 3: Process Image")
    
    # 显示处理信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", model_type.title())
    with col2:
        st.metric("Mode", "Transparent" if settings['transparent'] else "Remove")
    with col3:
        st.metric("Quality", settings.get('quality_mode', 'Balanced'))
    
    # 处理按钮
    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
        
        # 进度显示
        progress_container = st.container()
        with progress_container:
            progress = st.progress(0)
            status = st.empty()
            
            try:
                # 步骤1: 准备
                status.text("📸 Preparing image...")
                progress.progress(0.2)
                time.sleep(0.5)
                
                # 步骤2: 检测
                model_name = "Custom FPN+MIT-B5" if model_type == "custom" else "Florence-2"
                status.text(f"🎯 Detecting watermarks with {model_name}...")
                progress.progress(0.5)
                
                # 实际处理
                result = st.session_state.processor.process_image(
                    image=image,
                    transparent=settings['transparent'],
                    max_bbox_percent=settings['max_bbox_percent'],
                    force_format=settings['force_format']
                )
                
                # 步骤3: 处理
                if not settings['transparent']:
                    status.text("🎨 Inpainting watermarks...")
                else:
                    status.text("🎨 Applying transparency...")
                progress.progress(0.8)
                time.sleep(0.5)
                
                # 步骤4: 完成
                status.text("✨ Finalizing...")
                progress.progress(1.0)
                time.sleep(0.5)
                
                if result.success:
                    status.text("✅ Processing completed successfully!")
                    return result
                else:
                    status.text("❌ Processing failed!")
                    st.error(f"Error: {result.error_message}")
                    return None
                    
            except Exception as e:
                status.text("❌ Processing failed!")
                st.error(f"Error: {str(e)}")
                return None
    
    return None

def display_results(result: ProcessingResult, original_image: Image.Image, image_name: str):
    """显示结果"""
    st.header("🎉 Processing Results")
    
    # 结果对比
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(original_image, use_column_width=True)
        
        if result.mask_image:
            st.subheader("🎭 Detected Mask")
            st.image(result.mask_image, use_column_width=True)
    
    with col2:
        st.subheader("✨ Final Result")
        if result.result_image:
            st.image(result.result_image, use_column_width=True)
            
            # 显示透明背景选项（如果是RGBA）
            if result.result_image.mode == 'RGBA':
                bg_option = st.selectbox(
                    "Preview background", 
                    ["white", "black", "checkered"],
                    key="preview_bg"
                )
                
                if bg_option != "white":
                    display_img = show_with_background(result.result_image, bg_option)
                    st.image(display_img, caption=f"Preview with {bg_option} background", use_column_width=True)
    
    # 性能信息
    if result.processing_time:
        st.info(f"⏱️ Processing time: {result.processing_time:.2f} seconds")
    
    # 下载选项
    st.subheader("📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    formats = [("PNG", "image/png"), ("WEBP", "image/webp"), ("JPG", "image/jpeg")]
    
    for idx, (fmt, mime) in enumerate(formats):
        with [col1, col2, col3][idx]:
            create_download_button(result.result_image, f"{image_name}_processed.{fmt.lower()}", fmt, mime)

def show_with_background(image: Image.Image, bg_type: str) -> Image.Image:
    """显示带背景的图片"""
    if image.mode != 'RGBA':
        return image
    
    if bg_type == "black":
        bg = Image.new('RGB', image.size, (0, 0, 0))
    elif bg_type == "checkered":
        # 创建棋盘背景
        bg = Image.new('RGB', image.size, (255, 255, 255))
        for y in range(0, image.size[1], 20):
            for x in range(0, image.size[0], 20):
                if (x//20 + y//20) % 2:
                    bg.paste((200, 200, 200), (x, y, min(x+20, image.size[0]), min(y+20, image.size[1])))
    else:  # white
        bg = Image.new('RGB', image.size, (255, 255, 255))
    
    bg.paste(image, mask=image.split()[-1])
    return bg

def create_download_button(image: Image.Image, filename: str, format_type: str, mime: str):
    """创建下载按钮"""
    img_buffer = io.BytesIO()
    
    if format_type == "PNG":
        image.save(img_buffer, format="PNG")
    elif format_type == "WEBP":
        image.save(img_buffer, format="WEBP", quality=95)
    else:  # JPG
        if image.mode == "RGBA":
            rgb_img = Image.new("RGB", image.size, (255, 255, 255))
            rgb_img.paste(image, mask=image.split()[-1])
            image = rgb_img
        image.save(img_buffer, format="JPEG", quality=95)
    
    img_buffer.seek(0)
    
    st.download_button(
        label=f"📥 {format_type}",
        data=img_buffer.getvalue(),
        file_name=filename,
        mime=mime,
        use_container_width=True
    )

def main():
    """主应用"""
    
    # 标题
    st.title("🎨 AI Watermark Remover")
    st.markdown("**Simple & Reliable - Clear Workflow Edition**")
    
    # 加载处理器
    if not load_processor():
        return
    
    # 侧边栏
    with st.sidebar:
        st.header("📋 Processing Status")
        
        # 重置按钮
        if st.button("🔄 Reset Workflow"):
            st.session_state.selected_model = None
            st.rerun()
        
        st.divider()
        
        # 帮助信息
        with st.expander("💡 Quick Guide"):
            st.write("**Step 1:** Choose detection model")
            st.write("**Step 2:** Configure settings")
            st.write("**Step 3:** Upload & process")
            st.write("**Step 4:** Download results")
    
    # 主工作流程
    st.header("📸 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload an image with watermarks to remove"
    )
    
    if uploaded_file is not None:
        # 显示原图
        original_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)
            st.text(f"Size: {original_image.size[0]} × {original_image.size[1]}")
        
        with col2:
            # 步骤1: 模型选择
            selected_model = render_model_selection()
            
            if selected_model:
                # 步骤2: 设置配置
                settings = render_basic_settings(selected_model)
                
                # 步骤3: 处理图像
                result = process_image_step(original_image, selected_model, settings)
                
                if result and result.success:
                    # 步骤4: 显示结果
                    display_results(result, original_image, Path(uploaded_file.name).stem)
    
    # 页脚
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🎨 AI Watermark Remover - Simple & Reliable Edition</p>
        <p>🎯 Clear workflow • 🔧 Stable processing • 📥 Easy download</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()