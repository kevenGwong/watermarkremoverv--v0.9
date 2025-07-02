"""
Watermark Remover Web UI v2 - Clear Workflow & Model Selection
修复版本 - 清晰的工作流程和模型选择
"""
import streamlit as st
import time
import io
import json
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import Dict, Any, Optional

# 配置页面
st.set_page_config(
    page_title="AI Watermark Remover v2",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入后端模块
try:
    from web_backend_advanced import AdvancedWatermarkProcessor, AdvancedProcessingResult
except ImportError as e:
    st.error(f"Failed to import backend modules: {e}")
    st.stop()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'custom_settings' not in st.session_state:
    st.session_state.custom_settings = {}

def load_processor():
    """加载处理器"""
    if st.session_state.processor is None:
        with st.spinner("Loading AI models..."):
            try:
                st.session_state.processor = AdvancedWatermarkProcessor("web_config_advanced.yaml")
                st.success("✅ AI models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                return False
    return True

def save_custom_settings(settings: Dict[str, Any], name: str):
    """保存自定义设置"""
    st.session_state.custom_settings[name] = settings
    # 可选：保存到本地文件
    try:
        settings_file = Path("custom_settings.json")
        all_settings = {}
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                all_settings = json.load(f)
        all_settings[name] = settings
        with open(settings_file, 'w') as f:
            json.dump(all_settings, f, indent=2)
        st.success(f"✅ Settings '{name}' saved!")
    except Exception as e:
        st.warning(f"Settings saved to session but not to file: {e}")

def load_custom_settings() -> Dict[str, Dict[str, Any]]:
    """加载自定义设置"""
    try:
        settings_file = Path("custom_settings.json")
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load custom settings: {e}")
    return st.session_state.custom_settings

def render_step1_model_selection():
    """步骤1: 模型选择"""
    st.header("🎯 Step 1: Select Detection Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Custom Watermark Detection")
        st.info("使用专门训练的FPN+MIT-B5模型检测水印")
        use_custom = st.button("🎯 Use Custom Model", type="primary", use_container_width=True)
        
        if use_custom:
            st.session_state.selected_model = "custom"
            st.rerun()
    
    with col2:
        st.subheader("🔍 Florence-2 Detection") 
        st.info("使用Microsoft Florence-2多模态模型检测")
        use_florence = st.button("🔍 Use Florence-2 Model", use_container_width=True)
        
        if use_florence:
            st.session_state.selected_model = "florence"
            st.rerun()
    
    # 显示当前选择
    if 'selected_model' in st.session_state:
        model_name = "Custom FPN+MIT-B5" if st.session_state.selected_model == "custom" else "Florence-2"
        st.success(f"✅ Selected Model: **{model_name}**")
        return st.session_state.selected_model
    
    return None

def render_step2_parameters(model_type: str):
    """步骤2: 参数设置"""
    st.header("⚙️ Step 2: Configure Parameters")
    
    params = {}
    
    # 参数预设
    if st.session_state.processor:
        presets = st.session_state.processor.get_parameter_presets()
        custom_settings = load_custom_settings()
        
        preset_options = ["Custom"] + list(presets.keys()) + list(custom_settings.keys())
        selected_preset = st.selectbox("🎚️ Parameter Preset", preset_options, 
                                     help="Choose a preset or custom configuration")
        
        if selected_preset != "Custom":
            if selected_preset in presets:
                preset_params = presets[selected_preset]
                st.info(f"📋 Using **{selected_preset}** preset")
            else:
                preset_params = custom_settings[selected_preset]
                st.info(f"📋 Using custom settings: **{selected_preset}**")
            
            with st.expander("📊 Preset Parameters", expanded=False):
                st.json(preset_params)
            params.update(preset_params)
    
    # 模型特定参数
    with st.expander(f"🎯 {model_type.title()} Model Settings", expanded=True):
        if model_type == "custom":
            st.subheader("Custom Model Parameters")
            params['mask_threshold'] = st.slider(
                "Mask Threshold", 0.0, 1.0, 0.5, 0.05,
                help="二值化阈值 - 控制检测敏感度"
            )
            params['mask_dilate_kernel_size'] = st.slider(
                "Dilate Kernel Size", 1, 15, 3, 2,
                help="膨胀核大小 - 扩展检测区域"
            )
            params['mask_dilate_iterations'] = st.slider(
                "Dilate Iterations", 1, 5, 1,
                help="膨胀迭代次数"
            )
        else:  # florence
            st.subheader("Florence-2 Parameters")
            available_prompts = ["watermark", "logo", "text overlay", "signature"]
            prompt_option = st.selectbox("Detection Prompt", available_prompts,
                                       help="检测目标类型")
            params['detection_prompt'] = prompt_option
            
            params['max_bbox_percent'] = st.slider(
                "Max BBox Percent", 1.0, 50.0, 10.0, 1.0,
                help="最大边界框百分比"
            )
            params['confidence_threshold'] = st.slider(
                "Confidence Threshold", 0.1, 0.9, 0.3, 0.05,
                help="检测置信度阈值"
            )
    
    # LaMA处理参数
    with st.expander("🎨 LaMA Inpainting Settings", expanded=False):
        params['ldm_steps'] = st.slider(
            "Processing Steps", 10, 200, 50, 10,
            help="更多步数 = 更高质量但更慢"
        )
        params['ldm_sampler'] = st.selectbox(
            "Sampler", ["ddim", "plms"],
            help="采样算法选择 - ddim(稳定), plms(快速)"
        )
        params['hd_strategy'] = st.selectbox(
            "HD Strategy", ["CROP", "RESIZE", "ORIGINAL"],
            help="高分辨率处理策略"
        )
    
    # 保存自定义设置
    col1, col2 = st.columns(2)
    with col1:
        save_name = st.text_input("💾 Save settings as:", placeholder="my_settings")
    with col2:
        if st.button("💾 Save Settings") and save_name:
            save_custom_settings(params, save_name)
    
    return params

def render_step3_processing(image: Image.Image, model_type: str, params: Dict[str, Any], transparent: bool):
    """步骤3: 图像处理"""
    st.header("🚀 Step 3: Process Image")
    
    # 显示处理信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", model_type.title())
    with col2:
        st.metric("Mode", "Transparent" if transparent else "Remove")
    with col3:
        st.metric("Image Size", f"{image.size[0]}×{image.size[1]}")
    
    # 处理按钮
    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
        
        # 显示处理步骤
        with st.container():
            st.subheader("🔄 Processing Steps")
            
            # 步骤进度条
            progress = st.progress(0)
            status = st.empty()
            
            # 步骤1: 图像预处理
            status.text("📸 Step 1/4: Preprocessing image...")
            progress.progress(0.25)
            time.sleep(0.5)
            
            # 步骤2: 生成mask
            model_name = "Custom FPN+MIT-B5" if model_type == "custom" else "Florence-2"
            status.text(f"🎯 Step 2/4: Generating mask with {model_name}...")
            progress.progress(0.5)
            
            # 实际处理
            result = st.session_state.processor.process_image(
                image=image,
                transparent=transparent,
                advanced_params=params
            )
            
            # 步骤3: LaMA处理
            if not transparent:
                status.text("🎨 Step 3/4: LaMA inpainting...")
                progress.progress(0.75)
                time.sleep(0.5)
            else:
                status.text("🎨 Step 3/4: Applying transparency...")
                progress.progress(0.75)
                time.sleep(0.5)
            
            # 步骤4: 后处理
            status.text("✨ Step 4/4: Final post-processing...")
            progress.progress(1.0)
            time.sleep(0.5)
            
            status.text("✅ Processing completed!")
            
            return result
    
    return None

def render_results(result: AdvancedProcessingResult, image_name: str):
    """显示处理结果"""
    if not result.success:
        st.error(f"❌ Processing failed: {result.error_message}")
        return
    
    st.header("🎉 Processing Results")
    
    # 结果展示
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Generated Mask")
        if result.mask_image:
            st.image(result.mask_image, use_column_width=True)
    
    with col2:
        st.subheader("✨ Final Result")
        if result.result_image:
            st.image(result.result_image, use_column_width=True)
    
    # 性能信息
    if result.processing_time:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processing Time", f"{result.processing_time:.2f}s")
        with col2:
            if result.memory_usage:
                st.metric("Memory Usage", f"{result.memory_usage.get('memory_increase', 0):.1f}%")
        with col3:
            if result.memory_usage and 'gpu_memory_allocated' in result.memory_usage:
                st.metric("GPU Memory", f"{result.memory_usage['gpu_memory_allocated']:.0f}MB")
    
    # 下载选项
    st.subheader("📥 Download Results")
    col1, col2, col3 = st.columns(3)
    
    formats = [("PNG", "image/png"), ("WEBP", "image/webp"), ("JPG", "image/jpeg")]
    for idx, (fmt, mime) in enumerate(formats):
        with [col1, col2, col3][idx]:
            img_buffer = io.BytesIO()
            
            if fmt == "PNG":
                result.result_image.save(img_buffer, format="PNG")
            elif fmt == "WEBP":
                result.result_image.save(img_buffer, format="WEBP", quality=95)
            else:  # JPG
                img = result.result_image
                if img.mode == "RGBA":
                    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1] if len(img.split()) == 4 else None)
                    img = rgb_img
                img.save(img_buffer, format="JPEG", quality=95)
            
            img_buffer.seek(0)
            
            st.download_button(
                label=f"📥 {fmt}",
                data=img_buffer.getvalue(),
                file_name=f"{image_name}_processed.{fmt.lower()}",
                mime=mime,
                use_container_width=True
            )

def main():
    """主应用函数"""
    
    # 标题
    st.title("🎨 AI Watermark Remover v2")
    st.markdown("**Clear workflow with step-by-step processing**")
    
    # 加载处理器
    if not load_processor():
        return
    
    # 侧边栏设置
    with st.sidebar:
        st.header("⚙️ Basic Settings")
        
        transparent = st.checkbox(
            "Make transparent instead of remove",
            help="Make watermark regions transparent instead of inpainting"
        )
        
        if transparent:
            st.selectbox("Preview background", ["white", "black", "checkered"])
        
        st.divider()
        
        # 系统信息
        if st.button("🔄 System Info"):
            sys_info = st.session_state.processor.get_advanced_system_info()
            st.json(sys_info)
    
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
            # 清晰的工作流程
            # Step 1: 模型选择
            selected_model = render_step1_model_selection()
            
            if selected_model:
                # Step 2: 参数配置
                params = render_step2_parameters(selected_model)
                
                if params:
                    # Step 3: 图像处理
                    result = render_step3_processing(original_image, selected_model, params, transparent)
                    
                    if result:
                        # Step 4: 结果展示
                        render_results(result, Path(uploaded_file.name).stem)
    
    # 页脚
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🎨 AI Watermark Remover v2 | Clear Workflow Edition</p>
        <p>🎯 Step-by-step processing with model selection clarity</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()