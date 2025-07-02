"""
Professional Streamlit Web UI with Advanced Parameter Control
专业版Web界面 - 包含所有高级参数设置
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
from typing import List, Optional, Dict, Any

# 配置页面
st.set_page_config(
    page_title="AI Watermark Remover Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入高级后端模块
try:
    from web_backend_advanced import AdvancedWatermarkProcessor, AdvancedProcessingResult
except ImportError as e:
    st.error(f"Failed to import advanced backend modules: {e}")
    st.error("Please ensure all dependencies are installed.")
    st.stop()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'advanced_mode' not in st.session_state:
    st.session_state.advanced_mode = False

def load_processor():
    """加载高级处理器"""
    if st.session_state.processor is None:
        with st.spinner("Loading advanced AI models..."):
            try:
                st.session_state.processor = AdvancedWatermarkProcessor("web_config_advanced.yaml")
                st.success("✅ Advanced AI models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                return False
    return True

def create_parameter_help_section():
    """创建参数帮助说明"""
    with st.expander("📚 Parameter Guide", expanded=False):
        st.markdown("""
        ### 🎯 Mask Generation Parameters
        
        **Custom Model Parameters:**
        - **Mask Threshold** (0.0-1.0): 二值化阈值，控制mask的敏感度
          - 低值 (0.3): 检测更多区域，可能包含假阳性
          - 高值 (0.7): 检测更精确，可能遗漏部分水印
        
        - **Mask Dilate Kernel Size** (1-15): 膨胀核大小，扩展检测区域
          - 小值 (1-3): 精确边界，适合清晰水印
          - 大值 (7-15): 覆盖更大范围，适合模糊水印
        
        - **Dilate Iterations** (1-5): 膨胀操作迭代次数
          - 1次: 轻微扩展
          - 3-5次: 显著扩展覆盖范围
        
        **Florence-2 Parameters:**
        - **Detection Prompt**: 检测提示词，指定要检测的目标类型
          - "watermark": 通用水印
          - "logo": 标志/徽标
          - "text overlay": 文字叠加
          - "signature": 签名
          - Custom prompts: 自定义描述
        
        - **Max BBox Percent** (1-50): 最大边界框百分比
          - 限制检测区域大小，避免误检整个图像
        
        - **Confidence Threshold** (0.1-0.9): 检测置信度阈值
          - 高值: 更保守，只检测高置信度目标
          - 低值: 更激进，可能检测到更多目标
        
        ### 🎨 LaMA Inpainting Parameters
        
        **Core Processing:**
        - **LDM Steps** (10-200): 扩散模型步数
          - 20-50: 快速处理，适合预览
          - 50-100: 平衡质量和速度
          - 100+: 最高质量，处理时间较长
        
        - **Sampler**: 采样算法
          - **DDIM**: 确定性采样，结果稳定
          - **PLMS**: 更快的收敛
          - **DPM Solver++**: 高质量采样
        
        **High Resolution Strategy:**
        - **HD Strategy**: 处理大图策略
          - **CROP**: 分块处理，保持细节
          - **RESIZE**: 缩放处理，速度更快
          - **ORIGINAL**: 原尺寸处理
        
        - **Crop Margin** (32-256): 分块重叠边距
          - 增加边距可避免分块边界痕迹
        
        - **Trigger Size** (512-2048): 触发分块处理的尺寸
        - **Resize Limit** (1024-4096): 缩放处理的最大尺寸
        
        ### 🔧 Post-Processing Parameters
        
        **Mask Refinement:**
        - **Mask Blur Radius** (0-10): mask边缘模糊
          - 软化mask边界，减少硬边
        
        - **Mask Feather Size** (0-20): mask羽化
          - 创建渐变边界，更自然的融合
        
        - **Mask Erosion/Dilation** (-10 to 10): 形态学操作
          - 负值: 收缩mask（腐蚀）
          - 正值: 扩展mask（膨胀）
        
        **Result Enhancement:**
        - **Output Sharpening** (0.0-2.0): 结果锐化
          - 增强细节，但过度可能产生伪影
        
        - **Output Denoising** (0.0-1.0): 结果降噪
          - 减少噪点，但可能降低清晰度
        
        ### 🎯 Parameter Presets
        
        - **Fast**: 快速处理，适合批量或预览
        - **Balanced**: 平衡质量和速度，推荐日常使用
        - **Quality**: 高质量处理，适合重要图片
        - **Ultra**: 极致质量，适合专业需求
        """)

def render_advanced_parameters() -> Dict[str, Any]:
    """渲染高级参数设置界面"""
    params = {}
    
    if not st.session_state.advanced_mode:
        return params
    
    st.subheader("🔧 Advanced Parameters")
    
    # 参数预设
    if st.session_state.processor:
        presets = st.session_state.processor.get_parameter_presets()
        preset_names = ["Custom"] + list(presets.keys())
        selected_preset = st.selectbox("Parameter Preset", preset_names)
        
        if selected_preset != "Custom":
            preset_params = presets[selected_preset]
            st.info(f"📋 Using {selected_preset} preset")
            for key, value in preset_params.items():
                st.text(f"  {key}: {value}")
            params.update(preset_params)
    
    # 分组显示参数
    tabs = st.tabs(["🎯 Mask Generation", "🎨 LaMA Inpainting", "🖼️ Image Processing", "🔧 Post Processing"])
    
    with tabs[0]:  # Mask Generation
        st.subheader("Mask Generation Parameters")
        
        # 自定义模型参数
        st.write("**Custom Model Settings:**")
        params['mask_threshold'] = st.slider(
            "Mask Threshold",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="二值化阈值 - 控制mask的敏感度"
        )
        
        params['mask_dilate_kernel_size'] = st.slider(
            "Mask Dilate Kernel Size",
            min_value=1, max_value=15, value=3, step=2,
            help="膨胀核大小 - 扩展检测区域"
        )
        
        params['mask_dilate_iterations'] = st.slider(
            "Dilate Iterations",
            min_value=1, max_value=5, value=1,
            help="膨胀迭代次数"
        )
        
        st.divider()
        
        # Florence-2参数
        st.write("**Florence-2 Settings:**")
        if st.session_state.processor:
            available_prompts = st.session_state.processor.get_available_prompts()
        else:
            available_prompts = ["watermark", "logo", "text overlay"]
        
        prompt_option = st.selectbox(
            "Detection Prompt",
            ["Custom"] + available_prompts,
            help="检测目标类型"
        )
        
        if prompt_option == "Custom":
            params['detection_prompt'] = st.text_input(
                "Custom Prompt",
                value="watermark",
                help="自定义检测提示词"
            )
        else:
            params['detection_prompt'] = prompt_option
        
        params['max_bbox_percent'] = st.slider(
            "Max BBox Percent",
            min_value=1.0, max_value=50.0, value=10.0, step=1.0,
            help="最大边界框百分比"
        )
        
        params['confidence_threshold'] = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=0.9, value=0.3, step=0.05,
            help="检测置信度阈值"
        )
    
    with tabs[1]:  # LaMA Inpainting
        st.subheader("LaMA Inpainting Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params['ldm_steps'] = st.slider(
                "LDM Steps",
                min_value=10, max_value=200, value=50, step=10,
                help="扩散模型步数 - 更多步数 = 更高质量"
            )
            
            params['ldm_sampler'] = st.selectbox(
                "Sampler",
                ["ddim", "plms", "dpm_solver++"],
                help="采样算法"
            )
            
            params['hd_strategy'] = st.selectbox(
                "HD Strategy",
                ["CROP", "RESIZE", "ORIGINAL"],
                help="高分辨率处理策略"
            )
        
        with col2:
            params['hd_strategy_crop_margin'] = st.slider(
                "Crop Margin",
                min_value=32, max_value=256, value=64, step=16,
                help="分块重叠边距"
            )
            
            params['hd_strategy_crop_trigger_size'] = st.slider(
                "Crop Trigger Size",
                min_value=512, max_value=2048, value=800, step=64,
                help="触发分块处理的尺寸"
            )
            
            params['hd_strategy_resize_limit'] = st.slider(
                "Resize Limit",
                min_value=1024, max_value=4096, value=1600, step=128,
                help="缩放处理的最大尺寸"
            )
    
    with tabs[2]:  # Image Processing
        st.subheader("Image Preprocessing Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params['max_input_size'] = st.slider(
                "Max Input Size",
                min_value=512, max_value=4096, value=2048, step=128,
                help="最大输入图像尺寸"
            )
            
            params['gamma_correction'] = st.slider(
                "Gamma Correction",
                min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                help="Gamma校正"
            )
        
        with col2:
            params['contrast_enhancement'] = st.slider(
                "Contrast Enhancement",
                min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                help="对比度增强"
            )
    
    with tabs[3]:  # Post Processing
        st.subheader("Post Processing Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Mask Refinement:**")
            params['mask_blur_radius'] = st.slider(
                "Mask Blur Radius",
                min_value=0, max_value=10, value=0,
                help="mask边缘模糊"
            )
            
            params['mask_feather_size'] = st.slider(
                "Mask Feather Size",
                min_value=0, max_value=20, value=0,
                help="mask羽化大小"
            )
            
            params['mask_erosion_size'] = st.slider(
                "Mask Erosion/Dilation",
                min_value=-10, max_value=10, value=0,
                help="形态学操作（负值=腐蚀，正值=膨胀）"
            )
        
        with col2:
            st.write("**Result Enhancement:**")
            params['output_sharpening'] = st.slider(
                "Output Sharpening",
                min_value=0.0, max_value=2.0, value=0.0, step=0.1,
                help="输出锐化"
            )
            
            params['output_denoising'] = st.slider(
                "Output Denoising",
                min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                help="输出降噪"
            )
    
    return params

def visualize_advanced_results(result: AdvancedProcessingResult, show_intermediate: bool = False):
    """可视化高级处理结果"""
    if not result.success:
        st.error(f"❌ Processing failed: {result.error_message}")
        return
    
    # 主要结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Generated Mask")
        if result.mask_image:
            st.image(result.mask_image, use_column_width=True)
            
            # Mask统计信息
            mask_array = np.array(result.mask_image)
            white_pixels = np.sum(mask_array > 128)
            total_pixels = mask_array.size
            st.metric("White Pixels Ratio", f"{white_pixels/total_pixels*100:.1f}%")
    
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
    
    # 中间结果（Debug模式）
    if show_intermediate and result.intermediate_results:
        st.subheader("🔍 Intermediate Results")
        
        intermediate_cols = st.columns(len(result.intermediate_results))
        for idx, (name, img) in enumerate(result.intermediate_results.items()):
            with intermediate_cols[idx % len(intermediate_cols)]:
                st.text(name.replace('_', ' ').title())
                st.image(img, use_column_width=True)
    
    # 使用的参数
    if st.session_state.debug_mode and result.parameters_used:
        with st.expander("🔧 Parameters Used"):
            st.json(result.parameters_used)

def create_download_section(result: AdvancedProcessingResult, filename_prefix: str = "processed"):
    """创建下载区域"""
    if not result.success or not result.result_image:
        return
    
    st.subheader("📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    formats = ["PNG", "WEBP", "JPG"]
    for idx, fmt in enumerate(formats):
        with [col1, col2, col3][idx]:
            img_buffer = io.BytesIO()
            
            if fmt == "PNG":
                result.result_image.save(img_buffer, format="PNG")
                mime = "image/png"
            elif fmt == "WEBP":
                result.result_image.save(img_buffer, format="WEBP", quality=95)
                mime = "image/webp"
            else:  # JPG
                img = result.result_image
                if img.mode == "RGBA":
                    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1] if len(img.split()) == 4 else None)
                    img = rgb_img
                img.save(img_buffer, format="JPEG", quality=95)
                mime = "image/jpeg"
            
            img_buffer.seek(0)
            
            st.download_button(
                label=f"📥 {fmt}",
                data=img_buffer.getvalue(),
                file_name=f"{filename_prefix}.{fmt.lower()}",
                mime=mime,
                use_container_width=True
            )

def main():
    """主应用函数"""
    
    # 标题和模式切换
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("🎨 AI Watermark Remover Pro")
        st.markdown("**Professional edition with advanced parameter control**")
    
    with col2:
        st.session_state.advanced_mode = st.toggle(
            "⚙️ Advanced Mode",
            value=st.session_state.advanced_mode,
            help="Enable advanced parameter control"
        )
    
    with col3:
        st.session_state.debug_mode = st.toggle(
            "🔧 Debug Mode",
            value=st.session_state.debug_mode,
            help="Show detailed processing information"
        )
    
    # 加载处理器
    if not load_processor():
        return
    
    # 参数帮助
    create_parameter_help_section()
    
    # 侧边栏基础设置
    with st.sidebar:
        st.header("⚙️ Basic Settings")
        
        processing_mode = st.radio(
            "Processing Mode",
            ["Single Image", "Custom Mask", "Batch Processing"],
            help="Choose processing mode"
        )
        
        transparent = st.checkbox(
            "Make watermark transparent",
            value=False,
            help="Make watermark regions transparent instead of removing them"
        )
        
        if transparent:
            preview_bg = st.selectbox(
                "Preview background",
                ["white", "black", "checkered"],
                help="Background for transparent preview"
            )
        
        # 系统信息
        if st.button("🔄 System Info"):
            sys_info = st.session_state.processor.get_advanced_system_info()
            st.json(sys_info)
    
    # 主处理区域
    if processing_mode == "Single Image":
        st.header("📸 Professional Single Image Processing")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload an image with watermarks to remove"
        )
        
        if uploaded_file is not None:
            # 显示原图
            original_image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_column_width=True)
                st.text(f"Size: {original_image.size[0]} × {original_image.size[1]}")
            
            # 高级参数设置
            advanced_params = render_advanced_parameters()
            
            # 处理按钮
            if st.button("🚀 Process with Advanced Settings", type="primary"):
                with st.spinner("Processing with advanced settings..."):
                    result = st.session_state.processor.process_image(
                        image=original_image,
                        transparent=transparent,
                        advanced_params=advanced_params
                    )
                
                with col2:
                    st.subheader("Processing Results")
                    visualize_advanced_results(result, st.session_state.debug_mode)
                
                # 下载区域
                create_download_section(result, "watermark_removed_pro")
    
    elif processing_mode == "Custom Mask":
        st.header("🎯 Professional Custom Mask Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            uploaded_image = st.file_uploader(
                "Choose image file",
                type=['jpg', 'jpeg', 'png', 'webp'],
                key="custom_image_pro"
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
                key="custom_mask_pro"
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
            # 高级参数设置
            advanced_params = render_advanced_parameters()
            
            if st.button("🚀 Process with Custom Mask", type="primary"):
                image = Image.open(uploaded_image)
                mask = Image.open(uploaded_mask).convert('L')
                
                # 确保尺寸匹配
                if image.size != mask.size:
                    mask = mask.resize(image.size, Image.LANCZOS)
                    st.warning(f"⚠️ Mask resized to match image size: {image.size}")
                
                # 使用自定义mask（跳过mask生成步骤）
                with st.spinner("Processing with custom mask..."):
                    if transparent:
                        # 直接应用透明效果
                        result_img = st.session_state.processor._apply_transparent_effect(image, mask)
                    else:
                        # 使用LaMA修复
                        result_img = st.session_state.processor._process_with_advanced_lama(
                            image, mask, **advanced_params
                        )
                    
                    # 创建结果对象
                    from web_backend_advanced import AdvancedProcessingResult
                    result = AdvancedProcessingResult(
                        success=True,
                        result_image=result_img,
                        mask_image=mask,
                        parameters_used=advanced_params
                    )
                
                # 显示结果
                st.subheader("🎉 Professional Results")
                visualize_advanced_results(result, st.session_state.debug_mode)
                
                # 下载区域
                create_download_section(result, "custom_mask_pro")
    
    else:  # Batch Processing
        st.header("📁 Professional Batch Processing")
        st.info("Upload multiple images for batch processing with advanced settings")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.info(f"📊 {len(uploaded_files)} files uploaded")
            
            # 高级参数设置
            advanced_params = render_advanced_parameters()
            
            if st.button("🚀 Batch Process with Advanced Settings", type="primary"):
                # 批处理逻辑
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                    
                    image = Image.open(uploaded_file)
                    result = st.session_state.processor.process_image(
                        image=image,
                        transparent=transparent,
                        advanced_params=advanced_params
                    )
                    results.append((uploaded_file.name, result))
                    
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                
                status_text.text("✅ Batch processing completed!")
                
                # 显示批处理结果摘要
                successful = len([r for _, r in results if r.success])
                failed = len(results) - successful
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Processed", len(results))
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", failed)
                
                # 批量下载（简化版）
                if successful > 0:
                    st.info("💡 Individual download links available in detailed results below")
                
                # 详细结果展示
                st.subheader("📋 Detailed Batch Results")
                for filename, result in results:
                    with st.expander(f"{filename} - {'✅' if result.success else '❌'}"):
                        if result.success:
                            visualize_advanced_results(result, False)
                            create_download_section(result, f"{Path(filename).stem}_pro")
                        else:
                            st.error(f"Error: {result.error_message}")
    
    # 页脚
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🎨 AI Watermark Remover Pro | Professional Edition</p>
        <p>🔧 Advanced parameter control for professional watermark removal</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()