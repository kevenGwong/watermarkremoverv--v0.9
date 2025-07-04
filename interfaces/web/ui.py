"""
UI展示模块
负责Streamlit界面展示、参数控制和结果展示
"""

import streamlit as st
import numpy as np
import io
from pathlib import Path
from typing import Dict, Any, Tuple
from streamlit_image_comparison import image_comparison
from PIL import Image

from config.config import ConfigManager
from core.utils.image_utils import ImageProcessor, ImageDownloader, ImageValidator

class ParameterPanel:
    """参数面板"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def render(self) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], bool]:
        """渲染参数面板"""
        st.sidebar.header("🔬 Debug Parameters")
        
        # Mask模型选择
        mask_model, mask_params = self._render_mask_section()
        
        st.sidebar.divider()
        
        # Inpainting参数
        inpaint_params = self._render_inpaint_section()
        
        st.sidebar.divider()
        
        # 性能选项
        performance_params = self._render_performance_section()
        
        # 处理模式
        transparent = self._render_processing_mode()
        
        return mask_model, mask_params, inpaint_params, performance_params, transparent
    
    def _render_mask_section(self) -> Tuple[str, Dict[str, Any]]:
        """渲染mask生成部分"""
        st.sidebar.subheader("🎯 Mask Generation")
        mask_model = st.sidebar.selectbox(
            "Mask Model",
            ["custom", "florence2", "upload"],
            format_func=lambda x: "Custom Watermark" if x == "custom" else "Florence-2" if x == "florence2" else "Upload Custom Mask"
        )
        
        # Mask参数
        mask_params = {}
        if mask_model == "custom":
            st.sidebar.write("**Custom Model Parameters:**")
            mask_params['mask_threshold'] = st.sidebar.slider(
                "Mask Threshold", 0.0, 1.0, 0.5, 0.05,
                help="二值化阈值，控制检测敏感度"
            )
            mask_params['mask_dilate_kernel_size'] = st.sidebar.slider(
                "Dilate Kernel Size", 1, 50, 3, 2,
                help="膨胀核大小，扩展检测区域"
            )
            mask_params['mask_dilate_iterations'] = st.sidebar.slider(
                "Dilate Iterations", 1, 20, 1,
                help="膨胀迭代次数"
            )
        elif mask_model == "florence2":
            st.sidebar.write("**Florence-2 Parameters:**")
            mask_params['detection_prompt'] = st.sidebar.selectbox(
                "Detection Prompt",
                ["watermark", "logo", "text overlay", "signature", "copyright mark"],
                help="检测目标类型描述"
            )
            mask_params['max_bbox_percent'] = st.sidebar.slider(
                "Max BBox Percent", 1.0, 50.0, 10.0, 1.0,
                help="最大检测区域百分比"
            )
            mask_params['confidence_threshold'] = st.sidebar.slider(
                "Confidence Threshold", 0.1, 0.9, 0.3, 0.05,
                help="检测置信度阈值"
            )
        else:  # upload
            st.sidebar.write("**Upload Custom Mask:**")
            uploaded_mask = st.sidebar.file_uploader(
                "📂 Upload Mask File",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a binary mask (black background, white watermark regions)",
                key="mask_upload"
            )
            mask_params['uploaded_mask'] = uploaded_mask
            
            if uploaded_mask:
                st.sidebar.success("✅ Custom mask uploaded")
                # 可选的后处理参数
                mask_params['mask_dilate_kernel_size'] = st.sidebar.slider(
                    "Additional Dilate", 0, 20, 5, 1,
                    help="额外膨胀处理（0=不处理，建议5-10增强修复效果）"
                )
                mask_params['mask_dilate_iterations'] = st.sidebar.slider(
                    "Dilate Iterations", 1, 5, 2, 1,
                    help="膨胀迭代次数（更多次数=更大区域）"
                )
            else:
                st.sidebar.warning("⚠️ Please upload a mask file")
        
        return mask_model, mask_params
    
    def _render_inpaint_section(self) -> Dict[str, Any]:
        """渲染inpainting参数部分"""
        st.sidebar.subheader("🎨 Inpainting Parameters")
        inpaint_params = {}
        
        # Inpainting model selection
        inpaint_model = st.sidebar.selectbox(
            "Inpainting Model",
            ["powerpaint", "lama"],  # 将 PowerPaint 设为第一个选项
            index=0,  # 默认选择 PowerPaint
            format_func=lambda x: "PowerPaint (Object Removal)" if x == "powerpaint" else "LaMA (Fast)",
            help="Choose inpainting model: PowerPaint for object removal, LaMA for speed"
        )
        inpaint_params['inpaint_model'] = inpaint_model
        
        if inpaint_model == "powerpaint":
            # PowerPaint specific parameters
            st.sidebar.write("**PowerPaint Object Removal Parameters:**")
            
            # Task selection for PowerPaint
            task_type = st.sidebar.selectbox(
                "PowerPaint Task",
                ["object-removal", "text-guided"],
                format_func=lambda x: "Object Removal" if x == "object-removal" else "Text Guided",
                help="PowerPaint task type: Object removal for watermarks, Text guided for custom content"
            )
            inpaint_params['task'] = task_type
            
            if task_type == "object-removal":
                # Object removal specific prompts
                inpaint_params['prompt'] = st.sidebar.text_area(
                    "Positive Prompt",
                    value="empty scene blur, clean background, natural environment",
                    help="Describe the background to fill the removed area"
                )
                
                inpaint_params['negative_prompt'] = st.sidebar.text_area(
                    "Negative Prompt",
                    value="object, person, animal, vehicle, building, text, watermark, logo, worst quality, low quality, normal quality, bad quality, blurry, artifacts",
                    help="Describe what to avoid (objects to remove)"
                )
                
                st.sidebar.info("🎯 **Object Removal Mode**: Optimized for removing watermarks and objects")
                
            else:  # text-guided
                inpaint_params['prompt'] = st.sidebar.text_area(
                    "Positive Prompt",
                    value="high quality, detailed, clean, professional photo",
                    help="Describe the desired result quality and characteristics"
                )
                
                inpaint_params['negative_prompt'] = st.sidebar.text_area(
                    "Negative Prompt",
                    value="watermark, logo, text, signature, blurry, low quality, artifacts, distorted, deformed",
                    help="Describe what to avoid in the result"
                )
            
            inpaint_params['num_inference_steps'] = st.sidebar.slider(
                "Inference Steps", 10, 100, 50, 5,
                help="More steps = higher quality but slower"
            )
            
            inpaint_params['guidance_scale'] = st.sidebar.slider(
                "Guidance Scale", 1.0, 20.0, 7.5, 0.5,
                help="Higher values = more prompt adherence (recommend 7.5+ for object removal)"
            )
            
            inpaint_params['strength'] = st.sidebar.slider(
                "Strength", 0.1, 1.0, 1.0, 0.05,
                help="How much to change the masked area (1.0 = full change)"
            )
            
            # High-resolution processing
            st.sidebar.write("**High-Resolution Processing:**")
            
            inpaint_params['crop_trigger_size'] = st.sidebar.slider(
                "Crop Trigger Size", 256, 1024, 512, 64,
                help="Images larger than this will use crop strategy"
            )
            
            inpaint_params['crop_margin'] = st.sidebar.slider(
                "Crop Margin", 32, 128, 64, 16,
                help="Extra margin around mask regions when cropping"
            )
            
            inpaint_params['resize_to_512'] = st.sidebar.checkbox(
                "Resize crops to 512px", True,
                help="Resize crop regions to optimal size for SD1.5"
            )
            
            inpaint_params['blend_edges'] = st.sidebar.checkbox(
                "Blend edges", True,
                help="Smooth blending of processed regions"
            )
            
            if inpaint_params['blend_edges']:
                inpaint_params['edge_feather'] = st.sidebar.slider(
                    "Edge Feather", 1, 15, 5, 1,
                    help="Edge feathering strength for blending"
                )
            else:
                inpaint_params['edge_feather'] = 0
        
        else:
            # LaMA specific parameters
            st.sidebar.write("**LaMA Parameters:**")
            
            inpaint_params['prompt'] = ""  # LaMA doesn't use prompts
            st.sidebar.info("ℹ️ LaMA model doesn't use text prompts")
            
            inpaint_params['ldm_steps'] = st.sidebar.slider(
                "LDM Steps", 10, 200, 50, 10,
                help="扩散模型步数，更多步数=更高质量"
            )
            
            inpaint_params['ldm_sampler'] = st.sidebar.selectbox(
                "LDM Sampler",
                ["ddim", "plms"],
                help="采样器选择"
            )
            
            inpaint_params['hd_strategy'] = st.sidebar.selectbox(
                "HD Strategy",
                ["CROP", "RESIZE", "ORIGINAL"],
                help="高分辨率处理策略"
            )
            
            # 只在CROP策略下显示Crop Margin
            if inpaint_params['hd_strategy'] == "CROP":
                inpaint_params['hd_strategy_crop_margin'] = st.sidebar.slider(
                    "Crop Margin", 32, 256, 64, 16,
                    help="分块处理边距"
                )
                inpaint_params['hd_strategy_crop_trigger_size'] = st.sidebar.slider(
                    "Crop Trigger Size", 512, 2048, 800, 64,
                    help="触发分块处理的最小尺寸"
                )
            else:
                # 为其他策略设置默认值
                inpaint_params['hd_strategy_crop_margin'] = 64
                inpaint_params['hd_strategy_crop_trigger_size'] = 800
            
            # 只在RESIZE策略下显示Resize Limit
            if inpaint_params['hd_strategy'] == "RESIZE":
                inpaint_params['hd_strategy_resize_limit'] = st.sidebar.slider(
                    "Resize Limit", 512, 2048, 1600, 64,
                    help="调整尺寸上限"
                )
            else:
                inpaint_params['hd_strategy_resize_limit'] = 1600
        
        # Common parameters
        inpaint_params['seed'] = st.sidebar.number_input(
            "Seed", -1, 999999, -1,
            help="随机种子（-1为随机）"
        )
        
        return inpaint_params
    
    def _render_performance_section(self) -> Dict[str, Any]:
        """渲染性能选项部分"""
        st.sidebar.subheader("⚡ Performance Options")
        performance_params = {}
        
        performance_params['mixed_precision'] = st.sidebar.checkbox(
            "Mixed Precision",
            value=True,
            help="混合精度计算，提升速度"
        )
        
        performance_params['log_processing_time'] = st.sidebar.checkbox(
            "Log Processing Time",
            value=True,
            help="记录详细处理时间"
        )
        
        return performance_params
    
    def _render_processing_mode(self) -> bool:
        """渲染处理模式部分"""
        st.sidebar.divider()
        st.sidebar.subheader("🔧 Processing Mode")
        
        transparent = st.sidebar.checkbox(
            "Transparent Mode",
            value=False,
            help="创建透明区域而不是填充修复"
        )
        
        return transparent

class MainInterface:
    """主界面"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.parameter_panel = ParameterPanel(config_manager)
    
    def render(self, inference_manager, processing_result=None):
        """渲染主界面"""
        st.title("🔬 AI Watermark Remover - Debug Edition")
        st.markdown("**Parameter Control & Real-time Comparison**")
        
        # 图片上传
        uploaded_file = st.file_uploader(
            "📸 Upload Image for Debug",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload an image with watermarks to debug removal process"
        )
        
        if uploaded_file is not None:
            # 保存原始图片
            original_image = Image.open(uploaded_file)
            st.session_state.original_image = original_image
            
            # 获取参数
            mask_model, mask_params, inpaint_params, performance_params, transparent = self.parameter_panel.render()
            
            # 显示参数总结
            self._render_parameter_summary(mask_model, mask_params, inpaint_params, performance_params, transparent)
            
            # 处理按钮
            self._render_process_button(inference_manager, original_image, mask_model, 
                                      mask_params, inpaint_params, performance_params, transparent)
            
            # 显示结果
            if processing_result and processing_result.success:
                self._render_results(processing_result, original_image, transparent, uploaded_file.name)
            elif processing_result and not processing_result.success:
                st.error(f"❌ Processing failed: {processing_result.error_message}")
        else:
            # 显示参数面板但不处理
            self.parameter_panel.render()
            
            # 显示使用说明
            st.info("📸 Please upload an image to start debugging watermark removal parameters.")
            self._render_usage_guide()
    
    def _render_parameter_summary(self, mask_model, mask_params, inpaint_params, performance_params, transparent):
        """渲染参数总结"""
        with st.expander("📋 Current Parameters Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Mask Generation:**")
                st.write(f"Model: {mask_model}")
                for key, value in mask_params.items():
                    st.write(f"{key}: {value}")
            
            with col2:
                st.write("**Inpainting:**")
                st.write(f"Model: {inpaint_params.get('inpaint_model', 'lama')}")
                
                # Show relevant parameters based on model type
                if inpaint_params.get('inpaint_model') == 'powerpaint':
                    # PowerPaint parameters
                    key_params = ['num_inference_steps', 'guidance_scale', 'strength', 
                                'crop_trigger_size', 'crop_margin', 'seed']
                    for key in key_params:
                        if key in inpaint_params:
                            st.write(f"{key}: {inpaint_params[key]}")
                    
                    if inpaint_params.get('prompt'):
                        st.write(f"prompt: {inpaint_params['prompt'][:30]}...")
                    if inpaint_params.get('negative_prompt'):
                        st.write(f"negative_prompt: {inpaint_params['negative_prompt'][:30]}...")
                else:
                    # LaMA parameters
                    for key, value in inpaint_params.items():
                        if key in ['inpaint_model', 'prompt', 'negative_prompt']:
                            continue
                        if key == 'prompt' and not value:
                            continue
                        # 特殊处理策略相关参数显示
                        if key.startswith('hd_strategy_') and inpaint_params.get('hd_strategy') == 'ORIGINAL':
                            if key == 'hd_strategy_crop_margin' or key == 'hd_strategy_crop_trigger_size':
                                st.write(f"{key}: {value} *(不适用于ORIGINAL策略)*")
                            elif key == 'hd_strategy_resize_limit':
                                st.write(f"{key}: {value} *(不适用于ORIGINAL策略)*")
                            else:
                                st.write(f"{key}: {value}")
                        elif key.startswith('hd_strategy_') and inpaint_params.get('hd_strategy') == 'RESIZE':
                            if key == 'hd_strategy_crop_margin' or key == 'hd_strategy_crop_trigger_size':
                                st.write(f"{key}: {value} *(不适用于RESIZE策略)*")
                            else:
                                st.write(f"{key}: {value}")
                        elif key.startswith('hd_strategy_') and inpaint_params.get('hd_strategy') == 'CROP':
                            if key == 'hd_strategy_resize_limit':
                                st.write(f"{key}: {value} *(不适用于CROP策略)*")
                            else:
                                st.write(f"{key}: {value}")
                        else:
                            st.write(f"{key}: {value}")
            
            with col3:
                st.write("**Performance:**")
                st.write(f"Mode: {'Transparent' if transparent else 'Inpaint'}")
                for key, value in performance_params.items():
                    st.write(f"{key}: {value}")
    
    def _render_process_button(self, inference_manager, original_image, mask_model, 
                             mask_params, inpaint_params, performance_params, transparent):
        """渲染处理按钮"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process with Debug Parameters", type="primary", use_container_width=True):
                with st.spinner("Processing with debug parameters..."):
                    # 确保使用正确的处理器
                    if hasattr(inference_manager, 'enhanced_processor') and inference_manager.enhanced_processor is not None:
                        processor = inference_manager
                    else:
                        st.error("❌ Processor not loaded. Please refresh the page.")
                        return
                    
                    result = processor.process_image(
                        image=original_image,
                        mask_model=mask_model,
                        mask_params=mask_params,
                        inpaint_params=inpaint_params,
                        performance_params=performance_params,
                        transparent=transparent
                    )
                    st.session_state.processing_result = result
                    st.rerun()
    
    def _render_results(self, result, original_image, transparent, filename):
        """渲染结果"""
        st.subheader("🔄 Before vs After Comparison")
        
        # 图像对比
        if result.result_image:
            # 确保两个图像尺寸一致
            original_display = original_image
            result_display = result.result_image
            
            if original_display.size != result_display.size:
                result_display = ImageProcessor.resize_image(result_display, original_display.size)
            
            # 如果是透明图像，提供背景选择
            if result_display.mode == 'RGBA':
                bg_color = st.selectbox(
                    "Preview Background", 
                    ["white", "black", "checkered"],
                    key="comparison_bg"
                )
                
                if bg_color != "white":
                    result_display = ImageProcessor.add_background(result_display, bg_color)
            
            # 使用image_comparison组件
            image_comparison(
                img1=original_display,
                img2=result_display,
                label1="Original",
                label2="Processed",
                width=800,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )
        
        # 详细结果信息
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{result.processing_time:.2f}s")
        
        with col2:
            if result.mask_image:
                coverage = ImageProcessor.calculate_mask_coverage(result.mask_image)
                st.metric("Mask Coverage", f"{coverage:.1f}%")
        
        with col3:
            st.metric("Output Mode", "Transparent" if transparent else "Inpaint")
        
        # 显示生成的mask
        if result.mask_image:
            self._render_mask_section(result.mask_image, filename)
        else:
            st.warning("⚠️ No mask was generated")
        
        # 下载选项
        if result.result_image:
            st.subheader("📥 Download Results")
            self._render_download_section(result.result_image, filename)
        else:
            st.warning("⚠️ No result image available for download")
    
    def _render_mask_section(self, mask_image, filename):
        """渲染mask部分"""
        with st.expander("🎭 Generated Mask", expanded=False):
            col_mask1, col_mask2 = st.columns([3, 1])
            with col_mask1:
                st.image(mask_image, caption=f"Detected watermark mask ({mask_image.size[0]}×{mask_image.size[1]})", use_column_width=True)
            with col_mask2:
                # 添加mask下载按钮
                mask_buffer = io.BytesIO()
                mask_image.save(mask_buffer, format="PNG")
                mask_buffer.seek(0)
                st.download_button(
                    label="📥 Download Mask",
                    data=mask_buffer.getvalue(),
                    file_name=f"{Path(filename).stem}_mask.png",
                    mime="image/png",
                    use_container_width=True,
                    help=f"Download original resolution mask ({mask_image.size[0]}×{mask_image.size[1]})"
                )
    
    def _render_download_section(self, result_image, filename):
        """渲染下载部分"""
        filename_base = ImageDownloader.get_filename_base(filename)
        download_info = ImageDownloader.create_download_info(result_image, filename_base)
        
        col1, col2, col3 = st.columns(3)
        
        for idx, info in enumerate(download_info):
            with [col1, col2, col3][idx]:
                st.download_button(
                    label=f"📥 {info['format']}",
                    data=info['data'],
                    file_name=info['filename'],
                    mime=info['mime_type'],
                    use_container_width=True
                )
    
    def _render_usage_guide(self):
        """渲染使用指南"""
        with st.expander("💡 Debug Mode Guide", expanded=True):
            st.markdown("""
            ### 🔬 Debug Features:
            
            **Mask Generation Control:**
            - Choose between Custom and Florence-2 models
            - Adjust threshold, dilation, and detection parameters
            - Real-time parameter feedback
            
            **Inpainting Parameters:**
            - Control LDM steps and sampler
            - HD strategy for large images
            - Seed control for reproducible results
            
            **Performance Options:**
            - Mixed precision for speed
            - Processing time logging
            
            **Real-time Comparison:**
            - Side-by-side before/after view
            - Interactive slider comparison
            - Mask visualization
            """) 