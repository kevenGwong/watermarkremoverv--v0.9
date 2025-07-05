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
from core.inference import process_image, get_system_info

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
        
        # Inpainting model selection - 新的IOPaint模型
        inpaint_model = st.sidebar.selectbox(
            "Inpainting Model",
            ["iopaint", "lama"],
            index=0,  # 默认选择 IOPaint
            format_func=lambda x: "IOPaint (ZITS/MAT/FCF)" if x == "iopaint" else "LaMA (Fast)",
            help="Choose inpainting model: IOPaint supports ZITS/MAT/FCF models, LaMA for speed"
        )
        inpaint_params['inpaint_model'] = inpaint_model
        
        if inpaint_model == "iopaint":
            # IOPaint specific parameters
            st.sidebar.write("**IOPaint Model Parameters:**")
            
            # 具体模型选择
            specific_model = st.sidebar.selectbox(
                "Specific Model",
                ["auto", "zits", "mat", "fcf", "lama"],
                index=0,
                help="auto: 智能选择最佳模型 | zits: 最佳结构保持 | mat: 最佳质量 | fcf: 快速修复 | lama: 最快速度"
            )
            
            if specific_model != "auto":
                inpaint_params['force_model'] = specific_model
                inpaint_params['auto_model_selection'] = False
            else:
                inpaint_params['auto_model_selection'] = True
            
            # IOPaint通用参数
            st.sidebar.write("**Processing Parameters:**")
            
            inpaint_params['ldm_steps'] = st.sidebar.slider(
                "LDM Steps", 10, 100, 50, 5,
                help="扩散模型步数，更多步数=更高质量但更慢"
            )
            
            inpaint_params['hd_strategy'] = st.sidebar.selectbox(
                "HD Strategy",
                ["CROP", "RESIZE", "ORIGINAL"],
                index=0,
                help="高分辨率处理策略: CROP=分块处理, RESIZE=缩放处理, ORIGINAL=原图处理"
            )
            
            # 根据策略显示相关参数
            if inpaint_params['hd_strategy'] == "CROP":
                inpaint_params['hd_strategy_crop_margin'] = st.sidebar.slider(
                    "Crop Margin", 32, 128, 64, 16,
                    help="分块处理时的边距"
                )
                inpaint_params['hd_strategy_crop_trigger_size'] = st.sidebar.slider(
                    "Crop Trigger Size", 512, 2048, 1024, 64,
                    help="触发分块处理的最小尺寸"
                )
            elif inpaint_params['hd_strategy'] == "RESIZE":
                inpaint_params['hd_strategy_resize_limit'] = st.sidebar.slider(
                    "Resize Limit", 512, 2048, 2048, 64,
                    help="调整尺寸的上限"
                )
            
            # 模型选择提示
            st.sidebar.info(f"📍 **当前选择**: {specific_model}")
            if specific_model == "auto":
                st.sidebar.write("智能选择将根据图像复杂度和mask覆盖率自动选择最佳模型")
            else:
                model_descriptions = {
                    "zits": "最佳结构保持，适合复杂图像",
                    "mat": "最佳质量，适合大水印",
                    "fcf": "快速修复，平衡性能",
                    "lama": "最快速度，适合小水印"
                }
                st.sidebar.write(f"**{model_descriptions.get(specific_model, '')}**")
            
            # 模型选择变化提示
            if 'last_parameters' in st.session_state:
                last_model = st.session_state.last_parameters.get('inpaint_params', {}).get('force_model', 'auto')
                if last_model != specific_model:
                    st.sidebar.warning(f"🔄 模型已切换: {last_model} → {specific_model}")
                    st.sidebar.write("**请重新处理以查看新模型效果**")
        
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
    
    def _check_parameter_changes(self, mask_model, mask_params, inpaint_params, performance_params, transparent):
        """检查参数是否发生变化，如果有变化则清除旧结果以触发界面刷新"""
        current_params = {
            'mask_model': mask_model,
            'mask_params': mask_params.copy(),
            'inpaint_params': inpaint_params.copy(),
            'performance_params': performance_params.copy(),
            'transparent': transparent
        }
        
        # 检查是否有之前的参数状态
        if 'last_parameters' in st.session_state:
            last_params = st.session_state.last_parameters
            
            # 比较关键参数是否发生变化
            key_changes = []
            
            # 检查模型选择变化
            if last_params.get('mask_model') != current_params['mask_model']:
                key_changes.append('mask_model')
            
            # 检查inpaint模型变化
            if last_params.get('inpaint_params', {}).get('inpaint_model') != current_params['inpaint_params'].get('inpaint_model'):
                key_changes.append('inpaint_model')
            
            # 检查具体模型选择变化（IOPaint的force_model）
            if last_params.get('inpaint_params', {}).get('force_model') != current_params['inpaint_params'].get('force_model'):
                key_changes.append('specific_model')
            
            # 检查透明模式变化
            if last_params.get('transparent') != current_params['transparent']:
                key_changes.append('transparent_mode')
            
            # 如果有关键参数变化，清除旧结果
            if key_changes:
                if 'processing_result' in st.session_state:
                    del st.session_state.processing_result
                    # 显示参数变化提示
                    st.info(f"🔄 检测到参数变化: {', '.join(key_changes)}. 请重新处理以查看新结果。")
        
        # 更新当前参数状态（不包括处理结果）
        st.session_state.current_parameters = current_params
    
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
            
            # 检查参数变化，如果有变化则清除之前的结果
            self._check_parameter_changes(mask_model, mask_params, inpaint_params, performance_params, transparent)
            
            # 显示参数总结
            self._render_parameter_summary(mask_model, mask_params, inpaint_params, performance_params, transparent)
            
            # 显示当前选择的模型
            if inpaint_params.get('inpaint_model') == 'iopaint':
                selected_model = inpaint_params.get('force_model', 'auto')
                st.info(f"🎯 **当前选择的模型**: IOPaint - {selected_model.upper()}")
            else:
                st.info(f"🎯 **当前选择的模型**: {inpaint_params.get('inpaint_model', 'lama').upper()}")
            
            # 处理按钮
            self._render_process_button(inference_manager, original_image, mask_model, 
                                      mask_params, inpaint_params, performance_params, transparent)
            
            # 显示结果
            if processing_result and processing_result.success:
                self._render_results(processing_result, original_image, transparent, uploaded_file.name)
            elif processing_result and not processing_result.success:
                st.error(f"❌ Processing failed: {processing_result.error_message}")
        else:
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
                if inpaint_params.get('inpaint_model') == 'iopaint':
                    # IOPaint parameters
                    key_params = ['ldm_steps', 'hd_strategy', 'auto_model_selection']
                    for key in key_params:
                        if key in inpaint_params:
                            st.write(f"{key}: {inpaint_params[key]}")
                    
                    # Show forced model if specified
                    if inpaint_params.get('force_model'):
                        st.write(f"force_model: {inpaint_params['force_model']}")
                    
                    # Show strategy-specific params
                    if inpaint_params.get('hd_strategy') == 'CROP':
                        for key in ['hd_strategy_crop_margin', 'hd_strategy_crop_trigger_size']:
                            if key in inpaint_params:
                                st.write(f"{key}: {inpaint_params[key]}")
                    elif inpaint_params.get('hd_strategy') == 'RESIZE':
                        if 'hd_strategy_resize_limit' in inpaint_params:
                            st.write(f"hd_strategy_resize_limit: {inpaint_params['hd_strategy_resize_limit']}")
                else:
                    # LaMA parameters
                    pass
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
                # 清除之前的结果
                if 'processing_result' in st.session_state:
                    del st.session_state.processing_result
                
                with st.spinner("Processing with debug parameters..."):
                    # 使用新的模块化接口
                    try:
                        result = process_image(
                            image=original_image,
                            mask_model=mask_model,
                            mask_params=mask_params,
                            inpaint_params=inpaint_params,
                            performance_params=performance_params,
                            transparent=transparent
                        )
                        st.session_state.processing_result = result
                        # 保存当前参数状态
                        st.session_state.last_parameters = {
                            'mask_model': mask_model,
                            'mask_params': mask_params.copy(),
                            'inpaint_params': inpaint_params.copy(),
                            'performance_params': performance_params.copy(),
                            'transparent': transparent
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        return
    
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