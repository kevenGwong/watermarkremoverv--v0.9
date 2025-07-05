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
        
        # 透明模式选项
        transparent = st.sidebar.checkbox(
            "Transparent Mode",
            value=False,
            help="生成透明背景而非inpainting"
        )
        
        return mask_model, mask_params, inpaint_params, performance_params, transparent
    
    def _render_mask_section(self) -> Tuple[str, Dict[str, Any]]:
        """渲染mask生成部分"""
        st.sidebar.subheader("🎯 Mask Generation")
        mask_model = st.sidebar.selectbox(
            "Mask Model",
            ["custom", "upload"],
            index=0,
            key="mask_model_select",
            help="选择mask生成方式: custom=智能检测, upload=手动上传"
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

            # Custom model uses direct neural network inference, no prompt needed
            st.sidebar.info("ℹ️ 使用自定义分割模型直接检测水印，无需提示词")
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
            ["zits", "mat", "fcf", "lama"],
            index=0,
            key="inpaint_model_select",
            format_func=lambda x: x.upper(),
            help="Direct model selection: ZITS=结构保持, MAT=高质量, FCF=平衡, LaMA=快速"
        )
        inpaint_params['model_name'] = inpaint_model
        
        # 通用参数
        st.sidebar.write("**Processing Parameters:**")
        
        inpaint_params['ldm_steps'] = st.sidebar.slider(
            "LDM Steps", 10, 200, 50, 5,
            key="ldm_steps_slider",
            help="扩散模型步数，更多步数=更高质量但更慢"
        )
        
        inpaint_params['hd_strategy'] = st.sidebar.selectbox(
            "HD Strategy",
            ["CROP", "ORIGINAL"],
            index=0,
            key="hd_strategy_select",
            help="高分辨率处理策略: CROP=分块处理, ORIGINAL=原图处理"
        )
        
        # 根据策略显示相关参数
        if inpaint_params['hd_strategy'] == "CROP":
            inpaint_params['hd_strategy_crop_margin'] = st.sidebar.slider(
                "Crop Margin", 32, 128, 64, 16,
                key="crop_margin_slider",
                help="分块处理时的边距"
            )
            inpaint_params['hd_strategy_crop_trigger_size'] = st.sidebar.slider(
                "Crop Trigger Size", 512, 2048, 1024, 64,
                key="crop_trigger_slider",
                help="触发分块处理的最小尺寸"
            )
        else:
            # ORIGINAL策略，设置默认值
            inpaint_params['hd_strategy_crop_margin'] = 64
            inpaint_params['hd_strategy_crop_trigger_size'] = 1024
        
        # 模型描述
        model_descriptions = {
            "zits": "最佳结构保持，适合复杂图像",
            "mat": "最佳质量，适合大水印",
            "fcf": "快速修复，平衡性能",
            "lama": "最快速度，适合小水印"
        }
        st.sidebar.info(f"📍 **{inpaint_model.upper()}**: {model_descriptions.get(inpaint_model, '')}")
        
        # 设置resize limit默认值（内部使用）
        inpaint_params['hd_strategy_resize_limit'] = 1600
        
        # Common parameters
        inpaint_params['seed'] = st.sidebar.number_input(
            "Seed", -1, 999999, -1,
            key="seed_input",
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
            if last_params.get('inpaint_params', {}).get('model_name') != current_params['inpaint_params'].get('model_name'):
                key_changes.append('inpaint_model')
            
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
            if inpaint_params.get('model_name'):
                st.info(f"🎯 **当前选择的模型**: {inpaint_params.get('model_name', 'lama').upper()}")
            else:
                st.info(f"🎯 **当前选择的模型**: LAMA")
            
            # 处理按钮
            self._render_process_button(inference_manager, original_image, mask_model, 
                                      mask_params, inpaint_params, performance_params, transparent)
            
            # 显示结果
            if processing_result and processing_result.success:
                self._render_results(processing_result, original_image, uploaded_file.name, transparent)
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
                st.write(f"Model: {inpaint_params.get('model_name', 'lama')}")
                
                # Show key parameters
                key_params = ['ldm_steps', 'hd_strategy']
                for key in key_params:
                    if key in inpaint_params:
                        st.write(f"{key}: {inpaint_params[key]}")
                
                # Show strategy-specific params
                if inpaint_params.get('hd_strategy') == 'CROP':
                    for key in ['hd_strategy_crop_margin', 'hd_strategy_crop_trigger_size']:
                        if key in inpaint_params:
                            st.write(f"{key}: {inpaint_params[key]}")
                # Show remaining inpaint parameters
                for key, value in inpaint_params.items():
                    if key in ['model_name', 'ldm_steps', 'hd_strategy', 'hd_strategy_crop_margin', 'hd_strategy_crop_trigger_size']:
                        continue  # Already shown above
                    if key == 'seed' and value == -1:
                        st.write(f"{key}: Random")
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
    
    def _render_results(self, result, original_image, filename, transparent):
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