"""
AI Watermark Remover - Debug Edition with Parameter Control
调试版本 - 左侧参数控制，右侧对比显示
"""
import streamlit as st
import time
import io
import random
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import Optional, Dict, Any, Union
from streamlit_image_comparison import image_comparison

# 配置页面
st.set_page_config(
    page_title="AI Watermark Remover - Debug",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入后端模块
try:
    from web_backend import WatermarkProcessor, ProcessingResult
except ImportError as e:
    st.error(f"Failed to import backend modules: {e}")
    st.stop()

# 配置日志 - 在debug app中使用更简洁的格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 初始化session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

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

def create_enhanced_backend():
    """创建增强的后端处理器，支持更多参数"""
    class EnhancedWatermarkProcessor:
        def __init__(self, base_processor):
            self.base_processor = base_processor
        
        def process_image_with_params(self, 
                                    image: Union[Image.Image, bytes],
                                    mask_model: str,
                                    mask_params: Dict[str, Any],
                                    inpaint_params: Dict[str, Any],
                                    performance_params: Dict[str, Any],
                                    transparent: bool = False) -> ProcessingResult:
            """使用详细参数处理图像"""
            import time
            import logging
            logger = logging.getLogger(__name__)
            
            start_time = time.time()
            
            logger.info("🚀 开始增强处理流程...")
            logger.info(f"📸 输入图像类型: {type(image)}")
            logger.info(f"🎭 Mask模型: {mask_model}")
            logger.info(f"🎛️ Mask参数: {mask_params}")
            logger.info(f"⚙️ Inpaint参数: {inpaint_params}")
            logger.info(f"🔧 Performance参数: {performance_params}")
            logger.info(f"🫥 透明模式: {transparent}")
            
            try:
                # 处理输入图像
                if isinstance(image, bytes):
                    # 字节输入 - 直接传递给base processor，让它用OpenCV处理
                    image_for_processing = image
                    # 为mask生成创建PIL版本
                    import cv2
                    bytes_array = np.asarray(bytearray(image), dtype=np.uint8)
                    image_bgr = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb)
                    logger.info(f"📁 字节输入处理完成: PIL size={image_pil.size}")
                else:
                    # PIL输入
                    image_for_processing = image
                    image_pil = image
                    logger.info(f"🖼️ PIL输入: size={image_pil.size}, mode={image_pil.mode}")
                
                # 根据选择的模型生成mask
                logger.info(f"🎯 开始生成mask，模型: {mask_model}")
                if mask_model == "custom":
                    mask_image = self._generate_custom_mask(image_pil, mask_params)
                elif mask_model == "florence2":
                    mask_image = self._generate_florence_mask(image_pil, mask_params)
                else:  # upload
                    mask_image = self._generate_uploaded_mask(image_pil, mask_params)
                
                logger.info(f"✅ Mask生成完成: size={mask_image.size}, mode={mask_image.mode}")
                
                # 验证mask有效性
                mask_array = np.array(mask_image)
                white_pixels = np.sum(mask_array > 128)
                total_pixels = mask_array.size
                mask_coverage = white_pixels / total_pixels * 100
                logger.info(f"🔍 Mask验证: 覆盖率={mask_coverage:.2f}%, 白色像素={white_pixels}")
                
                # 应用处理
                if transparent:
                    logger.info("🫥 应用透明处理...")
                    result_image = self._apply_transparency(image_pil, mask_image)
                else:
                    logger.info("🎨 应用Inpainting处理...")
                    result_image = self._apply_inpainting(image_for_processing, mask_image, inpaint_params)
                
                processing_time = time.time() - start_time
                logger.info(f"⏱️ 处理完成，耗时: {processing_time:.2f}秒")
                
                return ProcessingResult(
                    success=True,
                    result_image=result_image,
                    mask_image=mask_image,
                    processing_time=processing_time
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                return ProcessingResult(
                    success=False,
                    error_message=str(e),
                    processing_time=processing_time
                )
        
        def _generate_custom_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
            """生成自定义mask"""
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info(f"🎯 生成Custom Mask，参数: {params}")
            logger.info(f"🖼️ 输入图像: size={image.size}, mode={image.mode}")
            
            # 更新自定义模型参数
            if hasattr(self.base_processor, 'mask_generator') and hasattr(self.base_processor.mask_generator, 'generate_mask'):
                generator = self.base_processor.mask_generator
                # 动态更新参数
                generator.mask_threshold = params.get('mask_threshold', 0.5)
                logger.info(f"🔧 更新mask_threshold: {generator.mask_threshold}")
                
                # 生成mask
                mask = generator.generate_mask(image)
                logger.info(f"🎭 原始mask生成: size={mask.size}, mode={mask.mode}")
                
                # 应用膨胀参数
                dilate_size = params.get('mask_dilate_kernel_size', 3)
                dilate_iterations = params.get('mask_dilate_iterations', 1)
                logger.info(f"🔍 膨胀参数: kernel_size={dilate_size}, iterations={dilate_iterations}")
                
                if dilate_size > 0:
                    import cv2
                    mask_array = np.array(mask)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
                    mask_array = cv2.dilate(mask_array, kernel, iterations=dilate_iterations)
                    mask = Image.fromarray(mask_array, mode='L')
                    logger.info(f"🔍 膨胀后mask: size={mask.size}")
                
                # 验证mask内容
                mask_array = np.array(mask)
                white_pixels = np.sum(mask_array > 128)
                total_pixels = mask_array.size
                coverage = white_pixels / total_pixels * 100
                logger.info(f"✅ Custom Mask覆盖率: {coverage:.2f}% ({white_pixels}/{total_pixels})")
                
                return mask
            else:
                logger.warning("⚠️ Custom mask generator不可用，使用备用方案")
                # 确保使用修复后的CustomMaskGenerator逻辑
                # 直接调用基础处理器的process_image，它会调用修复后的mask_generator
                result = self.base_processor.process_image(
                    image=image,
                    transparent=True,
                    max_bbox_percent=10.0,
                    force_format="PNG"
                )
                return result.mask_image if result.mask_image else Image.new('L', image.size, 0)
        
        def _generate_uploaded_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
            """处理上传的mask"""
            import logging
            logger = logging.getLogger(__name__)
            
            uploaded_mask = params.get('uploaded_mask')
            if not uploaded_mask:
                raise ValueError("No mask file uploaded")
            
            logger.info(f"📂 处理上传的mask文件: {uploaded_mask.name}")
            
            # 读取上传的mask
            mask = Image.open(uploaded_mask)
            logger.info(f"🎭 原始上传mask: size={mask.size}, mode={mask.mode}")
            
            # 确保mask是灰度图像
            if mask.mode != 'L':
                mask = mask.convert('L')
                logger.info(f"🔄 转换为灰度模式: mode={mask.mode}")
            
            # 调整mask尺寸以匹配图像
            if mask.size != image.size:
                logger.info(f"📏 调整mask尺寸: {mask.size} → {image.size}")
                mask = mask.resize(image.size, Image.LANCZOS)
            
            # 验证mask内容
            mask_array = np.array(mask)
            white_pixels = np.sum(mask_array > 128)
            total_pixels = mask_array.size
            coverage = white_pixels / total_pixels * 100
            logger.info(f"🔍 上传mask验证: 覆盖率={coverage:.2f}%, 白色像素={white_pixels}")
            
            # 应用额外的膨胀处理（如果需要）
            dilate_size = params.get('mask_dilate_kernel_size', 0)
            if dilate_size > 0:
                import cv2
                logger.info(f"🔍 应用膨胀处理: kernel_size={dilate_size}")
                mask_array = np.array(mask)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
                mask_array = cv2.dilate(mask_array, kernel, iterations=1)
                mask = Image.fromarray(mask_array, mode='L')
                
                # 膨胀后验证
                mask_array = np.array(mask)
                white_pixels_after = np.sum(mask_array > 128)
                coverage_after = white_pixels_after / total_pixels * 100
                logger.info(f"🔍 膨胀后验证: 覆盖率={coverage_after:.2f}%, 白色像素={white_pixels_after}")
            
            return mask
        
        def _generate_florence_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
            """生成Florence-2 mask"""
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info(f"🤖 生成Florence-2 Mask，参数: {params}")
            logger.info(f"🖼️ 输入图像: size={image.size}, mode={image.mode}")
            
            # 使用基础处理器，但传递参数
            max_bbox_percent = params.get('max_bbox_percent', 10.0)
            detection_prompt = params.get('detection_prompt', 'watermark')
            confidence_threshold = params.get('confidence_threshold', 0.3)
            
            logger.info(f"🎯 Florence-2参数:")
            logger.info(f"   max_bbox_percent: {max_bbox_percent}")
            logger.info(f"   detection_prompt: {detection_prompt}")
            logger.info(f"   confidence_threshold: {confidence_threshold}")
            
            result = self.base_processor.process_image(
                image=image,
                transparent=True,
                max_bbox_percent=max_bbox_percent,
                detection_prompt=detection_prompt,
                force_format="PNG"
            )
            
            if result.mask_image:
                # 验证mask内容
                mask_array = np.array(result.mask_image)
                white_pixels = np.sum(mask_array > 128)
                total_pixels = mask_array.size
                coverage = white_pixels / total_pixels * 100
                logger.info(f"✅ Florence-2 Mask覆盖率: {coverage:.2f}% ({white_pixels}/{total_pixels})")
                return result.mask_image
            else:
                logger.warning("⚠️ Florence-2 mask生成失败，返回空mask")
                return Image.new('L', image.size, 0)
        
        def _apply_transparency(self, image: Image.Image, mask: Image.Image) -> Image.Image:
            """应用透明效果"""
            image_rgba = image.convert("RGBA")
            img_array = np.array(image_rgba)
            mask_array = np.array(mask)
            
            # 应用透明效果
            transparent_mask = mask_array > 128
            img_array[transparent_mask, 3] = 0
            
            return Image.fromarray(img_array, 'RGBA')
        
        def _apply_inpainting(self, image: Image.Image, mask: Image.Image, params: Dict[str, Any]) -> Image.Image:
            """应用inpainting（使用自定义参数）"""
            import logging
            logger = logging.getLogger(__name__)
            
            # 构建LaMA配置
            lama_config = {}
            
            # 处理所有inpainting参数
            if 'ldm_steps' in params:
                lama_config['ldm_steps'] = params['ldm_steps']
            if 'ldm_sampler' in params:
                lama_config['ldm_sampler'] = params['ldm_sampler']
            if 'hd_strategy' in params:
                lama_config['hd_strategy'] = params['hd_strategy']
            if 'hd_strategy_crop_margin' in params:
                lama_config['hd_strategy_crop_margin'] = params['hd_strategy_crop_margin']
            if 'hd_strategy_crop_trigger_size' in params:
                lama_config['hd_strategy_crop_trigger_size'] = params['hd_strategy_crop_trigger_size']
            if 'hd_strategy_resize_limit' in params:
                lama_config['hd_strategy_resize_limit'] = params['hd_strategy_resize_limit']
            
            logger.info(f"🎛️ 前端参数传递: {params}")
            logger.info(f"⚙️ LaMA配置构建: {lama_config}")
            logger.info(f"🎭 Mask信息: size={mask.size}, mode={mask.mode}")
            
            # 直接调用web_backend的_process_with_lama方法，传递预生成的mask
            result_image = self.base_processor._process_with_lama(
                image=image,
                mask=mask,
                custom_config=lama_config
            )
            
            logger.info(f"✅ Inpainting完成: size={result_image.size}, mode={result_image.mode}")
            return result_image
    
    return EnhancedWatermarkProcessor(st.session_state.processor)

def render_parameter_panel():
    """渲染左侧参数面板"""
    st.sidebar.header("🔬 Debug Parameters")
    
    # 日志级别控制
    st.sidebar.subheader("📝 Logging Control")
    log_level = st.sidebar.selectbox(
        "Log Level",
        ["INFO", "WARNING", "ERROR"],
        help="控制日志详细程度"
    )
    
    # 设置日志级别
    level_map = {"INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    logging.getLogger().setLevel(level_map[log_level])
    
    # Mask模型选择
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
                "Additional Dilate", 0, 20, 0, 1,
                help="额外膨胀处理（0=不处理）"
            )
        else:
            st.sidebar.warning("⚠️ Please upload a mask file")
    
    st.sidebar.divider()
    
    # Inpainting参数
    st.sidebar.subheader("🎨 Inpainting Parameters")
    inpaint_params = {}
    
    inpaint_params['prompt'] = st.sidebar.text_input(
        "Prompt",
        value="",
        help="文本提示词（注意：当前LaMA模型不支持prompt，保留供未来使用）",
        disabled=True
    )
    
    if inpaint_params['prompt']:
        st.sidebar.info("💡 LaMA模型暂不支持prompt，此参数保留供未来扩展")
    else:
        st.sidebar.info("ℹ️ Prompt功能暂不可用，LaMA为无条件inpainting模型")
    
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
    
    inpaint_params['seed'] = st.sidebar.number_input(
        "Seed", -1, 999999, -1,
        help="随机种子（-1为随机）"
    )
    
    st.sidebar.divider()
    
    # 性能选项
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
    
    # 处理模式
    st.sidebar.divider()
    st.sidebar.subheader("🔧 Processing Mode")
    
    transparent = st.sidebar.checkbox(
        "Transparent Mode",
        value=False,
        help="创建透明区域而不是填充修复"
    )
    
    return mask_model, mask_params, inpaint_params, performance_params, transparent

def render_main_area():
    """渲染主区域"""
    st.title("🔬 AI Watermark Remover - Debug Edition")
    st.markdown("**Parameter Control & Real-time Comparison**")
    
    # 图片上传
    uploaded_file = st.file_uploader(
        "📸 Upload Image for Debug",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload an image with watermarks to debug removal process"
    )
    
    if uploaded_file is not None:
        # 优化图像读取 - 使用OpenCV避免PIL色彩校正
        # 保存原始文件数据用于OpenCV直接解码
        uploaded_file.seek(0)  # 确保从头读取
        file_bytes = uploaded_file.read()
        st.session_state.original_file_bytes = file_bytes
        
        # 同时保存PIL版本用于显示
        uploaded_file.seek(0)
        original_image = Image.open(uploaded_file)
        st.session_state.original_image = original_image
        
        # 获取参数
        mask_model, mask_params, inpaint_params, performance_params, transparent = render_parameter_panel()
        
        # 显示参数总结
        with st.expander("📋 Current Parameters Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Mask Generation:**")
                st.write(f"Model: {mask_model}")
                for key, value in mask_params.items():
                    st.write(f"{key}: {value}")
            
            with col2:
                st.write("**Inpainting:**")
                for key, value in inpaint_params.items():
                    if key == 'prompt' and not value:
                        continue
                    # 特殊处理策略相关参数显示
                    if key.startswith('hd_strategy_') and inpaint_params['hd_strategy'] == 'ORIGINAL':
                        if key == 'hd_strategy_crop_margin' or key == 'hd_strategy_crop_trigger_size':
                            st.write(f"{key}: {value} *(不适用于ORIGINAL策略)*")
                        elif key == 'hd_strategy_resize_limit':
                            st.write(f"{key}: {value} *(不适用于ORIGINAL策略)*")
                        else:
                            st.write(f"{key}: {value}")
                    elif key.startswith('hd_strategy_') and inpaint_params['hd_strategy'] == 'RESIZE':
                        if key == 'hd_strategy_crop_margin' or key == 'hd_strategy_crop_trigger_size':
                            st.write(f"{key}: {value} *(不适用于RESIZE策略)*")
                        else:
                            st.write(f"{key}: {value}")
                    elif key.startswith('hd_strategy_') and inpaint_params['hd_strategy'] == 'CROP':
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
        
        # 处理按钮
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process with Debug Parameters", type="primary", use_container_width=True):
                with st.spinner("Processing with debug parameters..."):
                    enhanced_processor = create_enhanced_backend()
                    
                    # 使用原始文件字节数据获得最佳效果
                    if hasattr(st.session_state, 'original_file_bytes'):
                        # 使用OpenCV直接解码，避免PIL色彩校正
                        image_input = st.session_state.original_file_bytes
                    else:
                        # 回退到PIL图像
                        image_input = original_image
                    
                    result = enhanced_processor.process_image_with_params(
                        image=image_input,
                        mask_model=mask_model,
                        mask_params=mask_params,
                        inpaint_params=inpaint_params,
                        performance_params=performance_params,
                        transparent=transparent
                    )
                    
                    st.session_state.processing_result = result
        
        # 显示结果
        if st.session_state.processing_result and st.session_state.processing_result.success:
            st.subheader("🔄 Before vs After Comparison")
            
            result = st.session_state.processing_result
            
            # 图像对比
            if result.result_image:
                # 确保两个图像尺寸一致
                original_display = original_image
                result_display = result.result_image
                
                if original_display.size != result_display.size:
                    result_display = result_display.resize(original_display.size, Image.LANCZOS)
                
                # 如果是透明图像，提供背景选择
                if result_display.mode == 'RGBA':
                    bg_color = st.selectbox(
                        "Preview Background", 
                        ["white", "black", "checkered"],
                        key="comparison_bg"
                    )
                    
                    if bg_color != "white":
                        result_display = add_background(result_display, bg_color)
                
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
                    mask_array = np.array(result.mask_image)
                    white_pixels = np.sum(mask_array > 128)
                    total_pixels = mask_array.size
                    coverage = white_pixels / total_pixels * 100
                    st.metric("Mask Coverage", f"{coverage:.1f}%")
            
            with col3:
                st.metric("Output Mode", "Transparent" if transparent else "Inpaint")
            
            # 显示生成的mask
            if result.mask_image:
                with st.expander("🎭 Generated Mask", expanded=False):
                    col_mask1, col_mask2 = st.columns([3, 1])
                    with col_mask1:
                        st.image(result.mask_image, caption=f"Detected watermark mask ({result.mask_image.size[0]}×{result.mask_image.size[1]})", use_column_width=True)
                    with col_mask2:
                        # 添加mask下载按钮
                        mask_buffer = io.BytesIO()
                        result.mask_image.save(mask_buffer, format="PNG")
                        mask_buffer.seek(0)
                        st.download_button(
                            label="📥 Download Mask",
                            data=mask_buffer.getvalue(),
                            file_name=f"{Path(uploaded_file.name).stem}_mask.png",
                            mime="image/png",
                            use_container_width=True,
                            help=f"Download original resolution mask ({result.mask_image.size[0]}×{result.mask_image.size[1]})"
                        )
            
            # 下载选项
            st.subheader("📥 Download Results")
            create_download_buttons(result.result_image, Path(uploaded_file.name).stem)
            
        elif st.session_state.processing_result and not st.session_state.processing_result.success:
            st.error(f"❌ Processing failed: {st.session_state.processing_result.error_message}")
    
    else:
        # 显示参数面板但不处理
        render_parameter_panel()
        
        # 显示使用说明
        st.info("📸 Please upload an image to start debugging watermark removal parameters.")
        
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

def add_background(rgba_image: Image.Image, bg_type: str) -> Image.Image:
    """为RGBA图像添加背景"""
    if bg_type == "black":
        bg = Image.new('RGB', rgba_image.size, (0, 0, 0))
    elif bg_type == "checkered":
        bg = Image.new('RGB', rgba_image.size, (255, 255, 255))
        # 创建棋盘背景
        for y in range(0, rgba_image.size[1], 20):
            for x in range(0, rgba_image.size[0], 20):
                if (x//20 + y//20) % 2:
                    for dy in range(min(20, rgba_image.size[1] - y)):
                        for dx in range(min(20, rgba_image.size[0] - x)):
                            bg.putpixel((x + dx, y + dy), (200, 200, 200))
    else:  # white
        bg = Image.new('RGB', rgba_image.size, (255, 255, 255))
    
    # 合并图像
    bg.paste(rgba_image, mask=rgba_image.split()[-1])
    return bg

def create_download_buttons(image: Image.Image, filename_base: str):
    """创建下载按钮"""
    col1, col2, col3 = st.columns(3)
    
    formats = [("PNG", "image/png"), ("WEBP", "image/webp"), ("JPG", "image/jpeg")]
    
    for idx, (fmt, mime) in enumerate(formats):
        with [col1, col2, col3][idx]:
            img_buffer = io.BytesIO()
            
            if fmt == "PNG":
                image.save(img_buffer, format="PNG")
            elif fmt == "WEBP":
                image.save(img_buffer, format="WEBP", quality=95)
            else:  # JPG
                if image.mode == "RGBA":
                    rgb_img = Image.new("RGB", image.size, (255, 255, 255))
                    rgb_img.paste(image, mask=image.split()[-1])
                    image = rgb_img
                image.save(img_buffer, format="JPEG", quality=95)
            
            img_buffer.seek(0)
            
            st.download_button(
                label=f"📥 {fmt}",
                data=img_buffer.getvalue(),
                file_name=f"{filename_base}_debug.{fmt.lower()}",
                mime=mime,
                use_container_width=True
            )

def main():
    """主应用函数"""
    
    # 加载处理器
    if not load_processor():
        return
    
    # 渲染主界面
    render_main_area()
    
    # 页脚信息
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔬 Debug Edition")
    with col2:
        st.caption("⚡ Real-time Parameters")  
    with col3:
        st.caption("🔄 Interactive Comparison")

if __name__ == "__main__":
    main()