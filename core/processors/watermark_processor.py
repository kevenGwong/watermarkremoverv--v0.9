"""
水印处理主类 - 模块化版本
"""

import logging
import time
import yaml
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
from pathlib import Path

from .processing_result import ProcessingResult
from ..models.mask_generators import CustomMaskGenerator, FlorenceMaskGenerator, FallbackMaskGenerator
from ..models.lama_processor import LamaProcessor
from ..models.iopaint_processor import IOPaintProcessor

logger = logging.getLogger(__name__)

class WatermarkProcessor:
    """水印处理主类 - 基于原始 web_backend.py"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "web_config.yaml"
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._resources = []  # Track resources for cleanup
        
        # Initialize components
        self._init_components()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup"""
        self.cleanup_resources()
    
    def cleanup_resources(self):
        """Clean up all processor resources"""
        try:
            # Clean up LaMA processor
            if hasattr(self, 'lama_processor') and self.lama_processor:
                if hasattr(self.lama_processor, 'cleanup_resources'):
                    self.lama_processor.cleanup_resources()
            
            # Clean up mask generator if it has cleanup method
            if hasattr(self, 'mask_generator') and hasattr(self.mask_generator, 'cleanup_resources'):
                self.mask_generator.cleanup_resources()
            
            # Clear CUDA cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ WatermarkProcessor resources cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error during WatermarkProcessor cleanup: {e}")
    
    def _init_components(self):
        """Initialize processor components"""
        # 初始化mask生成器
        mask_type = self.config['mask_generator']['model_type']
        try:
            if mask_type == "custom":
                self.mask_generator = CustomMaskGenerator(self.config)
            else:
                self.mask_generator = FlorenceMaskGenerator(self.config)
        except Exception as e:
            logger.error(f"Failed to initialize mask generator: {e}")
            # 提供降级方案 - 使用基础的空mask生成器
            self.mask_generator = FallbackMaskGenerator()
            logger.info("Using fallback mask generator")
        
        # 初始化LaMA处理器
        try:
            self.lama_processor = LamaProcessor(self.config)
        except Exception as e:
            logger.error(f"Failed to initialize LaMA processor: {e}")
            raise RuntimeError(f"Critical failure: LaMA processor initialization failed: {e}")
        
        logger.info(f"✅ WatermarkProcessor initialized with {mask_type} mask generator")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ Config loaded from: {config_path}")
                return config
        except Exception as e:
            logger.warning(f"Config file {config_path} not found, using defaults")
            # 使用ConfigManager的默认配置
            from config.config import ConfigManager
            config_manager = ConfigManager()
            return self._build_default_config(config_manager)
    
    def _build_default_config(self, config_manager) -> Dict[str, Any]:
        """构建默认配置"""
        return {
            'mask_generator': {
                'model_type': 'custom',
                'mask_model_path': '/home/duolaameng/SAM_Remove/Watermark_sam/output/checkpoints/epoch=071-valid_iou=0.7267.ckpt',
                'image_size': 768,
                'imagenet_mean': [0.485, 0.456, 0.406],
                'imagenet_std': [0.229, 0.224, 0.225],
                'mask_threshold': config_manager.app_config.default_mask_threshold,
            },
            'models': {
                'florence_model': 'microsoft/Florence-2-large',
                'lama_model': config_manager.get_model_config().get('lama_model', 'lama')
            }
        }
    
    def process_image(self, 
                     image: Image.Image,
                     transparent: bool = False,
                     max_bbox_percent: float = 10.0,
                     force_format: Optional[str] = None,
                     custom_inpaint_config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """处理单张图片"""
        start_time = time.time()
        
        try:
            logger.info("🚀 开始处理图像...")
            logger.info(f"📸 输入图像: size={image.size}, mode={image.mode}")
            logger.info(f"🎯 处理模式: {'透明' if transparent else '修复'}")
            
            # 生成mask
            logger.info("🎭 开始生成mask...")
            mask_params = {'max_bbox_percent': max_bbox_percent}
            mask_image = self.mask_generator.generate_mask(image, mask_params)
            
            # 验证mask
            mask_array = np.array(mask_image)
            white_pixels = np.sum(mask_array > 128)
            total_pixels = mask_array.size
            mask_coverage = white_pixels / total_pixels * 100
            logger.info(f"🔍 Mask验证: 覆盖率={mask_coverage:.2f}%")
            
            if transparent:
                logger.info("🫥 应用透明处理...")
                result_image = self._make_region_transparent(image, mask_image)
            else:
                logger.info("🎨 应用LaMA修复处理...")
                if custom_inpaint_config is None:
                    custom_inpaint_config = {}
                result_image_array = self.lama_processor.predict(image, mask_image, custom_inpaint_config)
                result_image = Image.fromarray(result_image_array)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ 处理完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                success=True,
                result_image=result_image,
                mask_image=mask_image,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 图像处理失败: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _make_region_transparent(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """使区域透明"""
        image = image.convert("RGBA")
        mask = mask.convert("L")
        transparent_image = Image.new("RGBA", image.size)
        
        for x in range(image.width):
            for y in range(image.height):
                if mask.getpixel((x, y)) > 0:
                    transparent_image.putpixel((x, y), (0, 0, 0, 0))
                else:
                    transparent_image.putpixel((x, y), image.getpixel((x, y)))
        
        return transparent_image
    
    def _process_with_lama(self, image: Image.Image, mask: Image.Image, lama_config: Dict[str, Any]) -> Image.Image:
        """使用LaMA进行inpainting - 兼容接口"""
        result_array = self.lama_processor.predict(image, mask, lama_config)
        return Image.fromarray(result_array)
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import torch
        import psutil
        
        info = {
            "cuda_available": torch.cuda.is_available(),
            "lama_loaded": self.lama_processor.model_loaded,
            "mask_generator": self.config['mask_generator']['model_type'],
            "ram_usage": f"{psutil.virtual_memory().percent:.1f}%",
            "cpu_usage": f"{psutil.cpu_percent():.1f}%"
        }
        
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            vram_total = gpu_info.total_memory // (1024 ** 2)
            vram_used = torch.cuda.memory_reserved(0) // (1024 ** 2)
            info["vram_usage"] = f"{vram_used}/{vram_total} MB"
            info["gpu_name"] = gpu_info.name
        
        return info

class EnhancedWatermarkProcessor:
    """增强的水印处理器 - 支持 IOPaint"""
    
    def __init__(self, base_processor: WatermarkProcessor):
        self.base_processor = base_processor
        self.iopaint_processor = None
        self._load_iopaint_processor()
    
    def _load_iopaint_processor(self):
        """Load IOPaint processor"""
        try:
            # Create IOPaint config
            config = {
                'models': {
                    'inpaint_model': 'mat',  # 默认使用MAT
                    'available_models': ['zits', 'mat', 'fcf', 'lama']
                },
                'iopaint_config': {
                    'hd_strategy': 'CROP',
                    'hd_strategy_crop_margin': 64,
                    'hd_strategy_crop_trigger_size': 1024,
                    'hd_strategy_resize_limit': 2048,
                    'ldm_steps': 50,
                    'auto_model_selection': True
                }
            }
            
            self.iopaint_processor = IOPaintProcessor(config)
            logger.info("✅ IOPaint processor loaded successfully")
            
        except Exception as e:
            logger.warning(f"IOPaint processor loading failed: {e}")
            logger.info("IOPaint functionality will not be available")
    
    def process_image_with_params(self, 
                                image: Image.Image,
                                mask_model: str,
                                mask_params: Dict[str, Any],
                                inpaint_params: Dict[str, Any],
                                performance_params: Dict[str, Any],
                                transparent: bool = False) -> ProcessingResult:
        """使用详细参数处理图像"""
        start_time = time.time()
        
        try:
            logger.info("🚀 开始增强处理流程...")
            logger.info(f"🎭 Mask模型: {mask_model}")
            logger.info(f"⚙️ Inpaint参数: {inpaint_params}")
            
            # 生成mask
            if mask_model == "upload":
                mask_image = self._generate_uploaded_mask(image, mask_params)
            elif mask_model == "florence2":
                mask_image = self.base_processor.mask_generator.generate_mask(image, mask_params)
            else:  # custom
                mask_image = self.base_processor.mask_generator.generate_mask(image, mask_params)
            
            # 验证mask
            mask_array = np.array(mask_image)
            white_pixels = np.sum(mask_array > 128)
            total_pixels = mask_array.size
            mask_coverage = white_pixels / total_pixels * 100
            logger.info(f"🔍 Mask验证: 覆盖率={mask_coverage:.2f}%")
            
            # 应用处理
            if transparent:
                logger.info("🫥 应用透明处理...")
                result_image = self.base_processor._make_region_transparent(image, mask_image)
            else:
                inpaint_model = inpaint_params.get('inpaint_model', 'lama')
                
                if inpaint_model == 'iopaint' and self.iopaint_processor and self.iopaint_processor.model_loaded:
                    logger.info("🎨 应用IOPaint处理...")
                    result_array = self.iopaint_processor.predict(image, mask_image, inpaint_params)
                    result_image = Image.fromarray(result_array)
                else:
                    logger.info("🎨 应用LaMA处理...")
                    result_array = self.base_processor.lama_processor.predict(image, mask_image, inpaint_params)
                    result_image = Image.fromarray(result_array)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ 处理完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                success=True,
                result_image=result_image,
                mask_image=mask_image,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 增强处理失败: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _generate_uploaded_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """处理上传的mask"""
        uploaded_mask = params.get('uploaded_mask')
        if not uploaded_mask:
            raise ValueError("No mask file uploaded")
        
        # Validate uploaded mask
        if hasattr(uploaded_mask, 'size') and uploaded_mask.size == 0:
            raise ValueError("Uploaded mask file is empty")
        
        try:
            logger.info(f"📂 处理上传的mask文件")
            logger.info(f"📏 原图尺寸: {image.size}")
            
            # 读取上传的mask
            if hasattr(uploaded_mask, 'read'):
                # Streamlit UploadedFile对象
                uploaded_mask.seek(0)  # 重置文件指针
                mask = Image.open(uploaded_mask)
            elif isinstance(uploaded_mask, Image.Image):
                # 已经是PIL Image对象
                mask = uploaded_mask
            else:
                # 文件路径
                mask = Image.open(uploaded_mask)
                
        except Exception as e:
            raise ValueError(f"Failed to open uploaded mask file: {e}")
        
        # Validate mask after loading
        if mask.size[0] <= 0 or mask.size[1] <= 0:
            raise ValueError("Invalid mask dimensions")
        
        logger.info(f"📏 原始mask尺寸: {mask.size}")
        logger.info(f"🎨 原始mask模式: {mask.mode}")
        
        # 确保mask是灰度图像
        if mask.mode != 'L':
            mask = mask.convert('L')
            logger.info(f"🔄 转换mask为灰度模式: {mask.mode}")
        
        # 调整mask尺寸以匹配图像
        if mask.size != image.size:
            logger.info(f"📐 调整mask尺寸: {mask.size} -> {image.size}")
            mask = mask.resize(image.size, Image.LANCZOS)
        else:
            logger.info(f"✅ Mask尺寸已匹配: {mask.size}")
        
        # 检查mask内容
        mask_array = np.array(mask)
        white_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        mask_coverage = white_pixels / total_pixels * 100
        logger.info(f"🔍 Mask内容分析: 白色像素={white_pixels}, 总像素={total_pixels}, 覆盖率={mask_coverage:.2f}%")
        logger.info(f"📊 Mask像素值范围: {mask_array.min()} - {mask_array.max()}")
        
        # 应用额外的膨胀处理（如果需要）
        dilate_size = params.get('mask_dilate_kernel_size', 0)
        if dilate_size > 0:
            import cv2
            logger.info(f"🔧 应用膨胀处理: kernel_size={dilate_size}")
            mask_array = np.array(mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
            mask = Image.fromarray(mask_array, mode='L')
            
            # 检查膨胀后的mask
            white_pixels_after = np.sum(mask_array > 128)
            coverage_after = white_pixels_after / total_pixels * 100
            logger.info(f"🔍 膨胀后Mask分析: 白色像素={white_pixels_after}, 覆盖率={coverage_after:.2f}%")
        
        logger.info(f"✅ 最终mask尺寸: {mask.size}, 模式: {mask.mode}")
        return mask 