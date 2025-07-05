"""
推理管理器模块
负责管理所有处理器的加载和协调
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image

from .processors.processing_result import ProcessingResult
from .models import UnifiedProcessor
from .models import CustomMaskGenerator, FlorenceMaskGenerator, FallbackMaskGenerator

logger = logging.getLogger(__name__)

class InferenceManager:
    """推理管理器"""
    
    def __init__(self, config_manager, config_path: Optional[str] = None):
        self.config_manager = config_manager
        self.config_path = config_path
        
        # 初始化统一处理器
        self.unified_processor = None
        
        # 初始化mask生成器
        self.custom_mask_generator = None
        self.florence_mask_generator = None
        self.fallback_mask_generator = None
    
    def load_processor(self) -> bool:
        """加载处理器"""
        try:
            # 获取配置
            config = self.config_manager.get_config()
            
            # 加载统一处理器
            self.unified_processor = UnifiedProcessor(config)
            
            # 加载mask生成器
            self.custom_mask_generator = CustomMaskGenerator(config)
            self.florence_mask_generator = FlorenceMaskGenerator(config)
            self.fallback_mask_generator = FallbackMaskGenerator(config)
            
            logger.info("✅ Inference manager loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load inference manager: {e}")
            return False
    
    def process_image(self, 
                     image: Image.Image,
                     mask_model: str,
                     mask_params: Dict[str, Any],
                     inpaint_params: Dict[str, Any],
                     performance_params: Dict[str, Any],
                     transparent: bool = False) -> ProcessingResult:
        """处理图像"""
        if self.unified_processor is None:
            return ProcessingResult(
                success=False,
                error_message="Processor not loaded"
            )
        
        start_time = time.time()
        
        try:
            # 生成或获取mask
            mask = self._generate_mask(image, mask_model, mask_params)
            
            # 手动选择模型
            model_name = inpaint_params.get('force_model', None)
            if model_name is None:
                # 如果没有指定模型，使用第一个可用的模型
                available_models = self.unified_processor.get_available_models()
                if not available_models:
                    raise RuntimeError("No models available")
                model_name = available_models[0]
                logger.info(f"No model specified, using default: {model_name}")
            
            logger.info(f"Using manually selected model: {model_name}")
            
            # 执行inpainting
            result_array = self.unified_processor.predict_with_model(
                model_name, image, mask, inpaint_params
            )
            
            # 创建结果图像
            result_image = Image.fromarray(result_array)
            
            # 处理透明背景
            if transparent:
                result_image = self._make_transparent(result_image, mask)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                result_image=result_image,
                mask_image=mask,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Image processing failed: {e}")
            
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _generate_mask(self, image: Image.Image, mask_model: str, mask_params: Dict[str, Any]) -> Image.Image:
        """生成mask"""
        try:
            if mask_model == "upload":
                # 处理上传的mask
                return self._handle_uploaded_mask(image, mask_params)
            elif mask_model == "florence" and hasattr(self.florence_mask_generator, 'is_available') and self.florence_mask_generator.is_available():
                return self.florence_mask_generator.generate_mask(image, mask_params)
            elif mask_model == "custom":
                return self.custom_mask_generator.generate_mask(image, mask_params)
            else:
                return self.fallback_mask_generator.generate_mask(image, mask_params)
        except Exception as e:
            logger.warning(f"Mask generation failed: {e}")
            return self.fallback_mask_generator.generate_mask(image, mask_params)
    
    def _handle_uploaded_mask(self, image: Image.Image, mask_params: Dict[str, Any]) -> Image.Image:
        """处理上传的mask"""
        try:
            uploaded_mask = mask_params.get('uploaded_mask')
            if uploaded_mask is None:
                logger.warning("No uploaded mask found")
                return self.fallback_mask_generator.generate_mask(image, mask_params)
            
            # 如果是PIL Image，直接使用
            if isinstance(uploaded_mask, Image.Image):
                mask = uploaded_mask.convert('L')
            else:
                # 如果是文件对象，读取并解析
                if hasattr(uploaded_mask, 'read'):
                    mask_data = uploaded_mask.read()
                    if isinstance(mask_data, bytes):
                        from io import BytesIO
                        mask = Image.open(BytesIO(mask_data)).convert('L')
                    else:
                        mask = uploaded_mask.convert('L')
                else:
                    logger.warning(f"Unknown uploaded mask type: {type(uploaded_mask)}")
                    return self.fallback_mask_generator.generate_mask(image, mask_params)
            
            # 确保mask尺寸与图像匹配
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
            
            # 应用膨胀处理
            dilate_kernel_size = mask_params.get('mask_dilate_kernel_size', 0)
            dilate_iterations = mask_params.get('mask_dilate_iterations', 0)
            
            if dilate_kernel_size > 0 and dilate_iterations > 0:
                import cv2
                mask_array = np.array(mask)
                kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
                mask_array = cv2.dilate(mask_array, kernel, iterations=dilate_iterations)
                mask = Image.fromarray(mask_array, mode='L')
            
            logger.info(f"✅ Uploaded mask processed successfully: {mask.size}")
            return mask
            
        except Exception as e:
            logger.error(f"❌ Failed to process uploaded mask: {e}")
            return self.fallback_mask_generator.generate_mask(image, mask_params)
    
    def _make_transparent(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """使背景透明"""
        try:
            # 转换mask为alpha通道
            mask_array = np.array(mask.convert("L"))
            alpha = 255 - mask_array  # 反转mask
            
            # 添加alpha通道
            result = image.convert("RGBA")
            result.putalpha(Image.fromarray(alpha))
            
            return result
        except Exception as e:
            logger.warning(f"Failed to make transparent: {e}")
            return image
    
    def switch_model(self, model_name: str) -> bool:
        """切换模型"""
        if self.unified_processor:
            return self.unified_processor.switch_model(model_name)
        return False
    
    def get_available_models(self) -> list:
        """获取可用模型列表"""
        if self.unified_processor:
            return self.unified_processor.get_available_models()
        return []
    
    def get_current_model(self) -> str:
        """获取当前模型"""
        if self.unified_processor:
            return self.unified_processor.get_current_model()
        return None
    
    def is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        if self.unified_processor:
            return self.unified_processor.is_model_loaded(model_name)
        return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        if self.unified_processor is None:
            return {"error": "Processor not loaded"}
        
        return {
            "available_models": self.get_available_models(),
            "current_model": self.get_current_model(),
            "total_models": len(self.get_available_models())
        }
    
    def cleanup_resources(self):
        """清理资源"""
        try:
            if self.unified_processor:
                self.unified_processor.cleanup_resources()
            logger.info("✅ Inference manager resources cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error during inference manager cleanup: {e}") 