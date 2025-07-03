"""
推理逻辑模块
负责AI模型推理、mask生成和inpainting处理
"""

import time
import logging
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

from core.utils.image_utils import ImageProcessor

logger = logging.getLogger(__name__)

class ProcessingResult:
    """处理结果数据类"""
    def __init__(self, success: bool, result_image: Optional[Image.Image] = None,
                 mask_image: Optional[Image.Image] = None, error_message: Optional[str] = None,
                 processing_time: float = 0.0):
        self.success = success
        self.result_image = result_image
        self.mask_image = mask_image
        self.error_message = error_message
        self.processing_time = processing_time

class MockProcessor:
    """模拟处理器 - 替代已弃用的web_backend.WatermarkProcessor"""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = {"models": {"lama_model": "lama"}}
        logger.info(f"MockProcessor initialized with config: {config_file}")
    
    def process_image(self, image: Image.Image, transparent: bool = False, 
                     max_bbox_percent: float = 10.0, force_format: str = "PNG",
                     custom_inpaint_config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """模拟图像处理 - 返回原始图像作为结果"""
        logger.info("MockProcessor: Processing image (returning original)")
        return ProcessingResult(
            success=True,
            result_image=image,
            mask_image=Image.new('L', image.size, 128),  # 灰色mask
            processing_time=0.1
        )
    
    def _process_with_lama(self, image: Image.Image, mask: Image.Image, 
                          lama_config: Dict[str, Any]) -> Image.Image:
        """模拟LaMA处理 - 返回原始图像"""
        logger.info("MockProcessor: LaMA processing (returning original)")
        return image
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "status": "MockProcessor active",
            "config_file": self.config_file,
            "model": "mock"
        }

class EnhancedWatermarkProcessor:
    """增强的水印处理器"""
    
    def __init__(self, base_processor):
        self.base_processor = base_processor
    
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
            # 根据选择的模型生成mask
            if mask_model == "custom":
                mask_image = self._generate_custom_mask(image, mask_params)
            elif mask_model == "florence2":
                mask_image = self._generate_florence_mask(image, mask_params)
            else:  # upload
                mask_image = self._generate_uploaded_mask(image, mask_params)
            
            # 应用处理
            if transparent:
                result_image = self._apply_transparency(image, mask_image)
            else:
                result_image = self._apply_inpainting(image, mask_image, inpaint_params)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                result_image=result_image,
                mask_image=mask_image,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Image processing failed: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _generate_custom_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """生成自定义mask"""
        try:
            # 更新自定义模型参数
            if hasattr(self.base_processor, 'mask_generator') and hasattr(self.base_processor.mask_generator, 'generate_mask'):
                generator = self.base_processor.mask_generator
                # 动态更新参数
                generator.mask_threshold = params.get('mask_threshold', 0.5)
                
                # 生成mask
                mask = generator.generate_mask(image)
                
                # 应用膨胀参数
                dilate_size = params.get('mask_dilate_kernel_size', 3)
                dilate_iterations = params.get('mask_dilate_iterations', 1)
                
                if dilate_size > 0:
                    mask = ImageProcessor.apply_mask_dilation(mask, dilate_size, dilate_iterations)
                
                return mask
            else:
                # 直接调用基础处理器的process_image
                result = self.base_processor.process_image(
                    image=image,
                    transparent=True,
                    max_bbox_percent=10.0,
                    force_format="PNG"
                )
                return result.mask_image if result.mask_image else Image.new('L', image.size, 0)
        except Exception as e:
            logger.error(f"Custom mask generation failed: {e}")
            raise
    
    def _generate_uploaded_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """处理上传的mask"""
        try:
            uploaded_mask = params.get('uploaded_mask')
            if not uploaded_mask:
                raise ValueError("No mask file uploaded")
            
            # 读取上传的mask
            mask = Image.open(uploaded_mask)
            
            # 确保mask是灰度图像
            mask = ImageProcessor.ensure_grayscale(mask)
            
            # 调整mask尺寸以匹配图像
            if mask.size != image.size:
                mask = ImageProcessor.resize_image(mask, image.size)
            
            # 应用膨胀处理以增强修复效果
            dilate_size = params.get('mask_dilate_kernel_size', 5)  # 默认增加膨胀
            dilate_iterations = params.get('mask_dilate_iterations', 2)  # 默认增加迭代次数
            
            if dilate_size > 0:
                mask = ImageProcessor.apply_mask_dilation(mask, dilate_size, dilate_iterations)
                logger.info(f"Applied mask dilation: kernel_size={dilate_size}, iterations={dilate_iterations}")
            
            return mask
        except Exception as e:
            logger.error(f"Uploaded mask processing failed: {e}")
            raise
    
    def _generate_florence_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """生成Florence-2 mask"""
        try:
            # 使用基础处理器，但传递参数
            max_bbox_percent = params.get('max_bbox_percent', 10.0)
            result = self.base_processor.process_image(
                image=image,
                transparent=True,
                max_bbox_percent=max_bbox_percent,
                force_format="PNG"
            )
            return result.mask_image if result.mask_image else Image.new('L', image.size, 0)
        except Exception as e:
            logger.error(f"Florence mask generation failed: {e}")
            raise
    
    def _apply_transparency(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """应用透明效果"""
        return ImageProcessor.apply_transparency(image, mask)
    
    def _apply_inpainting(self, image: Image.Image, mask: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """应用inpainting（使用自定义参数）"""
        try:
            # 直接调用LaMA模型，使用传入的mask
            if hasattr(self.base_processor, '_process_with_lama'):
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
                
                # 直接调用LaMA处理，使用传入的mask
                result_image = self.base_processor._process_with_lama(image, mask, lama_config)
                return result_image
            else:
                # 备用方案：使用process_image但传递自定义配置
                lama_config = {}
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
                
                result = self.base_processor.process_image(
                    image=image,
                    transparent=False,
                    max_bbox_percent=10.0,
                    force_format="PNG",
                    custom_inpaint_config=lama_config
                )
                return result.result_image if result.result_image else image
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            raise

class InferenceManager:
    """推理管理器"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.processor = None
        self.enhanced_processor = None
    
    def load_processor(self):
        """加载处理器"""
        try:
            # 使用MockProcessor替代已弃用的web_backend.WatermarkProcessor
            self.processor = MockProcessor(self.config_manager.config_file)
            self.enhanced_processor = EnhancedWatermarkProcessor(self.processor)
            logger.info("MockProcessor loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            return False
    
    def process_image(self, image: Image.Image, mask_model: str, mask_params: Dict[str, Any],
                     inpaint_params: Dict[str, Any], performance_params: Dict[str, Any],
                     transparent: bool) -> ProcessingResult:
        """处理图像"""
        if self.enhanced_processor is None:
            return ProcessingResult(
                success=False,
                error_message="Processor not loaded",
                processing_time=0.0
            )
        
        # 验证参数
        validated_mask_params = self.config_manager.validate_mask_params(mask_params)
        validated_inpaint_params = self.config_manager.validate_inpaint_params(inpaint_params)
        
        # 处理图像
        return self.enhanced_processor.process_image_with_params(
            image=image,
            mask_model=mask_model,
            mask_params=validated_mask_params,
            inpaint_params=validated_inpaint_params,
            performance_params=performance_params,
            transparent=transparent
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        if self.processor is None:
            return {"status": "Processor not loaded"}
        
        try:
            return self.processor.get_system_info()
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {"status": "Error getting system info"} 