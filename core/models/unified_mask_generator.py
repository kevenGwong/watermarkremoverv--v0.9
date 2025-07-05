"""
统一Mask生成器 - SIMP-LAMA架构
遵循Mask Decoupling原则，只负责输出单通道mask
确保与所有IOPaint模型兼容
"""

import logging
import time
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class UnifiedMaskGenerator:
    """
    统一Mask生成器 - SIMP-LAMA架构实现
    
    核心原则:
    - Mask Decoupling: 只负责输出单通道mask
    - Interface Unification: 统一的generate_mask接口
    - Minimal Params: 极简参数，隐藏复杂性
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.custom_generator = None
        self._init_generators()
    
    def _init_generators(self):
        """初始化各种mask生成器"""
        try:
            # 初始化自定义mask生成器
            self._init_custom_generator()
            logger.info("✅ UnifiedMaskGenerator initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize UnifiedMaskGenerator: {e}")
            raise
    
    def _init_custom_generator(self):
        """初始化自定义mask生成器"""
        try:
            from .mask_generators import CustomMaskGenerator
            
            mask_config = self.config.get('mask_generator', {})
            self.custom_generator = CustomMaskGenerator(self.config)
            
            if self.custom_generator.is_available():
                logger.info("✅ Custom mask generator ready")
            else:
                logger.warning("⚠️ Custom mask generator not available, will use fallback")
                
        except Exception as e:
            logger.warning(f"⚠️ Custom mask generator initialization failed: {e}")
            self.custom_generator = None
    
    def generate_mask(self, 
                     image: Image.Image, 
                     method: str = "custom",
                     params: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        统一mask生成接口 - SIMP-LAMA核心方法
        
        Args:
            image: 输入图像 (PIL Image)
            method: 生成方法 ("custom", "upload", "simple")
            params: 生成参数
            
        Returns:
            Image: 单通道mask (模式: 'L', 黑背景白前景)
        """
        if params is None:
            params = {}
            
        start_time = time.time()
        logger.info(f"🎭 开始生成mask - 方法: {method}")
        
        try:
            if method == "custom":
                mask = self._generate_custom_mask(image, params)
            elif method == "upload":
                mask = self._handle_uploaded_mask(image, params)
            elif method == "simple":
                mask = self._generate_simple_mask(image, params)
            else:
                logger.warning(f"未知mask方法: {method}, 使用简单生成")
                mask = self._generate_simple_mask(image, params)
            
            # 验证和标准化mask
            mask = self._validate_and_standardize_mask(mask, image.size)
            
            generation_time = time.time() - start_time
            coverage = self._calculate_coverage(mask)
            
            logger.info(f"✅ Mask生成完成 - 耗时: {generation_time:.2f}s, 覆盖率: {coverage:.2f}%")
            
            return mask
            
        except Exception as e:
            logger.error(f"❌ Mask生成失败: {e}")
            # 返回安全的fallback mask
            return self._generate_fallback_mask(image)
    
    def _generate_custom_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """生成自定义mask"""
        if self.custom_generator and self.custom_generator.is_available():
            try:
                return self.custom_generator.generate_mask(image, params)
            except Exception as e:
                logger.warning(f"Custom mask generation failed: {e}, using fallback")
                return self._generate_fallback_mask(image)
        else:
            logger.warning("Custom generator not available, using fallback")
            return self._generate_fallback_mask(image)
    
    def _handle_uploaded_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """处理上传的mask"""
        uploaded_mask = params.get('uploaded_mask')
        if not uploaded_mask:
            raise ValueError("No uploaded mask provided")
        
        try:
            # 处理不同类型的上传mask
            if hasattr(uploaded_mask, 'read'):
                # 文件对象
                mask_image = Image.open(uploaded_mask)
            elif isinstance(uploaded_mask, (str, Path)):
                # 文件路径
                mask_image = Image.open(uploaded_mask)
            elif isinstance(uploaded_mask, Image.Image):
                # PIL Image对象
                mask_image = uploaded_mask
            else:
                raise ValueError(f"Unsupported uploaded mask type: {type(uploaded_mask)}")
            
            # 转换为灰度并调整尺寸
            mask_image = mask_image.convert('L')
            if mask_image.size != image.size:
                mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
            
            # 可选的后处理
            dilate_size = params.get('mask_dilate_kernel_size', 0)
            if dilate_size > 0:
                mask_image = self._apply_morphological_ops(mask_image, dilate_size, params)
            
            logger.info(f"📤 上传mask处理完成: {mask_image.size}")
            return mask_image
            
        except Exception as e:
            logger.error(f"Failed to process uploaded mask: {e}")
            raise
    
    def _generate_simple_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """生成简单mask（用于测试和fallback）"""
        width, height = image.size
        coverage_percent = params.get('coverage_percent', 25)
        
        # 创建中心圆形mask
        mask = Image.new('L', (width, height), 0)
        mask_array = np.array(mask)
        
        # 计算圆形区域
        area = width * height
        target_area = area * coverage_percent / 100
        radius = int(np.sqrt(target_area / np.pi))
        
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask_array[circle_mask] = 255
        
        logger.info(f"🔧 简单mask生成: {coverage_percent}% 覆盖率")
        return Image.fromarray(mask_array, mode='L')
    
    def _generate_fallback_mask(self, image: Image.Image) -> Image.Image:
        """生成安全的fallback mask"""
        logger.warning("🚨 使用fallback mask生成")
        
        width, height = image.size
        mask = Image.new('L', (width, height), 0)
        
        # 创建保守的中心椭圆区域
        center_x, center_y = width // 2, height // 2
        rx, ry = min(width, height) // 6, min(width, height) // 8
        
        mask_array = np.array(mask)
        y, x = np.ogrid[:height, :width]
        ellipse_mask = ((x - center_x) ** 2 / (rx ** 2) + 
                       (y - center_y) ** 2 / (ry ** 2)) <= 1
        mask_array[ellipse_mask] = 255
        
        return Image.fromarray(mask_array, mode='L')
    
    def _validate_and_standardize_mask(self, mask: Image.Image, target_size: tuple) -> Image.Image:
        """验证和标准化mask格式"""
        # 确保是灰度模式
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # 确保尺寸匹配
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
        
        # 确保是二值化的
        mask_array = np.array(mask)
        
        # 如果不是二值化的，应用阈值
        if len(np.unique(mask_array)) > 2:
            threshold = 128
            mask_array = (mask_array > threshold).astype(np.uint8) * 255
            mask = Image.fromarray(mask_array, mode='L')
        
        return mask
    
    def _apply_morphological_ops(self, mask: Image.Image, 
                                kernel_size: int, 
                                params: Dict[str, Any]) -> Image.Image:
        """应用形态学操作"""
        try:
            import cv2
            
            mask_array = np.array(mask)
            iterations = params.get('mask_dilate_iterations', 1)
            
            # 创建椭圆核
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # 膨胀操作
            if kernel_size > 0 and iterations > 0:
                mask_array = cv2.dilate(mask_array, kernel, iterations=iterations)
            
            return Image.fromarray(mask_array, mode='L')
            
        except Exception as e:
            logger.warning(f"Morphological operations failed: {e}")
            return mask
    
    def _calculate_coverage(self, mask: Image.Image) -> float:
        """计算mask覆盖率"""
        mask_array = np.array(mask)
        total_pixels = mask_array.size
        white_pixels = np.sum(mask_array > 128)
        return (white_pixels / total_pixels) * 100
    
    def validate_mask_compatibility(self, mask: Image.Image, model_name: str) -> bool:
        """验证mask与特定模型的兼容性"""
        try:
            # 基础格式检查
            if mask.mode != 'L':
                logger.warning(f"Mask mode should be 'L', got '{mask.mode}'")
                return False
            
            # 检查是否为二值化
            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            
            if len(unique_values) > 10:  # 允许一些中间值（抗锯齿）
                logger.warning("Mask should be binary or near-binary")
                return False
            
            # 检查覆盖率
            coverage = self._calculate_coverage(mask)
            if coverage < 0.1:
                logger.warning(f"Mask coverage too low: {coverage:.2f}%")
                return False
            elif coverage > 80:
                logger.warning(f"Mask coverage too high: {coverage:.2f}%")
                return False
            
            logger.info(f"✅ Mask validated for {model_name}: {coverage:.2f}% coverage")
            return True
            
        except Exception as e:
            logger.error(f"Mask validation failed: {e}")
            return False
    
    def get_mask_info(self, mask: Image.Image) -> Dict[str, Any]:
        """获取mask详细信息"""
        mask_array = np.array(mask)
        
        return {
            'size': mask.size,
            'mode': mask.mode,
            'coverage_percent': self._calculate_coverage(mask),
            'unique_values': len(np.unique(mask_array)),
            'min_value': int(mask_array.min()),
            'max_value': int(mask_array.max()),
            'total_pixels': mask_array.size,
            'white_pixels': int(np.sum(mask_array > 128))
        }
    
    def cleanup_resources(self):
        """清理资源"""
        try:
            if self.custom_generator:
                self.custom_generator.cleanup_resources()
                self.custom_generator = None
            
            logger.info("✅ UnifiedMaskGenerator resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# 工厂函数
def create_mask_generator(config: Dict[str, Any]) -> UnifiedMaskGenerator:
    """创建统一mask生成器实例"""
    return UnifiedMaskGenerator(config)