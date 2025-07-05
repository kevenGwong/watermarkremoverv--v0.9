"""
颜色空间处理工具
解决不同模型的颜色格式差异
"""

import cv2
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ColorSpaceProcessor:
    """颜色空间处理器"""
    
    @staticmethod
    def prepare_image_for_model(image: np.ndarray, model_name: str) -> np.ndarray:
        """
        根据模型类型准备图像
        
        Args:
            image: RGB格式的输入图像
            model_name: 模型名称 ('lama', 'zits', 'mat', 'fcf')
            
        Returns:
            处理后的图像数组
        """
        if model_name.lower() == 'lama':
            # LaMA需要BGR输入
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # 其他模型使用RGB
            return image
    
    @staticmethod
    def process_output_for_display(result: np.ndarray, model_name: str) -> np.ndarray:
        """
        处理模型输出用于显示
        
        Args:
            result: 模型输出结果
            model_name: 模型名称
            
        Returns:
            RGB格式的图像数组
        """
        if model_name.lower() == 'lama':
            # LaMA输出BGR，转换为RGB显示
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        else:
            # 其他模型输出RGB
            return result
    
    @staticmethod
    def validate_color_format(image: np.ndarray, expected_format: str = 'RGB') -> bool:
        """
        验证图像颜色格式
        
        Args:
            image: 图像数组
            expected_format: 期望的格式 ('RGB' 或 'BGR')
            
        Returns:
            格式是否正确
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return False
            
        # 简单的颜色分布检查
        if expected_format == 'RGB':
            # RGB图像通常R通道值较高
            return np.mean(image[:, :, 0]) > np.mean(image[:, :, 2])
        else:
            # BGR图像通常B通道值较高
            return np.mean(image[:, :, 2]) > np.mean(image[:, :, 0])
    
    @staticmethod
    def fix_color_channels(image: np.ndarray, current_format: str, target_format: str) -> np.ndarray:
        """
        修复颜色通道
        
        Args:
            image: 输入图像
            current_format: 当前格式
            target_format: 目标格式
            
        Returns:
            修复后的图像
        """
        if current_format == target_format:
            return image
            
        if current_format == 'RGB' and target_format == 'BGR':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif current_format == 'BGR' and target_format == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            logger.warning(f"Unsupported color conversion: {current_format} -> {target_format}")
            return image

class ModelColorConfig:
    """模型颜色配置"""
    
    # 模型颜色格式配置
    MODEL_COLOR_CONFIGS = {
        'lama': {
            'input_format': 'BGR',
            'output_format': 'BGR',
            'display_format': 'RGB'
        },
        'zits': {
            'input_format': 'RGB',
            'output_format': 'RGB',
            'display_format': 'RGB'
        },
        'mat': {
            'input_format': 'RGB',
            'output_format': 'RGB',
            'display_format': 'RGB'
        },
        'fcf': {
            'input_format': 'RGB',
            'output_format': 'RGB',
            'display_format': 'RGB'
        }
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, str]:
        """获取模型颜色配置"""
        return cls.MODEL_COLOR_CONFIGS.get(model_name.lower(), {
            'input_format': 'RGB',
            'output_format': 'RGB',
            'display_format': 'RGB'
        })
    
    @classmethod
    def prepare_input(cls, image: np.ndarray, model_name: str) -> np.ndarray:
        """准备模型输入"""
        config = cls.get_model_config(model_name)
        return ColorSpaceProcessor.prepare_image_for_model(image, model_name)
    
    @classmethod
    def process_output(cls, result: np.ndarray, model_name: str) -> np.ndarray:
        """处理模型输出"""
        config = cls.get_model_config(model_name)
        return ColorSpaceProcessor.process_output_for_display(result, model_name)
