"""
处理器模块
包含所有图像处理相关的处理器类
"""

from .processing_result import ProcessingResult
from .simplified_watermark_processor import SimplifiedWatermarkProcessor

# 兼容性别名 - 避免循环导入
WatermarkProcessor = SimplifiedWatermarkProcessor
EnhancedWatermarkProcessor = SimplifiedWatermarkProcessor

__all__ = [
    'ProcessingResult',
    'SimplifiedWatermarkProcessor',
    'WatermarkProcessor', 
    'EnhancedWatermarkProcessor'
] 