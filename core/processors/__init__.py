"""
处理器模块
包含所有图像处理相关的处理器类
"""

from .processing_result import ProcessingResult
from .watermark_processor import WatermarkProcessor, EnhancedWatermarkProcessor

__all__ = [
    'ProcessingResult',
    'WatermarkProcessor', 
    'EnhancedWatermarkProcessor'
] 