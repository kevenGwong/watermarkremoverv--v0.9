"""
模型模块
包含所有AI模型相关的处理器
"""

from .lama_processor import LamaProcessor
from .zits_processor import ZitsProcessor
from .mat_processor import MatProcessor
from .fcf_processor import FcfProcessor
from .unified_processor import UnifiedProcessor
from .mask_generators import CustomMaskGenerator, FlorenceMaskGenerator, FallbackMaskGenerator

__all__ = [
    'LamaProcessor',
    'ZitsProcessor',
    'MatProcessor',
    'FcfProcessor',
    'UnifiedProcessor',
    'CustomMaskGenerator',
    'FlorenceMaskGenerator', 
    'FallbackMaskGenerator'
]