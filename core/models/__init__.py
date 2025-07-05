"""
模型模块
包含所有AI模型相关的处理器
"""

from .lama_processor_simplified import SimplifiedLamaProcessor
from .zits_processor import ZitsProcessor
from .mat_processor import MatProcessor
from .fcf_processor import FcfProcessor
from .unified_mask_generator import UnifiedMaskGenerator

# 兼容性别名
UnifiedProcessor = SimplifiedLamaProcessor  # 统一处理器指向简化LaMA处理器
CustomMaskGenerator = UnifiedMaskGenerator
SimpleMaskGenerator = UnifiedMaskGenerator
FlorenceMaskGenerator = UnifiedMaskGenerator  # 临时别名
FallbackMaskGenerator = UnifiedMaskGenerator  # 临时别名

__all__ = [
    'SimplifiedLamaProcessor',
    'ZitsProcessor',
    'MatProcessor', 
    'FcfProcessor',
    'UnifiedMaskGenerator',
    'UnifiedProcessor',
    'CustomMaskGenerator',
    'SimpleMaskGenerator',
    'FlorenceMaskGenerator',
    'FallbackMaskGenerator',
]