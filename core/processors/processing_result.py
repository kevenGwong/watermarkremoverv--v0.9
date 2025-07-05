"""
处理结果数据类
"""

from dataclasses import dataclass
from typing import Optional
from PIL import Image

@dataclass
class ProcessingResult:
    """处理结果数据类 - 增强版本支持模型信息和内存监控"""
    success: bool
    result_image: Optional[Image.Image] = None
    mask_image: Optional[Image.Image] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    model_used: Optional[str] = None
    memory_info: Optional[dict] = None
    mask_coverage: float = 0.0
    image_info: Optional[dict] = None 