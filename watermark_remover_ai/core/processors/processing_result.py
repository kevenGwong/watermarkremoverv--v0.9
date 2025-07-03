"""
处理结果数据类
"""

from dataclasses import dataclass
from typing import Optional
from PIL import Image

@dataclass
class ProcessingResult:
    """结果数据类"""
    success: bool
    result_image: Optional[Image.Image] = None
    mask_image: Optional[Image.Image] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None 