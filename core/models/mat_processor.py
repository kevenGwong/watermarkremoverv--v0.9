"""
MAT处理器模块
负责MAT模型的加载和inpainting处理
MAT: Mask-Aware Transformer for Large Hole Image Inpainting
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
from .base_inpainter import IOPaintBaseProcessor, ModelRegistry

logger = logging.getLogger(__name__)

class MatProcessor(IOPaintBaseProcessor):
    """MAT inpainting处理器 - 继承统一接口"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "mat")
        self._load_model()
    
# 注册MAT模型到模型注册表
ModelRegistry.register("mat", MatProcessor) 