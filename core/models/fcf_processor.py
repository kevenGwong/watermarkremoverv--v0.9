"""
FCF处理器模块
负责FCF模型的加载和inpainting处理
FCF: Fast Context-Free Inpainting
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
from .base_inpainter import IOPaintBaseProcessor, ModelRegistry

logger = logging.getLogger(__name__)

class FcfProcessor(IOPaintBaseProcessor):
    """FCF inpainting处理器 - 继承统一接口"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "fcf")
        self._load_model()
    
# 注册FCF模型到模型注册表
ModelRegistry.register("fcf", FcfProcessor) 