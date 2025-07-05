"""
ZITS处理器模块
负责ZITS模型的加载和inpainting处理
ZITS: Zero-shot Image-to-Image Translation for Structure-aware Image Inpainting
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
from .base_inpainter import IOPaintBaseProcessor, ModelRegistry

logger = logging.getLogger(__name__)

class ZitsProcessor(IOPaintBaseProcessor):
    """ZITS inpainting处理器 - 继承统一接口"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "zits")
        self._load_model()
    
# 注册ZITS模型到模型注册表
ModelRegistry.register("zits", ZitsProcessor) 