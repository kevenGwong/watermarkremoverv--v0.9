"""
简化LaMA处理器模块
强制使用IOPaint内置LaMA实现，与其他模型保持完全一致
符合SIMP-LAMA架构原则，代码简化到20行以内
"""

import logging
from typing import Dict, Any
from .base_inpainter import IOPaintBaseProcessor, ModelRegistry

logger = logging.getLogger(__name__)

class SimplifiedLamaProcessor(IOPaintBaseProcessor):
    """简化LaMA inpainting处理器 - 继承统一IOPaint接口"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "lama")
        self._load_model()

# 注册LaMA模型到模型注册表
ModelRegistry.register("lama", SimplifiedLamaProcessor)