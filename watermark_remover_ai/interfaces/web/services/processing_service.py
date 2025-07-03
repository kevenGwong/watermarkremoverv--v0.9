"""
处理服务
"""

import time
import logging
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from watermark_remover_ai.core.processors.watermark_processor import WatermarkProcessor
from watermark_remover_ai.core.processors.processing_result import ProcessingResult

logger = logging.getLogger(__name__)

class ProcessingService:
    """处理服务"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.processor = None
        self._load_processor()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "models": {
                "lama_model": "lama"
            },
            "mask_generator": {
                "model_type": "custom"
            },
            "interfaces": {
                "web": {"port": 8501, "host": "localhost"}
            }
        }
    
    def _load_processor(self):
        """加载处理器"""
        try:
            self.processor = WatermarkProcessor(self.config)
            logger.info("WatermarkProcessor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            self.processor = None
    
    def process_image(self, image: Image.Image, mask_model: str, 
                     mask_params: Dict[str, Any], inpaint_params: Dict[str, Any],
                     performance_params: Dict[str, Any]) -> ProcessingResult:
        """处理图像"""
        if self.processor is None:
            return ProcessingResult(
                success=False,
                error_message="Processor not loaded",
                processing_time=0.0
            )
        
        try:
            # 调用实际的处理器
            result = self.processor.process_image(
                image=image,
                mask_method=mask_model,
                transparent=inpaint_params.get('transparent', False),
                max_bbox_percent=mask_params.get('max_bbox_percent', 10.0),
                custom_inpaint_config=inpaint_params
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=0.0
            ) 