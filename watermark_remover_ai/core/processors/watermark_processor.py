"""
水印处理主类 - 模块化版本
"""

import logging
import time
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw

from .processing_result import ProcessingResult
from ..models.mask_generators import CustomMaskGenerator, FlorenceMaskGenerator
from ..models.lama_inpainter import LamaInpainter
from ..utils.config_utils import load_config

logger = logging.getLogger(__name__)

class WatermarkProcessor:
    """水印处理主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化mask生成器
        mask_type = self.config.get('mask_generator', {}).get('model_type', 'custom')
        if mask_type == "custom":
            self.mask_generator = CustomMaskGenerator(self.config)
        else:
            self.mask_generator = FlorenceMaskGenerator(self.config)
        
        # 初始化inpainting模型
        self.model_manager = None
        self._load_lama_model()
    
    def _load_lama_model(self):
        """加载LaMA inpainting模型"""
        try:
            self.lama_inpainter = LamaInpainter(self.config)
            logger.info("LaMA model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LaMA model: {e}")
            raise
    
    def process_image(self, 
                     image: Image.Image,
                     mask_method: str = "custom",
                     transparent: bool = False,
                     max_bbox_percent: float = 10.0,
                     force_format: Optional[str] = None,
                     custom_inpaint_config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """处理单张图片"""
        start_time = time.time()
        
        try:
            # 生成mask
            if mask_method == "custom" or isinstance(self.mask_generator, CustomMaskGenerator):
                mask_image = self.mask_generator.generate_mask(image)
            else:
                mask_image = self.mask_generator.generate_mask(image, max_bbox_percent)
            
            if transparent:
                # 透明处理
                result_image = self._make_region_transparent(image, mask_image)
            else:
                # LaMA inpainting (支持自定义配置)
                result_image = self._process_with_lama(image, mask_image, custom_inpaint_config)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                result_image=result_image,
                mask_image=mask_image,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _make_region_transparent(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """使区域透明"""
        image = image.convert("RGBA")
        mask = mask.convert("L")
        transparent_image = Image.new("RGBA", image.size)
        
        for x in range(image.width):
            for y in range(image.height):
                if mask.getpixel((x, y)) > 0:
                    transparent_image.putpixel((x, y), (0, 0, 0, 0))
                else:
                    transparent_image.putpixel((x, y), image.getpixel((x, y)))
        
        return transparent_image
    
    def _process_with_lama(self, image: Image.Image, mask: Image.Image, custom_config: Optional[Dict[str, Any]] = None) -> Image.Image:
        """使用LaMA进行inpainting"""
        try:
            # 使用我们的LamaInpainter
            result_image = self.lama_inpainter.inpaint_image(image, mask, custom_config)
            return result_image
            
        except Exception as e:
            logger.error(f"LaMA inpainting failed: {e}, using fallback method")
            # 简单的fallback方法
            return self._simple_inpaint(image, mask)
    
    def _simple_inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """简单的inpainting fallback方法"""
        # 使用OpenCV的简单inpainting
        image_np = np.array(image.convert("RGB"))
        mask_np = np.array(mask.convert("L"))
        
        # 确保mask是二值的
        _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        
        # 使用TELEA算法进行inpainting
        result = cv2.inpaint(image_np, mask_np, 3, cv2.INPAINT_TELEA)
        
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            import psutil
            
            info = {
                "cuda_available": torch.cuda.is_available(),
                "device": str(self.device),
                "ram_usage": f"{psutil.virtual_memory().percent:.1f}%",
                "cpu_usage": f"{psutil.cpu_percent():.1f}%"
            }
            
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_properties(0)
                vram_total = gpu_info.total_memory // (1024 ** 2)
                vram_used = vram_total - (torch.cuda.memory_reserved(0) // (1024 ** 2))
                info["vram_usage"] = f"{vram_used}/{vram_total} MB"
            else:
                info["vram_usage"] = "N/A"
            
            return info
        except ImportError:
            return {
                "cuda_available": torch.cuda.is_available(),
                "device": str(self.device),
                "ram_usage": "N/A",
                "cpu_usage": "N/A",
                "vram_usage": "N/A"
            } 