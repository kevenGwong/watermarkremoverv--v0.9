"""
FCF处理器模块
负责FCF模型的加载和inpainting处理
FCF: Fast Context-Free Inpainting
"""

import logging
import time
import yaml
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)

class FcfProcessor:
    """FCF inpainting处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """加载FCF模型"""
        try:
            import torch
            import cv2
            from iopaint.model_manager import ModelManager
            from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 加载FCF模型
            self.model_manager = ModelManager(name="fcf", device=str(self.device))
            
            # 存储配置类
            self.HDStrategy = HDStrategy
            self.LDMSampler = LDMSampler
            self.Config = Config
            
            self.model_loaded = True
            logger.info(f"✅ FCF model loaded successfully")
            logger.info(f"   Device: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FCF model: {e}")
            self.model = None
            self.model_loaded = False
    
    def predict(self, image: Image.Image, mask: Image.Image, config: Dict[str, Any] = None) -> np.ndarray:
        """使用FCF进行inpainting"""
        if not self.model_loaded:
            raise RuntimeError("FCF model not loaded")
        
        try:
            import torch
            import cv2
            
            # 使用默认配置或自定义配置
            if config is None:
                config = {}
            
            # 获取参数
            ldm_steps = config.get('ldm_steps', 50)
            ldm_sampler = config.get('ldm_sampler', 'ddim')
            hd_strategy = config.get('hd_strategy', 'CROP')
            hd_strategy_crop_margin = config.get('hd_strategy_crop_margin', 64)
            hd_strategy_crop_trigger_size = config.get('hd_strategy_crop_trigger_size', 1024)
            hd_strategy_resize_limit = config.get('hd_strategy_resize_limit', 2048)
            
            # 转换图像格式
            image_array = np.array(image.convert("RGB"))
            mask_array = np.array(mask.convert("L"))
            
            # 确保mask是二值的
            mask_array = (mask_array > 128).astype(np.uint8) * 255
            
            # 处理高分辨率图像
            if hd_strategy == 'CROP' and max(image_array.shape[:2]) > hd_strategy_crop_trigger_size:
                # 裁剪策略
                image_array, mask_array = self._crop_for_inpainting(
                    image_array, mask_array, hd_strategy_crop_margin
                )
            elif hd_strategy == 'RESIZE' and max(image_array.shape[:2]) > hd_strategy_resize_limit:
                # 缩放策略
                scale = hd_strategy_resize_limit / max(image_array.shape[:2])
                new_h, new_w = int(image_array.shape[0] * scale), int(image_array.shape[1] * scale)
                image_array = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask_array = cv2.resize(mask_array, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            logger.info(f"🎨 FCF processing: {image_array.shape}")
            
            # 构建IOPaint配置
            strategy_map = {
                'CROP': self.HDStrategy.CROP,
                'RESIZE': self.HDStrategy.RESIZE,
                'ORIGINAL': self.HDStrategy.ORIGINAL
            }
            
            iopaint_config = self.Config(
                ldm_steps=ldm_steps,
                ldm_sampler=self.LDMSampler.ddim,
                hd_strategy=strategy_map.get(hd_strategy, self.HDStrategy.CROP),
                hd_strategy_crop_margin=hd_strategy_crop_margin,
                hd_strategy_crop_trigger_size=hd_strategy_crop_trigger_size,
                hd_strategy_resize_limit=hd_strategy_resize_limit,
            )
            
            # 执行FCF inpainting
            result = self.model_manager(image_array, mask_array, iopaint_config)
            
            # 恢复原始尺寸
            if result.shape[:2] != image_array.shape[:2]:
                result = cv2.resize(result, (image_array.shape[1], image_array.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
            
            logger.info(f"✅ FCF processing completed")
            return result
            
        except Exception as e:
            logger.error(f"FCF prediction failed: {e}")
            raise
    
    def _crop_for_inpainting(self, image: np.ndarray, mask: np.ndarray, margin: int) -> tuple:
        """为inpainting裁剪图像"""
        import cv2
        
        # 找到mask的边界框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, mask
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contours[0])
        for contour in contours[1:]:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            x = min(x, x1)
            y = min(y, y1)
            w = max(w, x1 + w1 - x)
            h = max(h, y1 + h1 - y)
        
        # 添加边距
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # 裁剪
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
        return cropped_image, cropped_mask
    
    def cleanup_resources(self):
        """清理资源"""
        try:
            if self.model_manager is not None:
                del self.model_manager
            self.model_manager = None
            self.model_loaded = False
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ FCF processor resources cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error during FCF processor cleanup: {e}") 