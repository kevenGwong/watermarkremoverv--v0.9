"""
简化的Mask生成器模块
移除Florence-2相关复杂性，专注于自定义mask生成
"""

import time
import logging
import yaml
import io
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class CustomMaskGenerator:
    """自定义mask生成器 - 基于 Watermark_sam 模型"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """加载自定义分割模型"""
        try:
            import torch
            import segmentation_models_pytorch as smp
            import albumentations as A
            import cv2
            from albumentations.pytorch import ToTensorV2
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 定义模型架构
            class WMModel(torch.nn.Module):
                def __init__(self, freeze_encoder=True):
                    super().__init__()
                    self.net = smp.create_model("FPN", encoder_name="mit_b5", in_channels=3, classes=1)
                    if freeze_encoder:
                        for p in self.net.encoder.parameters():
                            p.requires_grad = False

                def forward(self, x):
                    return self.net(x)
            
            # 加载模型
            self.model = WMModel(freeze_encoder=False).to(self.device)
            
            # 加载checkpoint
            mask_config = self.config.get('mask_generator', {})
            ckpt_path = mask_config.get('mask_model_path')
            
            if not ckpt_path or not Path(ckpt_path).exists():
                logger.warning(f"Custom mask model not found: {ckpt_path}")
                logger.warning("Custom mask generation will use fallback")
                self.model = None
                return
                
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
            self.model.net.load_state_dict(state_dict)
            self.model.eval()
            
            # Setup preprocessing with defaults
            self.image_size = mask_config.get('image_size', 768)
            self.imagenet_mean = mask_config.get('imagenet_mean', [0.485, 0.456, 0.406])
            self.imagenet_std = mask_config.get('imagenet_std', [0.229, 0.224, 0.225])
            self.mask_threshold = mask_config.get('mask_threshold', 0.5)
            
            self.aug_val = A.Compose([
                A.LongestMaxSize(max_size=self.image_size),
                A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std, max_pixel_value=255.0),
                ToTensorV2(),
            ])
            
            logger.info(f"✅ Custom mask model loaded from: {ckpt_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load custom mask model: {e}")
            self._cleanup_model()
    
    def _cleanup_model(self):
        """清理模型资源"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
            self.model = None
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")
    
    def is_available(self) -> bool:
        """检查自定义mask生成器是否可用"""
        return self.model is not None
    
    def generate_mask(self, image: Image.Image, mask_params: Dict[str, Any] = None) -> Image.Image:
        """生成水印mask"""
        if not self.is_available():
            logger.warning("Custom mask model not available, using fallback")
            return self._generate_fallback_mask(image)
        
        try:
            import torch
            import cv2
            
            # 使用动态参数或默认值
            if mask_params is None:
                mask_params = {}
                
            mask_threshold = mask_params.get('mask_threshold', self.mask_threshold)
            dilate_size = mask_params.get('mask_dilate_kernel_size', 3)
            dilate_iterations = mask_params.get('mask_dilate_iterations', 1)
            
            # 转换为numpy数组
            image_rgb = np.array(image.convert("RGB"))
            orig_h, orig_w = image_rgb.shape[:2]
            
            # 预处理
            sample = self.aug_val(image=image_rgb, mask=None)
            img_tensor = sample["image"].unsqueeze(0).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                pred_mask = self.model(img_tensor)
                pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
            
            # 后处理
            scale = min(self.image_size / orig_h, self.image_size / orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            
            pad_top = (self.image_size - new_h) // 2
            pad_left = (self.image_size - new_w) // 2
            
            pred_crop = pred_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
            pred_mask = cv2.resize(pred_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # 二值化
            binary_mask = (pred_mask > mask_threshold).astype(np.uint8) * 255
            
            # 膨胀处理
            if dilate_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=dilate_iterations)
            
            # 验证mask质量
            mask_coverage = np.sum(binary_mask > 0) / (orig_w * orig_h) * 100
            logger.info(f"🎭 Custom mask coverage: {mask_coverage:.2f}%")
            
            return Image.fromarray(binary_mask, mode='L')
            
        except Exception as e:
            logger.error(f"Custom mask generation failed: {e}")
            return self._generate_fallback_mask(image)
    
    def _generate_fallback_mask(self, image: Image.Image) -> Image.Image:
        """生成降级mask - 简单的中心区域"""
        logger.warning("Using fallback mask generation")
        
        width, height = image.size
        mask = Image.new('L', (width, height), 0)
        
        # 创建中心区域mask作为降级方案
        center_x, center_y = width // 2, height // 2
        mask_size = min(width, height) // 4
        
        import numpy as np
        mask_array = np.array(mask)
        
        # 简单的椭圆形区域
        y, x = np.ogrid[:height, :width]
        ellipse_mask = ((x - center_x) ** 2 / (mask_size ** 2) + 
                       (y - center_y) ** 2 / (mask_size ** 2)) <= 1
        mask_array[ellipse_mask] = 255
        
        return Image.fromarray(mask_array, mode='L')
    
    def cleanup_resources(self):
        """清理资源"""
        self._cleanup_model()
        logger.info("✅ CustomMaskGenerator resources cleaned up")

class SimpleMaskGenerator:
    """简单mask生成器 - 用于测试和降级场景"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def is_available(self) -> bool:
        """始终可用"""
        return True
    
    def generate_mask(self, image: Image.Image, mask_params: Dict[str, Any] = None) -> Image.Image:
        """生成简单的中心区域mask"""
        width, height = image.size
        mask = Image.new('L', (width, height), 0)
        
        # 获取参数
        if mask_params is None:
            mask_params = {}
            
        coverage_percent = mask_params.get('coverage_percent', 25)  # 默认覆盖25%
        
        import numpy as np
        mask_array = np.array(mask)
        
        # 计算mask区域大小
        area = width * height
        target_area = area * coverage_percent / 100
        radius = int(np.sqrt(target_area / np.pi))
        
        center_x, center_y = width // 2, height // 2
        
        # 创建圆形mask
        y, x = np.ogrid[:height, :width]
        circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask_array[circle_mask] = 255
        
        logger.info(f"🎭 Simple mask generated: {coverage_percent}% coverage")
        return Image.fromarray(mask_array, mode='L')
    
    def cleanup_resources(self):
        """清理资源（无资源需要清理）"""
        pass