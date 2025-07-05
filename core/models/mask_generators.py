"""
Mask生成器模块
包含自定义mask生成器和Florence-2 mask生成器
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
            mask_config = self.config['mask_generator']
            ckpt_path = mask_config['mask_model_path']
            
            if not Path(ckpt_path).exists():
                logger.error(f"Custom mask model not found: {ckpt_path}")
                logger.error("Custom mask generation will not be available")
                self.model = None
                raise FileNotFoundError(f"Custom mask model not found: {ckpt_path}")
                
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
            self.model.net.load_state_dict(state_dict)
            self.model.eval()
            
            # Setup preprocessing
            mask_config = self.config['mask_generator']
            self.image_size = mask_config['image_size']
            self.imagenet_mean = mask_config['imagenet_mean']
            self.imagenet_std = mask_config['imagenet_std']
            self.mask_threshold = mask_config['mask_threshold']
            
            self.aug_val = A.Compose([
                A.LongestMaxSize(max_size=self.image_size),
                A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std, max_pixel_value=255.0),
                ToTensorV2(),
            ])
            
            logger.info(f"✅ Custom mask model loaded from: {ckpt_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load custom mask model: {e}")
            # Clean up any partially loaded resources
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def generate_mask(self, image: Image.Image, mask_params: Dict[str, Any] = None) -> Image.Image:
        """生成水印mask"""
        if self.model is None:
            logger.warning("Custom mask model not available, returning empty mask")
            return Image.new('L', image.size, 0)
        
        try:
            import torch
            import cv2
            
            # 使用动态参数或默认值
            mask_threshold = mask_params.get('mask_threshold', self.mask_threshold) if mask_params else self.mask_threshold
            dilate_size = mask_params.get('mask_dilate_kernel_size', 3) if mask_params else 3
            dilate_iterations = mask_params.get('mask_dilate_iterations', 1) if mask_params else 1
            
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
            
            return Image.fromarray(binary_mask, mode='L')
            
        except Exception as e:
            logger.error(f"Custom mask generation failed: {e}")
            return Image.new('L', image.size, 0)

class FlorenceMaskGenerator:
    """Florence-2 mask生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.florence_model = None
        self.florence_processor = None
        self._load_model()
    
    def _load_model(self):
        """加载Florence-2模型"""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = self.config['models']['florence_model']
            
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to(device).eval()
            self.florence_processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.info(f"✅ Florence-2 model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Florence-2 model: {e}")
            self.florence_model = None
            self.florence_processor = None
    
    def is_available(self) -> bool:
        """检查Florence-2模型是否可用"""
        return self.florence_model is not None and self.florence_processor is not None
    
    def generate_mask(self, image: Image.Image, mask_params: Dict[str, Any] = None) -> Image.Image:
        """使用Florence-2生成mask"""
        if self.florence_model is None:
            logger.warning("Florence-2 model not available, returning empty mask")
            return Image.new('L', image.size, 0)
        
        try:
            # 获取参数
            max_bbox_percent = mask_params.get('max_bbox_percent', 10.0) if mask_params else 10.0
            detection_prompt = mask_params.get('detection_prompt', 'watermark') if mask_params else 'watermark'
            
            # TODO: 实现Florence-2检测逻辑
            # 由于缺少utils模块，暂时返回空mask
            logger.warning("Florence-2 detection logic not implemented yet")
            return Image.new('L', image.size, 0)
            
        except Exception as e:
            logger.error(f"Florence mask generation failed: {e}")
            return Image.new('L', image.size, 0)

class FallbackMaskGenerator:
    """降级mask生成器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config
    
    def generate_mask(self, image: Image.Image, mask_params: Dict[str, Any] = None) -> Image.Image:
        logger.warning("Using fallback mask generator - returning empty mask")
        return Image.new('L', image.size, 0) 