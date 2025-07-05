"""
LaMA处理器修复版本 - 支持可选安装
"""

import logging
import time
import yaml
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)

class OptionalLamaProcessor:
    """可选的LaMA处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = None
        self.model_loaded = False
        self.available = self._check_dependencies()
        
        if self.available:
            self._load_model()
        else:
            logger.warning("⚠️ LaMA dependencies not available, processor disabled")
    
    def _check_dependencies(self) -> bool:
        """检查LaMA依赖是否可用"""
        try:
            import saicinpainting
            return True
        except ImportError:
            logger.warning("❌ saicinpainting not available")
            return False
        except Exception as e:
            logger.warning(f"❌ LaMA dependency check failed: {e}")
            return False
    
    def _load_model(self):
        """加载LaMA模型"""
        if not self.available:
            return
            
        try:
            import torch
            import cv2
            from saicinpainting.evaluation.data import pad_img_to_modulo
            from saicinpainting.evaluation.utils import move_to_device
            from saicinpainting.evaluation.data import load_image, load_mask, get_img
            from saicinpainting.training.trainers import load_checkpoint
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 获取模型配置
            model_config = self.config.get('models', {})
            model_path = model_config.get('lama_model', 'lama')
            
            # 加载模型
            train_config_path = Path(model_path) / 'config.yaml'
            model_path = Path(model_path) / 'models' / 'best.ckpt'
            
            if not model_path.exists():
                logger.error(f"LaMA model not found: {model_path}")
                raise FileNotFoundError(f"LaMA model not found: {model_path}")
            
            with open(train_config_path, 'r') as f:
                train_config = yaml.safe_load(f)
            
            train_config['model']['input_channels'] = 4
            train_config['model']['output_channels'] = 3
            
            # 创建模型
            from saicinpainting.training.data.datasets import make_default_val_dataset
            from saicinpainting.training.models import make_model
            
            model = make_model(train_config['model'], kind='inpainting')
            model.to(self.device)
            
            # 加载checkpoint
            checkpoint = load_checkpoint(train_config, model_path, strict=False, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
            
            self.model = model
            self.model_loaded = True
            logger.info(f"✅ LaMA model loaded from: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load LaMA model: {e}")
            self.model = None
            self.model_loaded = False
    
    def predict(self, image: Image.Image, mask: Image.Image, config: Dict[str, Any] = None) -> np.ndarray:
        """使用LaMA进行inpainting"""
        if not self.available:
            raise RuntimeError("LaMA not available - dependencies missing")
            
        if not self.model_loaded:
            raise RuntimeError("LaMA model not loaded")
        
        try:
            import torch
            import cv2
            from saicinpainting.evaluation.data import pad_img_to_modulo
            from saicinpainting.evaluation.utils import move_to_device
            from saicinpainting.evaluation.data import load_image, load_mask, get_img
            
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
            
            # 转换图像格式 - LaMA需要BGR输入
            image_array = np.array(image.convert("RGB"))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # RGB to BGR
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
            
            # 准备输入
            img = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float() / 255.0
            
            # 移动到设备
            img = move_to_device(img, self.device)
            mask = move_to_device(mask, self.device)
            
            # 填充到模型要求的尺寸
            img = pad_img_to_modulo(img, mod=8)
            mask = pad_img_to_modulo(mask, mod=8)
            
            # 模型推理
            with torch.no_grad():
                inpainted = self.model(img, mask)
                inpainted = torch.clamp(inpainted, 0, 1)
            
            # 后处理 - LaMA输出BGR，转换为RGB
            inpainted = inpainted.cpu().permute(0, 2, 3, 1).numpy()[0]
            inpainted = (inpainted * 255).astype(np.uint8)
            inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)  # BGR to RGB
            
            # 恢复原始尺寸
            if inpainted.shape[:2] != image_array.shape[:2]:
                inpainted = cv2.resize(inpainted, (image_array.shape[1], image_array.shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            
            return inpainted
            
        except Exception as e:
            logger.error(f"LaMA prediction failed: {e}")
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
            if self.model is not None:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
            self.model = None
            self.model_loaded = False
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ LaMA processor resources cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error during LaMA processor cleanup: {e}")
    
    def is_available(self) -> bool:
        """检查LaMA是否可用"""
        return self.available and self.model_loaded
