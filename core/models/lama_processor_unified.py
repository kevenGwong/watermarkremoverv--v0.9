"""
LaMA处理器统一版本
实现与其他IOPaint模型统一的接口，同时支持可选安装
符合SIMP-LAMA原则的实现
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
from .base_inpainter import BaseInpainter, ModelRegistry

logger = logging.getLogger(__name__)

class LamaProcessor(BaseInpainter):
    """LaMA inpainting处理器 - 可选依赖实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = "lama"
        self.saicinpainting_available = self._check_saicinpainting()
        
        if self.saicinpainting_available:
            self._load_model()
        else:
            logger.warning("⚠️ LaMA使用IOPaint fallback模式（saicinpainting不可用）")
            self._load_iopaint_fallback()
    
    def _check_saicinpainting(self) -> bool:
        """检查saicinpainting依赖是否可用"""
        try:
            import saicinpainting
            logger.info("✅ saicinpainting available - 使用原生LaMA")
            return True
        except ImportError:
            logger.info("ℹ️ saicinpainting不可用 - 将使用IOPaint的LaMA实现")
            return False
        except Exception as e:
            logger.warning(f"⚠️ saicinpainting检查失败: {e}")
            return False
    
    def _load_model(self):
        """加载LaMA模型"""
        if self.saicinpainting_available:
            self._load_native_lama()
        else:
            self._load_iopaint_fallback()
    
    def _load_native_lama(self):
        """加载原生saicinpainting LaMA模型"""
        try:
            import torch
            import yaml
            from pathlib import Path
            from saicinpainting.training.trainers import load_checkpoint
            from saicinpainting.training.models import make_model
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 获取模型路径
            model_config = self.config.get('models', {})
            model_path = model_config.get('lama_model_path', 'lama')
            
            # 配置文件路径
            train_config_path = Path(model_path) / 'config.yaml'
            checkpoint_path = Path(model_path) / 'models' / 'best.ckpt'
            
            if not checkpoint_path.exists():
                logger.error(f"LaMA checkpoint not found: {checkpoint_path}")
                raise FileNotFoundError(f"LaMA model not found: {checkpoint_path}")
            
            # 加载配置
            with open(train_config_path, 'r') as f:
                train_config = yaml.safe_load(f)
            
            train_config['model']['input_channels'] = 4
            train_config['model']['output_channels'] = 3
            
            # 创建并加载模型
            model = make_model(train_config['model'], kind='inpainting')
            model.to(self.device)
            
            checkpoint = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
            
            self.model = model
            self.model_loaded = True
            logger.info(f"✅ 原生LaMA模型加载成功: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"❌ 原生LaMA模型加载失败: {e}")
            logger.info("🔄 尝试使用IOPaint fallback模式...")
            self.saicinpainting_available = False
            self._load_iopaint_fallback()
    
    def _load_iopaint_fallback(self):
        """加载IOPaint的LaMA实现作为fallback"""
        try:
            import torch
            from iopaint.model_manager import ModelManager
            from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 使用IOPaint的LaMA实现
            self.model_manager = ModelManager(name="lama", device=str(self.device))
            self.HDStrategy = HDStrategy
            self.LDMSampler = LDMSampler
            self.Config = Config
            
            self.model_loaded = True
            logger.info("✅ IOPaint LaMA fallback模式加载成功")
            
        except Exception as e:
            logger.error(f"❌ IOPaint LaMA fallback加载失败: {e}")
            self.model_loaded = False
            raise
    
    def predict(self, image: Image.Image, mask: Image.Image, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """统一的LaMA推理接口"""
        if not self.model_loaded:
            raise RuntimeError("LaMA model not loaded")
        
        # 验证输入
        if not self.validate_inputs(image, mask):
            raise ValueError("Invalid inputs")
        
        # 预处理
        image, mask = self.preprocess_inputs(image, mask)
        
        if self.saicinpainting_available:
            return self._predict_native_lama(image, mask, config)
        else:
            return self._predict_iopaint_lama(image, mask, config)
    
    def _predict_native_lama(self, image: Image.Image, mask: Image.Image, config: Dict[str, Any]) -> np.ndarray:
        """使用原生saicinpainting进行推理"""
        try:
            import torch
            import cv2
            from saicinpainting.evaluation.data import pad_img_to_modulo
            from saicinpainting.evaluation.utils import move_to_device
            
            if config is None:
                config = {}
            
            # LaMA特定参数
            hd_strategy = config.get('hd_strategy', 'CROP')
            hd_strategy_crop_margin = config.get('hd_strategy_crop_margin', 64)
            hd_strategy_crop_trigger_size = config.get('hd_strategy_crop_trigger_size', 1024)
            hd_strategy_resize_limit = config.get('hd_strategy_resize_limit', 2048)
            
            from ..utils.image_utils import ImageUtils
            
            # LaMA需要BGR输入，使用专门的预处理
            image_array, mask_array = ImageUtils.prepare_arrays_for_lama(image, mask)
            
            # 高分辨率处理策略
            original_size = image_array.shape[:2]
            if hd_strategy == 'CROP' and max(original_size) > hd_strategy_crop_trigger_size:
                image_array, mask_array = self._crop_for_inpainting(
                    image_array, mask_array, hd_strategy_crop_margin
                )
            elif hd_strategy == 'RESIZE' and max(original_size) > hd_strategy_resize_limit:
                scale = hd_strategy_resize_limit / max(original_size)
                new_h, new_w = int(original_size[0] * scale), int(original_size[1] * scale)
                image_array = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask_array = cv2.resize(mask_array, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # 准备张量输入
            img_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float() / 255.0
            
            # 移动到设备
            img_tensor = move_to_device(img_tensor, self.device)
            mask_tensor = move_to_device(mask_tensor, self.device)
            
            # 填充到模型要求的尺寸
            img_tensor = pad_img_to_modulo(img_tensor, mod=8)
            mask_tensor = pad_img_to_modulo(mask_tensor, mod=8)
            
            # 模型推理
            with torch.no_grad():
                inpainted = self.model(img_tensor, mask_tensor)
                inpainted = torch.clamp(inpainted, 0, 1)
            
            # 后处理
            result = inpainted.cpu().permute(0, 2, 3, 1).numpy()[0]
            result = (result * 255).astype(np.uint8)
            
            # LaMA输出BGR，转换为RGB
            result = ImageUtils.postprocess_lama_result(result)
            
            # 恢复原始尺寸
            if result.shape[:2] != original_size:
                result = cv2.resize(result, (original_size[1], original_size[0]), 
                                  interpolation=cv2.INTER_LINEAR)
            
            logger.info("✅ 原生LaMA推理完成")
            return result
            
        except Exception as e:
            logger.error(f"原生LaMA推理失败: {e}")
            raise
    
    def _predict_iopaint_lama(self, image: Image.Image, mask: Image.Image, config: Dict[str, Any]) -> np.ndarray:
        """使用IOPaint LaMA进行推理"""
        try:
            from ..utils.image_utils import ImageUtils
            
            # IOPaint LaMA fallback使用标准RGB处理
            image_array, mask_array = ImageUtils.prepare_arrays_for_iopaint(image, mask)
            
            logger.info(f"🎨 IOPaint LaMA processing: {image_array.shape}")
            
            # 构建IOPaint配置
            iopaint_config = self._build_iopaint_config(config or {})
            
            # 执行推理
            result = self.model_manager(image_array, mask_array, iopaint_config)
            
            logger.info("✅ IOPaint LaMA推理完成")
            return result
            
        except Exception as e:
            logger.error(f"IOPaint LaMA推理失败: {e}")
            raise
    
    def _build_iopaint_config(self, config: Dict[str, Any]) -> object:
        """构建IOPaint配置对象"""
        # 默认参数
        default_config = {
            'ldm_steps': 50,
            'ldm_sampler': 'ddim',
            'hd_strategy': 'CROP',
            'hd_strategy_crop_margin': 64,
            'hd_strategy_crop_trigger_size': 1024,
            'hd_strategy_resize_limit': 2048
        }
        
        # 合并用户配置
        merged_config = {**default_config, **config}
        
        # 映射策略
        strategy_map = {
            'CROP': self.HDStrategy.CROP,
            'RESIZE': self.HDStrategy.RESIZE,
            'ORIGINAL': self.HDStrategy.ORIGINAL
        }
        
        # 映射采样器
        sampler_map = {
            'ddim': self.LDMSampler.ddim,
            'pndm': self.LDMSampler.pndm,
            'k_euler': self.LDMSampler.k_euler,
            'k_euler_a': self.LDMSampler.k_euler_a
        }
        
        return self.Config(
            ldm_steps=merged_config['ldm_steps'],
            ldm_sampler=sampler_map.get(merged_config['ldm_sampler'], self.LDMSampler.ddim),
            hd_strategy=strategy_map.get(merged_config['hd_strategy'], self.HDStrategy.CROP),
            hd_strategy_crop_margin=merged_config['hd_strategy_crop_margin'],
            hd_strategy_crop_trigger_size=merged_config['hd_strategy_crop_trigger_size'],
            hd_strategy_resize_limit=merged_config['hd_strategy_resize_limit']
        )
    
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
        """清理LaMA资源"""
        try:
            if self.saicinpainting_available:
                if hasattr(self, 'model') and self.model is not None:
                    if hasattr(self.model, 'cpu'):
                        self.model.cpu()
                    del self.model
                self.model = None
            else:
                if hasattr(self, 'model_manager') and self.model_manager is not None:
                    del self.model_manager
                self.model_manager = None
            
            self.model_loaded = False
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ LaMA processor resources cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during LaMA processor cleanup: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取LaMA模型信息"""
        info = super().get_model_info()
        info.update({
            'model_name': self.model_name,
            'saicinpainting_available': self.saicinpainting_available,
            'mode': 'native' if self.saicinpainting_available else 'iopaint_fallback'
        })
        return info

# 注册LaMA模型到模型注册表
ModelRegistry.register("lama", LamaProcessor)