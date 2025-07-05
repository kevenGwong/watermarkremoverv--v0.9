"""
IOPaint统一处理器
支持ZITS、MAT、FCF、LaMA等多种模型
修改版本：按用户需求支持ZITS、MAT、FCF + LaMA
"""

import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

# 尝试导入torch，如果失败则提供降级方案
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ PyTorch not available, IOPaint functionality will be limited")

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class IOPaintProcessor(BaseModel):
    """IOPaint统一处理器，支持多种先进模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_manager = None
        self.current_model = None
        self.available_models = ["zits", "mat", "fcf", "lama"]
        self._load_model()
    
    def _load_model(self):
        """加载IOPaint模型"""
        try:
            from iopaint.model_manager import ModelManager
            from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
            
            # 获取模型名称，默认使用MAT
            model_name = self.config.get('models', {}).get('inpaint_model', 'mat')
            
            # 确保模型在支持列表中
            if model_name not in self.available_models:
                logger.warning(f"不支持的模型 {model_name}，使用默认MAT")
                model_name = 'mat'
            
            self.model_manager = ModelManager(name=model_name, device=str(self.device))
            self.current_model = model_name
            self.register_model(self.model_manager)
            
            # 存储配置类
            self.HDStrategy = HDStrategy
            self.LDMSampler = LDMSampler
            self.Config = Config
            
            self.model_loaded = True
            logger.info(f"✅ IOPaint模型加载成功: {model_name}")
            logger.info(f"   设备: {self.device}")
            logger.info(f"   支持的模型: {self.available_models}")
            
        except Exception as e:
            logger.error(f"❌ IOPaint模型加载失败: {e}")
            self.model_loaded = False
            raise
    
    def switch_model(self, model_name: str):
        """动态切换模型"""
        if model_name == self.current_model:
            return
            
        if model_name not in self.available_models:
            logger.error(f"不支持的模型: {model_name}，支持的模型: {self.available_models}")
            return
            
        try:
            from iopaint.model_manager import ModelManager
            
            # 清理旧模型
            if self.model_manager:
                del self.model_manager
                if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清理GPU内存
                
            # 加载新模型
            self.model_manager = ModelManager(name=model_name, device=str(self.device))
            self.current_model = model_name
            self.register_model(self.model_manager)
            
            logger.info(f"🔄 模型切换成功: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ 模型切换失败: {e}")
            raise
    
    def predict(self, 
                image: Union[Image.Image, np.ndarray], 
                mask: Union[Image.Image, np.ndarray],
                custom_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """执行Inpainting预测"""
        
        if not self.model_loaded:
            raise RuntimeError("IOPaint模型未加载")
        
        # 验证输入
        image, mask = self.validate_inputs(image, mask)
        
        # 获取处理参数
        params = self._get_processing_params(custom_config)
        
        # 智能模型选择（如果启用）
        if params.get('auto_model_selection', True):
            optimal_model = self._choose_optimal_model(image, mask, params)
            if optimal_model != self.current_model:
                self.switch_model(optimal_model)
        
        # 手动模型选择（如果指定）
        if 'force_model' in params and params['force_model'] in self.available_models:
            if params['force_model'] != self.current_model:
                self.switch_model(params['force_model'])
        
        try:
            # 处理图像格式
            if isinstance(image, Image.Image):
                image_rgb = np.array(image.convert("RGB"))
            else:
                image_rgb = image
                
            if isinstance(mask, Image.Image):
                mask_gray = np.array(mask.convert("L"))
            else:
                mask_gray = mask
            
            logger.info(f"🎨 使用{self.current_model}模型处理: {image_rgb.shape}")
            
            # 构建IOPaint配置
            config = self._build_iopaint_config(params)
            
            # 执行inpainting
            result = self.model_manager(image_rgb, mask_gray, config)
            
            logger.info(f"✅ {self.current_model}处理完成")
            return result
            
        except Exception as e:
            logger.error(f"❌ {self.current_model}处理失败: {e}")
            raise
    
    def _choose_optimal_model(self, image, mask, params) -> str:
        """智能选择最优模型"""
        
        # 计算mask覆盖率
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask.convert("L"))
        else:
            mask_array = mask
            
        mask_coverage = np.sum(mask_array > 128) / mask_array.size * 100
        
        # 获取图像复杂度（边缘密度）
        image_complexity = self._calculate_image_complexity(image)
        
        # 智能选择策略（根据用户需求调整）
        if mask_coverage > 30:
            return 'mat'      # 大水印用MAT（最佳质量）
        elif image_complexity > 0.7:
            return 'zits'     # 复杂结构用ZITS（最佳结构保持）
        elif mask_coverage < 5:
            return 'lama'     # 小水印用LaMA（最快速度）
        else:
            return 'fcf'      # 中等情况用FCF（快速修复）
    
    def _calculate_image_complexity(self, image) -> float:
        """计算图像复杂度"""
        # 简单的边缘密度计算
        import cv2
        
        if isinstance(image, Image.Image):
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        edges = cv2.Canny(gray, 50, 150)
        complexity = np.sum(edges > 0) / edges.size
        return complexity
    
    def _build_iopaint_config(self, params):
        """构建IOPaint配置"""
        
        strategy_map = {
            'CROP': self.HDStrategy.CROP,
            'RESIZE': self.HDStrategy.RESIZE,
            'ORIGINAL': self.HDStrategy.ORIGINAL
        }
        
        config = self.Config(
            ldm_steps=params.get('ldm_steps', 50),
            ldm_sampler=self.LDMSampler.ddim,
            hd_strategy=strategy_map.get(params.get('hd_strategy', 'CROP')),
            hd_strategy_crop_margin=params.get('hd_strategy_crop_margin', 64),
            hd_strategy_crop_trigger_size=params.get('hd_strategy_crop_trigger_size', 1024),
            hd_strategy_resize_limit=params.get('hd_strategy_resize_limit', 2048),
        )
        
        return config
    
    def _get_processing_params(self, custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """获取处理参数"""
        
        default_params = {
            'ldm_steps': 50,
            'hd_strategy': 'CROP',
            'hd_strategy_crop_margin': 64,
            'hd_strategy_crop_trigger_size': 1024,
            'hd_strategy_resize_limit': 2048,
            'auto_model_selection': True,
        }
        
        if custom_config:
            default_params.update(custom_config)
            
        return default_params
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            "model_type": "IOPaint_Unified",
            "current_model": self.current_model,
            "supported_models": self.available_models,
            "intelligent_selection": True,
            "framework": "IOPaint",
            "model_descriptions": {
                "zits": "最佳结构保持，适合复杂图像",
                "mat": "最佳质量，适合大水印",
                "fcf": "快速修复，平衡性能",
                "lama": "最快速度，适合小水印"
            }
        })
        return info
    
    def get_available_models(self) -> list:
        """获取可用模型列表"""
        return self.available_models
    
    def get_current_model(self) -> str:
        """获取当前使用的模型"""
        return self.current_model