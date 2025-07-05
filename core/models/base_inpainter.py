"""
基础Inpainting模型接口
定义所有模型必须实现的统一接口
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class BaseInpainter(ABC):
    """
    基础Inpainting模型抽象类
    所有IOPaint模型都必须实现这个接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_loaded = False
        self.device = None
    
    @abstractmethod
    def _load_model(self):
        """加载模型 - 子类必须实现"""
        pass
    
    @abstractmethod
    def predict(self, image: Image.Image, mask: Image.Image, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        统一的inpainting推理接口
        
        Args:
            image: 输入图像 (PIL.Image)
            mask: 输入mask (PIL.Image, 灰度图)
            config: 推理配置参数
            
        Returns:
            np.ndarray: 处理后的图像数组 (RGB格式)
        """
        pass
    
    @abstractmethod
    def cleanup_resources(self):
        """清理模型资源 - 子类必须实现"""
        pass
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_loaded': self.model_loaded,
            'device': str(self.device) if self.device else None,
            'config': self.config
        }
    
    def get_available_models(self) -> list:
        """获取可用的模型列表 - 默认实现"""
        return [getattr(self, 'model_name', 'unknown')]
    
    def get_current_model(self) -> str:
        """获取当前模型名称 - 默认实现"""
        return getattr(self, 'model_name', 'unknown')
    
    def predict_with_model(self, image: Image.Image, mask: Image.Image, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """使用模型进行预测 - 默认调用predict方法"""
        return self.predict(image, mask, config)
    
    def process_image(self, image: Image.Image, mask: Image.Image, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """处理图像 - 默认调用predict方法"""
        return self.predict(image, mask, config)
    
    def load_model(self):
        """加载模型 - 默认调用_load_model方法"""
        return self._load_model()
    
    def validate_inputs(self, image: Image.Image, mask: Image.Image) -> bool:
        """验证输入参数"""
        try:
            # 检查图像
            if not isinstance(image, Image.Image):
                logger.error("Input image must be PIL.Image")
                return False
            
            if image.size[0] <= 0 or image.size[1] <= 0:
                logger.error("Invalid image size")
                return False
            
            # 检查mask
            if not isinstance(mask, Image.Image):
                logger.error("Input mask must be PIL.Image")
                return False
                
            if mask.size != image.size:
                logger.error(f"Size mismatch: image {image.size}, mask {mask.size}")
                return False
            
            # 检查mask格式
            if mask.mode != 'L':
                logger.warning(f"Mask mode {mask.mode} will be converted to 'L'")
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def preprocess_inputs(self, image: Image.Image, mask: Image.Image) -> tuple:
        """预处理输入数据"""
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 确保mask是灰度格式
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # 确保尺寸匹配
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)
        
        return image, mask

class IOPaintBaseProcessor(BaseInpainter):
    """
    IOPaint模型的基础处理器
    提供通用的IOPaint集成功能
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        super().__init__(config)
        self.model_name = model_name
        self.model_manager = None
        self._load_iopaint_classes()
    
    def _load_iopaint_classes(self):
        """加载IOPaint相关类"""
        try:
            from iopaint.model_manager import ModelManager
            from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
            
            self.ModelManager = ModelManager
            self.HDStrategy = HDStrategy
            self.LDMSampler = LDMSampler
            self.Config = Config
            
        except ImportError as e:
            logger.error(f"Failed to import IOPaint classes: {e}")
            raise
    
    def _load_model(self):
        """加载IOPaint模型"""
        try:
            import torch
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 加载指定模型
            self.model_manager = self.ModelManager(name=self.model_name, device=str(self.device))
            
            self.model_loaded = True
            logger.info(f"✅ {self.model_name.upper()} model loaded successfully")
            logger.info(f"   Device: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.model_name} model: {e}")
            self.model_loaded = False
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
        
        # 映射采样器（仅支持IOPaint实际可用的采样器）
        sampler_map = {
            'ddim': self.LDMSampler.ddim,
            'plms': self.LDMSampler.plms
        }
        
        return self.Config(
            ldm_steps=merged_config['ldm_steps'],
            ldm_sampler=sampler_map.get(merged_config['ldm_sampler'], self.LDMSampler.ddim),
            hd_strategy=strategy_map.get(merged_config['hd_strategy'], self.HDStrategy.CROP),
            hd_strategy_crop_margin=merged_config['hd_strategy_crop_margin'],
            hd_strategy_crop_trigger_size=merged_config['hd_strategy_crop_trigger_size'],
            hd_strategy_resize_limit=merged_config['hd_strategy_resize_limit']
        )
    
    def predict(self, image: Image.Image, mask: Image.Image, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """统一的IOPaint推理接口"""
        if not self.model_loaded:
            raise RuntimeError(f"{self.model_name} model not loaded")
        
        # 验证输入
        if not self.validate_inputs(image, mask):
            raise ValueError("Invalid inputs")
        
        # 预处理
        image, mask = self.preprocess_inputs(image, mask)
        
        try:
            from ..utils.image_utils import ImageUtils
            
            # 为IOPaint准备数组（标准RGB处理）
            image_array, mask_array = ImageUtils.prepare_arrays_for_iopaint(image, mask)
            
            logger.info(f"🎨 {self.model_name.upper()} processing: {image_array.shape}")
            
            # 构建配置
            if config is None:
                config = {}
            iopaint_config = self._build_iopaint_config(config)
            
            # 执行推理
            result = self.model_manager(image_array, mask_array, iopaint_config)
            
            logger.info(f"✅ {self.model_name.upper()} processing completed")
            return result
            
        except Exception as e:
            logger.error(f"{self.model_name} prediction failed: {e}")
            raise
    
    def cleanup_resources(self):
        """清理IOPaint资源"""
        try:
            if self.model_manager is not None:
                del self.model_manager
            self.model_manager = None
            self.model_loaded = False
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"✅ {self.model_name.upper()} processor resources cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during {self.model_name} processor cleanup: {e}")

class ModelRegistry:
    """模型注册表 - 管理所有可用的模型类"""
    
    _models = {}
    
    @classmethod
    def register(cls, name: str, model_class: type):
        """注册模型类"""
        cls._models[name] = model_class
        logger.info(f"✅ Model {name} registered")
    
    @classmethod
    def get_model_class(cls, name: str) -> type:
        """获取模型类"""
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered")
        return cls._models[name]
    
    @classmethod
    def get_available_models(cls) -> list:
        """获取所有可用模型"""
        return list(cls._models.keys())
    
    @classmethod
    def create_model(cls, name: str, config: Dict[str, Any]) -> BaseInpainter:
        """创建模型实例"""
        model_class = cls.get_model_class(name)
        return model_class(config)