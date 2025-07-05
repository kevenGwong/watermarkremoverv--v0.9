"""
统一处理器模块
管理ZITS、MAT、FCF、LaMA四个独立的处理器
提供统一的接口和智能模型选择
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image

from .zits_processor import ZitsProcessor
from .mat_processor import MatProcessor
from .fcf_processor import FcfProcessor
from .lama_processor import LamaProcessor

logger = logging.getLogger(__name__)

class UnifiedProcessor:
    """统一处理器，管理多个inpainting模型"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processors = {}
        self.current_processor = None
        self.available_models = ["zits", "mat", "fcf", "lama"]
        self._load_processors()
    
    def _load_processors(self):
        """加载所有处理器"""
        try:
            # 加载ZITS处理器
            try:
                self.processors["zits"] = ZitsProcessor(self.config)
                logger.info("✅ ZITS processor loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load ZITS processor: {e}")
            
            # 加载MAT处理器
            try:
                self.processors["mat"] = MatProcessor(self.config)
                logger.info("✅ MAT processor loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load MAT processor: {e}")
            
            # 加载FCF处理器
            try:
                self.processors["fcf"] = FcfProcessor(self.config)
                logger.info("✅ FCF processor loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load FCF processor: {e}")
            
            # 加载LaMA处理器
            try:
                self.processors["lama"] = LamaProcessor(self.config)
                logger.info("✅ LaMA processor loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load LaMA processor: {e}")
            
            # 设置默认处理器
            if self.processors:
                self.current_processor = list(self.processors.keys())[0]
                logger.info(f"✅ Unified processor initialized with {len(self.processors)} models")
                logger.info(f"   Available models: {list(self.processors.keys())}")
                logger.info(f"   Default model: {self.current_processor}")
            else:
                logger.error("❌ No processors loaded successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize unified processor: {e}")
    
    def switch_model(self, model_name: str) -> bool:
        """切换模型"""
        if model_name not in self.available_models:
            logger.error(f"❌ Unsupported model: {model_name}")
            return False
        
        if model_name not in self.processors:
            logger.error(f"❌ Model {model_name} not loaded")
            return False
        
        self.current_processor = model_name
        logger.info(f"🔄 Switched to {model_name} model")
        return True
    
    def predict(self, 
                image: Image.Image, 
                mask: Image.Image,
                config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """使用当前模型进行inpainting"""
        
        if not self.processors:
            raise RuntimeError("No processors available")
        
        if self.current_processor is None:
            raise RuntimeError("No current processor selected")
        
        processor = self.processors[self.current_processor]
        if not processor.model_loaded:
            raise RuntimeError(f"{self.current_processor} processor not loaded")
        
        logger.info(f"🎨 Using {self.current_processor} for inpainting")
        return processor.predict(image, mask, config)
    
    def predict_with_model(self, 
                          model_name: str,
                          image: Image.Image, 
                          mask: Image.Image,
                          config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """使用指定模型进行inpainting"""
        
        if model_name not in self.processors:
            raise RuntimeError(f"Model {model_name} not available")
        
        processor = self.processors[model_name]
        if not processor.model_loaded:
            raise RuntimeError(f"{model_name} processor not loaded")
        
        logger.info(f"🎨 Using {model_name} for inpainting")
        return processor.predict(image, mask, config)
    
    def choose_optimal_model(self, image: Image.Image, mask: Image.Image) -> str:
        """智能选择最优模型"""
        
        if not self.processors:
            return None
        
        # 计算mask覆盖率
        mask_array = np.array(mask.convert("L"))
        mask_coverage = np.sum(mask_array > 128) / mask_array.size * 100
        
        # 获取图像复杂度（边缘密度）
        image_complexity = self._calculate_image_complexity(image)
        
        # 智能选择策略
        if mask_coverage > 30:
            if "mat" in self.processors:
                return "mat"      # 大水印用MAT（最佳质量）
        elif image_complexity > 0.7:
            if "zits" in self.processors:
                return "zits"     # 复杂结构用ZITS（最佳结构保持）
        elif mask_coverage < 5:
            if "lama" in self.processors:
                return "lama"     # 小水印用LaMA（最快速度）
        else:
            if "fcf" in self.processors:
                return "fcf"      # 中等情况用FCF（快速修复）
        
        # 降级选择
        for model in ["mat", "fcf", "zits", "lama"]:
            if model in self.processors:
                return model
        
        return None
    
    def _calculate_image_complexity(self, image: Image.Image) -> float:
        """计算图像复杂度"""
        try:
            import cv2
            
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            complexity = np.sum(edges > 0) / edges.size
            return complexity
        except Exception as e:
            logger.warning(f"Failed to calculate image complexity: {e}")
            return 0.5  # 默认中等复杂度
    
    def get_available_models(self) -> list:
        """获取可用的模型列表"""
        return list(self.processors.keys())
    
    def get_current_model(self) -> str:
        """获取当前模型"""
        return self.current_processor
    
    def is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        return model_name in self.processors and self.processors[model_name].model_loaded
    
    def cleanup_resources(self):
        """清理所有资源"""
        for model_name, processor in self.processors.items():
            try:
                processor.cleanup_resources()
                logger.info(f"✅ {model_name} processor resources cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up {model_name} processor: {e}")
        
        self.processors.clear()
        self.current_processor = None
        logger.info("✅ All unified processor resources cleaned up") 