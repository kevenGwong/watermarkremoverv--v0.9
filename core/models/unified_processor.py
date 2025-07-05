"""
真正的统一处理器
根据用户选择动态切换模型，支持ZITS、MAT、FCF、LaMA
"""

import logging
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

from .base_inpainter import BaseInpainter, ModelRegistry

logger = logging.getLogger(__name__)

class UnifiedProcessor:
    """
    统一处理器 - 根据用户选择动态切换模型
    替代之前错误的 UnifiedProcessor = SimplifiedLamaProcessor 实现
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_processor = None
        self.current_model_name = None
        self.loaded_processors = {}  # 缓存已加载的处理器
        
        # 注册所有可用模型
        self._register_models()
        
        logger.info("✅ UnifiedProcessor initialized with dynamic model switching")
    
    def _register_models(self):
        """注册所有可用的模型类"""
        try:
            from .zits_processor import ZitsProcessor
            from .mat_processor import MatProcessor
            from .fcf_processor import FcfProcessor
            from .lama_processor_simplified import SimplifiedLamaProcessor
            
            # 注册所有模型
            ModelRegistry.register('zits', ZitsProcessor)
            ModelRegistry.register('mat', MatProcessor)
            ModelRegistry.register('fcf', FcfProcessor)
            ModelRegistry.register('lama', SimplifiedLamaProcessor)
            
            logger.info("✅ All models registered successfully")
            
        except ImportError as e:
            logger.error(f"❌ Failed to register models: {e}")
            raise
    
    def switch_model(self, model_name: str) -> bool:
        """切换到指定模型"""
        try:
            if model_name == self.current_model_name and self.current_processor is not None:
                logger.info(f"✅ Model {model_name} already loaded and active")
                return True
            
            # 检查模型是否可用
            available_models = self.get_available_models()
            if model_name not in available_models:
                logger.error(f"❌ Model {model_name} not available. Available: {available_models}")
                return False
            
            # 清理当前处理器（但保留在缓存中以备复用）
            if self.current_processor is not None:
                logger.info(f"🔄 Switching from {self.current_model_name} to {model_name}")
            
            # 检查是否已经加载过该模型
            if model_name in self.loaded_processors:
                logger.info(f"♻️ Reusing cached {model_name} processor")
                self.current_processor = self.loaded_processors[model_name]
            else:
                # 创建新的处理器实例
                logger.info(f"🚀 Loading new {model_name} processor...")
                processor = ModelRegistry.create_model(model_name, self.config)
                processor._load_model()
                
                # 缓存处理器
                self.loaded_processors[model_name] = processor
                self.current_processor = processor
                
                logger.info(f"✅ {model_name.upper()} processor loaded and cached")
            
            self.current_model_name = model_name
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to switch to model {model_name}: {e}")
            return False
    
    def predict_with_model(self, image: Image.Image, mask: Image.Image, config: Dict[str, Any]) -> np.ndarray:
        """使用选定的模型进行预测"""
        # 从配置中获取模型名称
        model_name = config.get('model_name')
        if not model_name:
            raise ValueError("No model specified in config")
        
        # 切换到指定模型
        if not self.switch_model(model_name):
            raise RuntimeError(f"Failed to switch to model: {model_name}")
        
        # 确保有当前处理器
        if self.current_processor is None:
            raise RuntimeError("No processor available after model switch")
        
        logger.info(f"🎨 Processing with {model_name.upper()} model")
        
        # 执行预测
        try:
            result = self.current_processor.predict(image, mask, config)
            logger.info(f"✅ {model_name.upper()} processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ {model_name.upper()} processing failed: {e}")
            raise
    
    def get_available_models(self) -> list:
        """获取所有可用模型列表"""
        return ModelRegistry.get_available_models()
    
    def get_current_model(self) -> str:
        """获取当前活动的模型名称"""
        return self.current_model_name or "none"
    
    def is_model_loaded(self, model_name: str) -> bool:
        """检查指定模型是否已加载"""
        return model_name in self.loaded_processors
    
    def get_loaded_models(self) -> list:
        """获取已加载的模型列表"""
        return list(self.loaded_processors.keys())
    
    def cleanup_resources(self):
        """清理所有处理器资源"""
        try:
            logger.info("🧹 Cleaning up UnifiedProcessor resources...")
            
            # 清理所有已加载的处理器
            for model_name, processor in self.loaded_processors.items():
                try:
                    processor.cleanup_resources()
                    logger.info(f"✅ {model_name.upper()} processor cleaned up")
                except Exception as e:
                    logger.warning(f"⚠️ Error cleaning up {model_name}: {e}")
            
            # 清空缓存
            self.loaded_processors.clear()
            self.current_processor = None
            self.current_model_name = None
            
            logger.info("✅ UnifiedProcessor cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during UnifiedProcessor cleanup: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "current_model": self.current_model_name,
            "available_models": self.get_available_models(),
            "loaded_models": self.get_loaded_models(),
            "total_cached_processors": len(self.loaded_processors)
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出 - 自动清理资源"""
        self.cleanup_resources()