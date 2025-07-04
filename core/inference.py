"""
推理逻辑模块 - 重构版本
基于模块化架构的简洁入口点
"""

import logging
from typing import Dict, Any, Optional
from PIL import Image

from .inference_manager import InferenceManager as RealInferenceManager
from .processors.processing_result import ProcessingResult

logger = logging.getLogger(__name__)

# 全局推理管理器实例
_inference_manager = None

def get_inference_manager(config_manager=None, config_path: Optional[str] = None) -> RealInferenceManager:
    """获取推理管理器实例（单例模式）"""
    global _inference_manager
    
    if _inference_manager is None:
        _inference_manager = RealInferenceManager(config_manager, config_path)
        if not _inference_manager.load_processor():
            logger.error("Failed to initialize inference manager")
            return None
    
    return _inference_manager

def process_image(image: Image.Image,
                 mask_model: str = "custom",
                 mask_params: Optional[Dict[str, Any]] = None,
                 inpaint_params: Optional[Dict[str, Any]] = None,
                 performance_params: Optional[Dict[str, Any]] = None,
                 transparent: bool = False,
                 config_manager=None,
                 config_path: Optional[str] = None) -> ProcessingResult:
    """
    处理图像的主接口
    
    Args:
        image: 输入图像
        mask_model: mask生成模型 ("custom", "florence2", "upload")
        mask_params: mask生成参数
        inpaint_params: inpainting参数
        performance_params: 性能参数
        transparent: 是否透明处理
        config_manager: 配置管理器
        config_path: 配置文件路径
        
    Returns:
        ProcessingResult: 处理结果
    """
    # 获取推理管理器
    manager = get_inference_manager(config_manager, config_path)
    if manager is None:
        return ProcessingResult(
            success=False,
            error_message="Failed to initialize inference manager"
        )
    
    # 设置默认参数
    if mask_params is None:
        mask_params = {}
    if inpaint_params is None:
        inpaint_params = {}
    if performance_params is None:
        performance_params = {}
    
    # 处理图像
    return manager.process_image(
        image=image,
        mask_model=mask_model,
        mask_params=mask_params,
        inpaint_params=inpaint_params,
        performance_params=performance_params,
        transparent=transparent
    )

def get_system_info(config_manager=None) -> Dict[str, Any]:
    """获取系统信息"""
    manager = get_inference_manager(config_manager)
    if manager is None:
        return {"error": "Inference manager not available"}
    
    return manager.get_system_info()

def cleanup_resources():
    """清理所有资源"""
    global _inference_manager
    
    if _inference_manager is not None:
        _inference_manager.cleanup_resources()
        _inference_manager = None
        logger.info("✅ All inference resources cleaned up")

# 向后兼容的接口
class WatermarkProcessor:
    """向后兼容的WatermarkProcessor类"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.manager = get_inference_manager(config_path=config_path)
        if self.manager is None:
            raise RuntimeError("Failed to initialize inference manager")
    
    def process_image(self, 
                     image: Image.Image,
                     transparent: bool = False,
                     max_bbox_percent: float = 10.0,
                     force_format: Optional[str] = None,
                     custom_inpaint_config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """向后兼容的处理接口"""
        mask_params = {'max_bbox_percent': max_bbox_percent}
        inpaint_params = custom_inpaint_config or {}
        
        return self.manager.process_image(
            image=image,
            mask_model="custom",
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params={},
            transparent=transparent
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return self.manager.get_system_info()
    
    def cleanup_resources(self):
        """清理资源"""
        self.manager.cleanup_resources()

class EnhancedWatermarkProcessor:
    """向后兼容的EnhancedWatermarkProcessor类"""
    
    def __init__(self, base_processor: WatermarkProcessor):
        self.manager = base_processor.manager
    
    def process_image_with_params(self, 
                                image: Image.Image,
                                mask_model: str,
                                mask_params: Dict[str, Any],
                                inpaint_params: Dict[str, Any],
                                performance_params: Dict[str, Any],
                                transparent: bool = False) -> ProcessingResult:
        """向后兼容的增强处理接口"""
        return self.manager.process_image(
            image=image,
            mask_model=mask_model,
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params=performance_params,
            transparent=transparent
        )

class InferenceManager:
    """向后兼容的InferenceManager类"""
    
    def __init__(self, config_manager, config_path: Optional[str] = None):
        self.manager = RealInferenceManager(config_manager, config_path)
        if not self.manager.load_processor():
            raise RuntimeError("Failed to initialize inference manager")
    
    def load_processor(self) -> bool:
        """加载处理器"""
        return self.manager is not None
    
    def process_image(self, 
                     image: Image.Image,
                     mask_model: str,
                     mask_params: Dict[str, Any],
                     inpaint_params: Dict[str, Any],
                     performance_params: Dict[str, Any],
                     transparent: bool = False) -> ProcessingResult:
        """处理图像"""
        return self.manager.process_image(
            image=image,
            mask_model=mask_model,
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params=performance_params,
            transparent=transparent
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return self.manager.get_system_info()
    
    def cleanup_resources(self):
        """清理资源"""
        self.manager.cleanup_resources()