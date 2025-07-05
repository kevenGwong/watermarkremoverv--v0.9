"""
简化水印处理器 - SIMP-LAMA架构实现
统一入口点，支持MAT/ZITS/FCF/LaMA模型选择
遵循单一入口、接口统一、自动资源管理原则
"""

import logging
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, Union
from PIL import Image
from pathlib import Path

from .processing_result import ProcessingResult
from ..models.unified_mask_generator import UnifiedMaskGenerator
from ..utils.memory_monitor import MemoryMonitor
from ..utils.image_utils import ImageUtils

logger = logging.getLogger(__name__)

class SimplifiedWatermarkProcessor:
    """
    简化水印处理器 - SIMP-LAMA架构
    
    核心特性:
    - Single Entry: 统一的process_image接口
    - Interface Unification: 所有模型统一接口
    - Auto Resource: 智能模型切换和内存管理
    - Pluggable Models: 支持MAT/ZITS/FCF/LaMA模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_model = None
        self.current_model_name = None
        self.model_cache = {}  # 模型缓存，实现懒加载
        self.memory_monitor = MemoryMonitor()
        
        # 初始化mask生成器
        self._init_mask_generator()
        
        # 支持的模型列表
        self.available_models = ["mat", "zits", "fcf", "lama"]
        
        logger.info("✅ SimplifiedWatermarkProcessor initialized")
        logger.info(f"   Available models: {self.available_models}")
        logger.info(f"   Memory management: enabled")
    
    def _init_mask_generator(self):
        """初始化统一mask生成器"""
        try:
            self.mask_generator = UnifiedMaskGenerator(self.config)
            logger.info("✅ Unified mask generator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize mask generator: {e}")
            raise RuntimeError(f"Mask generator initialization failed: {e}")
    
    def process_image(self, 
                     image: Image.Image,
                     model_name: str = "mat",
                     mask_method: str = "custom",
                     config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        统一入口点 - 处理图像
        
        Args:
            image: 输入图像
            model_name: 模型名称 (mat/zits/fcf/lama)
            mask_method: mask生成方法 (custom/upload)
            config: 处理配置
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 开始处理图像 - 模型: {model_name}")
            logger.info(f"📸 输入图像: {image.size}, 模式: {image.mode}")
            
            # 验证模型名称
            if model_name not in self.available_models:
                raise ValueError(f"不支持的模型: {model_name}. 可用模型: {self.available_models}")
            
            # 1. 生成mask
            mask_image = self._generate_mask(image, mask_method, config)
            
            # 2. 切换到指定模型
            self._switch_model(model_name)
            
            # 3. 执行inpainting
            result_image = self._inpaint_image(image, mask_image, config)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ 处理完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                success=True,
                result_image=result_image,
                mask_image=mask_image,
                processing_time=processing_time,
                model_used=model_name,
                memory_info=self.memory_monitor.get_memory_info()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 图像处理失败: {e}")
            
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                model_used=model_name,
                memory_info=self.memory_monitor.get_memory_info()
            )
    
    def _generate_mask(self, image: Image.Image, method: str, config: Dict[str, Any]) -> Image.Image:
        """生成mask - SIMP-LAMA统一接口"""
        try:
            logger.info(f"🎭 生成mask - 方法: {method}")
            
            # 准备参数
            mask_params = config.get('mask_params', {})
            if method == "upload":
                mask_params['uploaded_mask'] = config.get('uploaded_mask')
            
            # 使用统一mask生成器
            mask_image = self.mask_generator.generate_mask(image, method, mask_params)
            
            # 验证mask兼容性
            if not self.mask_generator.validate_mask_compatibility(mask_image, self.current_model_name or "unknown"):
                logger.warning("⚠️ Mask兼容性检查未通过，但继续处理")
            
            # 获取mask信息
            mask_info = self.mask_generator.get_mask_info(mask_image)
            logger.info(f"🔍 Mask信息: {mask_info['coverage_percent']:.2f}% 覆盖率, {mask_info['unique_values']} 唯一值")
            
            return mask_image
            
        except Exception as e:
            logger.error(f"❌ Mask生成失败: {e}")
            raise
    
    def _switch_model(self, model_name: str):
        """智能模型切换 - 自动内存管理"""
        if self.current_model_name == model_name and self.current_model is not None:
            logger.info(f"✅ 模型 {model_name} 已加载，无需切换")
            return
        
        try:
            # 清理当前模型
            if self.current_model is not None:
                logger.info(f"🔄 卸载当前模型: {self.current_model_name}")
                self._cleanup_current_model()
            
            # 检查内存状态
            memory_info = self.memory_monitor.get_memory_info()
            logger.info(f"📊 切换前内存状态: {memory_info}")
            
            # 加载新模型
            logger.info(f"⏳ 加载模型: {model_name}")
            self.current_model = self._load_model(model_name)
            self.current_model_name = model_name
            
            # 检查切换后内存状态
            memory_info = self.memory_monitor.get_memory_info()
            logger.info(f"📊 切换后内存状态: {memory_info}")
            logger.info(f"✅ 模型切换完成: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ 模型切换失败: {e}")
            self.current_model = None
            self.current_model_name = None
            raise
    
    def _load_model(self, model_name: str):
        """加载指定模型 - 使用模型注册表实现懒加载"""
        try:
            # 导入模型以确保它们被注册
            from ..models.mat_processor import MatProcessor
            from ..models.zits_processor import ZitsProcessor  
            from ..models.fcf_processor import FcfProcessor
            # LaMA可能有依赖问题，单独处理
            
            # 导入简化LaMA处理器以确保注册
            from ..models.lama_processor_simplified import SimplifiedLamaProcessor
            
            # 使用模型注册表统一创建模型
            from ..models.base_inpainter import ModelRegistry
            return ModelRegistry.create_model(model_name, self.config)
                
        except Exception as e:
            logger.error(f"❌ 模型 {model_name} 加载失败: {e}")
            raise
    
    def _cleanup_current_model(self):
        """清理当前模型资源"""
        try:
            if hasattr(self.current_model, 'cleanup_resources'):
                self.current_model.cleanup_resources()
            
            del self.current_model
            self.current_model = None
            
            # 强制清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            logger.info("✅ 模型资源清理完成")
            
        except Exception as e:
            logger.warning(f"⚠️ 模型清理时出现错误: {e}")
    
    def _inpaint_image(self, image: Image.Image, mask: Image.Image, config: Dict[str, Any]) -> Image.Image:
        """统一inpainting接口"""
        if self.current_model is None:
            raise RuntimeError("当前没有加载的模型")
        
        try:
            # 预处理图像和mask
            processed_image, processed_mask = ImageUtils.preprocess_for_model(
                image, mask, self.current_model_name
            )
            
            # 执行inpainting
            logger.info(f"🎨 执行{self.current_model_name.upper()}推理...")
            inpaint_config = config.get('inpaint_params', {})
            
            # 调用模型的统一接口
            result_array = self.current_model.predict(processed_image, processed_mask, inpaint_config)
            
            # 后处理结果
            result_image = ImageUtils.postprocess_result(
                result_array, self.current_model_name, image.size
            )
            
            return result_image
            
        except Exception as e:
            logger.error(f"❌ {self.current_model_name}推理失败: {e}")
            raise
    
    def get_model_status(self) -> Dict[str, Any]:
        """获取模型状态信息"""
        memory_info = self.memory_monitor.get_memory_info()
        
        return {
            "current_model": self.current_model_name,
            "available_models": self.available_models,
            "model_loaded": self.current_model is not None,
            "memory_info": memory_info,
            "mask_generator_status": "ready" if self.mask_generator else "error"
        }
    
    def cleanup(self):
        """清理所有资源"""
        try:
            logger.info("🧹 开始清理所有资源...")
            
            # 清理当前模型
            if self.current_model is not None:
                self._cleanup_current_model()
            
            # 清理mask生成器
            if hasattr(self.mask_generator, 'cleanup_resources'):
                self.mask_generator.cleanup_resources()
            
            # 清理模型缓存
            self.model_cache.clear()
            
            logger.info("✅ 所有资源清理完成")
            
        except Exception as e:
            logger.warning(f"⚠️ 资源清理时出现错误: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - 自动资源清理"""
        self.cleanup()

# 全局实例管理器
class ProcessorManager:
    """处理器管理器 - 单例模式"""
    
    _instance = None
    _processor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_processor(self, config: Dict[str, Any]) -> SimplifiedWatermarkProcessor:
        """获取处理器实例"""
        if self._processor is None:
            self._processor = SimplifiedWatermarkProcessor(config)
        return self._processor
    
    def cleanup(self):
        """清理处理器"""
        if self._processor is not None:
            self._processor.cleanup()
            self._processor = None

# 全局管理器实例
processor_manager = ProcessorManager()