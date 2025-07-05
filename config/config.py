"""
配置管理模块
负责应用配置、参数验证和默认值管理
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """应用配置数据类"""
    # 页面配置
    page_title: str = "AI Watermark Remover - Debug"
    page_icon: str = "🔬"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # 模型配置
    config_file: str = "web_config.yaml"
    
    # 默认参数
    default_mask_threshold: float = 0.5
    default_dilate_kernel_size: int = 3
    default_dilate_iterations: int = 1
    default_max_bbox_percent: float = 10.0
    default_ldm_steps: int = 50
    default_ldm_sampler: str = "ddim"
    default_hd_strategy: str = "CROP"
    default_crop_margin: int = 64
    default_crop_trigger_size: int = 800
    default_resize_limit: int = 1600

class ConfigManager:
    """配置管理器 - SIMP-LAMA统一配置"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/unified_config.yaml"
        self.app_config = AppConfig()
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self.config = self._get_default_config()
                logger.warning(f"Config file {self.config_file} not found, using defaults")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置 - SIMP-LAMA统一架构"""
        return {
            "app": {
                "title": "AI Watermark Remover - SIMP-LAMA Edition",
                "host": "0.0.0.0",
                "port": 8501
            },
            "models": {
                "available_models": ["mat", "zits", "fcf", "lama"],
                "default_model": "mat"
            },
            "mask_generator": {
                "model_type": "custom",
                "mask_threshold": 0.5,
                "mask_dilate_kernel_size": 3
            },
            "model_configs": {
                "mat": {"ldm_steps": 50, "hd_strategy": "CROP"},
                "zits": {"ldm_steps": 50, "hd_strategy": "CROP"},
                "fcf": {"ldm_steps": 40, "hd_strategy": "CROP"},
                "lama": {"ldm_steps": 50, "hd_strategy": "CROP"}
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config.get("models", {})
    
    def get_mask_config(self) -> Dict[str, Any]:
        """获取mask生成器配置"""
        return self.config.get("mask_generator", {})
    
    def get_model_specific_config(self, model_name: str) -> Dict[str, Any]:
        """获取特定模型的配置"""
        model_configs = self.config.get("model_configs", {})
        return model_configs.get(model_name, {})
    
    def get_available_models(self) -> list:
        """获取可用模型列表"""
        return self.config.get("models", {}).get("available_models", ["mat", "zits", "fcf", "lama"])
    
    def get_default_model(self) -> str:
        """获取默认模型"""
        return self.config.get("models", {}).get("default_model", "mat")
    
    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI配置"""
        return self.config.get("ui", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能配置"""
        return self.config.get("performance", {})
    
    def validate_mask_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证mask参数"""
        validated = params.copy()
        
        # 验证阈值范围
        if 'mask_threshold' in validated:
            validated['mask_threshold'] = max(0.0, min(1.0, validated['mask_threshold']))
        
        # 验证膨胀参数
        if 'mask_dilate_kernel_size' in validated:
            validated['mask_dilate_kernel_size'] = max(1, min(50, validated['mask_dilate_kernel_size']))
        
        if 'mask_dilate_iterations' in validated:
            validated['mask_dilate_iterations'] = max(1, min(20, validated['mask_dilate_iterations']))
        
        # 验证bbox百分比
        if 'max_bbox_percent' in validated:
            validated['max_bbox_percent'] = max(1.0, min(50.0, validated['max_bbox_percent']))
        
        return validated
    
    def validate_inpaint_params(self, params: Dict[str, Any], model_name: str = None) -> Dict[str, Any]:
        """验证inpainting参数 - SIMP-LAMA统一验证"""
        validated = params.copy()
        
        # 获取模型名称
        if model_name is None:
            model_name = validated.get('inpaint_model', self.get_default_model())
        
        # 验证模型名称
        available_models = self.get_available_models()
        if model_name not in available_models:
            model_name = self.get_default_model()
            validated['inpaint_model'] = model_name
        
        # 验证步数
        if 'ldm_steps' in validated:
            validated['ldm_steps'] = max(10, min(100, validated['ldm_steps']))
        
        # 验证HD策略 (SIMP-LAMA: 移除RESIZE，只支持CROP和ORIGINAL)
        if 'hd_strategy' in validated:
            if validated['hd_strategy'] not in ['CROP', 'ORIGINAL']:
                validated['hd_strategy'] = 'CROP'
        
        # 验证采样器
        if 'ldm_sampler' in validated:
            if validated['ldm_sampler'] not in ['ddim', 'plms']:
                validated['ldm_sampler'] = 'ddim'
        
        # 验证crop参数
        if 'hd_strategy_crop_margin' in validated:
            validated['hd_strategy_crop_margin'] = max(32, min(128, validated['hd_strategy_crop_margin']))
        
        if 'hd_strategy_crop_trigger_size' in validated:
            validated['hd_strategy_crop_trigger_size'] = max(512, min(2048, validated['hd_strategy_crop_trigger_size']))
        
        # 验证种子
        if 'seed' in validated:
            if validated['seed'] < -1 or validated['seed'] > 999999:
                validated['seed'] = -1
        
        # 应用模型特定配置
        model_config = self.get_model_specific_config(model_name)
        for key, value in model_config.items():
            if key not in validated:
                validated[key] = value
        
        return validated
    
    def get_default_mask_params(self, model_type: str) -> Dict[str, Any]:
        """获取默认mask参数 - SIMP-LAMA统一配置"""
        mask_config = self.get_mask_config()
        
        if model_type == "custom":
            return {
                'mask_threshold': mask_config.get('mask_threshold', 0.5),
                'mask_dilate_kernel_size': mask_config.get('mask_dilate_kernel_size', 3),
                'mask_dilate_iterations': mask_config.get('mask_dilate_iterations', 1),
                'detection_prompt': 'watermark',
                'max_bbox_percent': 10.0,
                'confidence_threshold': 0.3
            }
        else:  # upload (florence2 removed in SIMP-LAMA)
            return {
                'uploaded_mask': None,
                'mask_dilate_kernel_size': mask_config.get('upload_additional_dilate', 5),
                'mask_dilate_iterations': mask_config.get('upload_dilate_iterations', 2)
            }
    
    def get_default_inpaint_params(self, model_name: str = None) -> Dict[str, Any]:
        """获取默认inpainting参数 - SIMP-LAMA统一配置"""
        if model_name is None:
            model_name = self.get_default_model()
        
        # 获取模型特定配置
        model_config = self.get_model_specific_config(model_name)
        
        # 基础默认参数
        base_params = {
            'inpaint_model': model_name,
            'ldm_steps': 50,
            'ldm_sampler': 'ddim',
            'hd_strategy': 'CROP',
            'hd_strategy_crop_margin': 64,
            'hd_strategy_crop_trigger_size': 1024,
            'seed': -1
        }
        
        # 合并模型特定配置
        base_params.update(model_config)
        
        return base_params
    
    def get_default_performance_params(self) -> Dict[str, Any]:
        """获取默认性能参数 - SIMP-LAMA统一配置"""
        performance_config = self.get_performance_config()
        return {
            'mixed_precision': performance_config.get('enable_mixed_precision', True),
            'log_processing_time': performance_config.get('log_processing_time', True),
            'log_memory_usage': performance_config.get('log_memory_usage', True)
        } 