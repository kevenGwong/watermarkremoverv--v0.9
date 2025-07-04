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
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "web_config.yaml"
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
        """获取默认配置"""
        return {
            "models": {
                "lama_model": "lama"
            },
            "mask_generator": {
                "model_type": "custom"
            },
            "interfaces": {
                "web": {"port": 8501, "host": "localhost"}
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
    
    def validate_inpaint_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证inpainting参数"""
        validated = params.copy()
        
        model_type = validated.get('inpaint_model', 'lama')
        
        if model_type == 'powerpaint':
            # PowerPaint parameter validation
            if 'num_inference_steps' in validated:
                validated['num_inference_steps'] = max(10, min(100, validated['num_inference_steps']))
            
            if 'guidance_scale' in validated:
                validated['guidance_scale'] = max(1.0, min(20.0, validated['guidance_scale']))
            
            if 'strength' in validated:
                validated['strength'] = max(0.1, min(1.0, validated['strength']))
            
            if 'crop_trigger_size' in validated:
                validated['crop_trigger_size'] = max(256, min(1024, validated['crop_trigger_size']))
            
            if 'crop_margin' in validated:
                validated['crop_margin'] = max(32, min(128, validated['crop_margin']))
            
            if 'edge_feather' in validated:
                validated['edge_feather'] = max(1, min(15, validated['edge_feather']))
            
            # Ensure boolean parameters
            validated['resize_to_512'] = bool(validated.get('resize_to_512', True))
            validated['blend_edges'] = bool(validated.get('blend_edges', True))
            
        else:
            # LaMA parameter validation
            if 'ldm_steps' in validated:
                validated['ldm_steps'] = max(10, min(200, validated['ldm_steps']))
            
            # 验证采样器
            if 'ldm_sampler' in validated:
                if validated['ldm_sampler'] not in ['ddim', 'plms']:
                    validated['ldm_sampler'] = 'ddim'
            
            # 验证HD策略
            if 'hd_strategy' in validated:
                if validated['hd_strategy'] not in ['CROP', 'RESIZE', 'ORIGINAL']:
                    validated['hd_strategy'] = 'CROP'
            
            # 验证crop参数
            if 'hd_strategy_crop_margin' in validated:
                validated['hd_strategy_crop_margin'] = max(32, min(256, validated['hd_strategy_crop_margin']))
            
            if 'hd_strategy_crop_trigger_size' in validated:
                validated['hd_strategy_crop_trigger_size'] = max(512, min(2048, validated['hd_strategy_crop_trigger_size']))
            
            # 验证resize参数
            if 'hd_strategy_resize_limit' in validated:
                validated['hd_strategy_resize_limit'] = max(512, min(2048, validated['hd_strategy_resize_limit']))
        
        return validated
    
    def get_default_mask_params(self, model_type: str) -> Dict[str, Any]:
        """获取默认mask参数"""
        if model_type == "custom":
            return {
                'mask_threshold': self.app_config.default_mask_threshold,
                'mask_dilate_kernel_size': self.app_config.default_dilate_kernel_size,
                'mask_dilate_iterations': self.app_config.default_dilate_iterations
            }
        elif model_type == "florence2":
            return {
                'detection_prompt': 'watermark',
                'max_bbox_percent': self.app_config.default_max_bbox_percent,
                'confidence_threshold': 0.3
            }
        else:  # upload
            return {
                'uploaded_mask': None,
                'mask_dilate_kernel_size': 0
            }
    
    def get_default_inpaint_params(self, model_type: str = "lama") -> Dict[str, Any]:
        """获取默认inpainting参数"""
        if model_type == "powerpaint":
            return {
                'inpaint_model': 'powerpaint',
                'prompt': 'high quality, detailed, clean, professional photo',
                'negative_prompt': 'watermark, logo, text, signature, blurry, low quality, artifacts, distorted, deformed',
                'num_inference_steps': 50,
                'guidance_scale': 7.5,
                'strength': 1.0,
                'crop_trigger_size': 512,
                'crop_margin': 64,
                'resize_to_512': True,
                'blend_edges': True,
                'edge_feather': 5,
                'seed': -1
            }
        else:
            return {
                'inpaint_model': 'lama',
                'ldm_steps': self.app_config.default_ldm_steps,
                'ldm_sampler': self.app_config.default_ldm_sampler,
                'hd_strategy': self.app_config.default_hd_strategy,
                'hd_strategy_crop_margin': self.app_config.default_crop_margin,
                'hd_strategy_crop_trigger_size': self.app_config.default_crop_trigger_size,
                'hd_strategy_resize_limit': self.app_config.default_resize_limit,
                'seed': -1
            }
    
    def get_default_performance_params(self) -> Dict[str, Any]:
        """获取默认性能参数"""
        return {
            'mixed_precision': True,
            'log_processing_time': True
        } 