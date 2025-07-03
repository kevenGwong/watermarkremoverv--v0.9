"""
Configuration Utilities
配置管理相关的工具函数
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    保存配置文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"配置文件保存成功: {config_path}")
    except Exception as e:
        logger.error(f"配置文件保存失败: {e}")
        raise


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置字典
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    merged = base_config.copy()
    
    def deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
    
    deep_merge(merged, override_config)
    return merged


def get_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    default: Any = None) -> Any:
    """
    从嵌套配置中获取值
    
    Args:
        config: 配置字典
        key_path: 键路径，用点分隔，如 "models.lama_model"
        default: 默认值
        
    Returns:
        配置值
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    value: Any) -> None:
    """
    设置嵌套配置中的值
    
    Args:
        config: 配置字典
        key_path: 键路径，用点分隔
        value: 要设置的值
    """
    keys = key_path.split('.')
    current = config
    
    # 导航到最后一个键的父级
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # 设置值
    current[keys[-1]] = value


def validate_config(config: Dict[str, Any], 
                   required_keys: list = None) -> bool:
    """
    验证配置完整性
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
        
    Returns:
        是否有效
    """
    if required_keys is None:
        required_keys = [
            'models.florence_model',
            'models.lama_model',
            'mask_generator.model_type',
            'mask_generator.mask_model_path'
        ]
    
    for key_path in required_keys:
        if get_config_value(config, key_path) is None:
            logger.error(f"配置缺少必需项: {key_path}")
            return False
    
    return True


def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'app': {
            'title': "AI Watermark Remover",
            'host': "0.0.0.0",
            'port': 8501,
            'debug': False
        },
        'processing': {
            'default_max_bbox_percent': 10.0,
            'default_transparent': False,
            'default_overwrite': False,
            'default_force_format': "None",
            'supported_formats': ["jpg", "jpeg", "png", "webp"]
        },
        'mask_generator': {
            'model_type': "custom",
            'mask_model_path': "./models/epoch=071-valid_iou=0.7267.ckpt",
            'image_size': 768,
            'imagenet_mean': [0.485, 0.456, 0.406],
            'imagenet_std': [0.229, 0.224, 0.225],
            'mask_threshold': 0.5,
            'mask_dilate_kernel_size': 3
        },
        'models': {
            'florence_model': "microsoft/Florence-2-large",
            'lama_model': "lama"
        },
        'files': {
            'max_upload_size': 10,
            'temp_dir': "./temp",
            'output_dir': "./output",
            'allowed_extensions': [".jpg", ".jpeg", ".png", ".webp"]
        },
        'ui': {
            'show_advanced_options': True,
            'show_system_info': True,
            'enable_batch_processing': True,
            'default_download_format': "png"
        }
    }


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置（兼容性函数）
    
    Returns:
        默认配置字典
    """
    return create_default_config()


def get_environment_config(env: str = None) -> Dict[str, Any]:
    """
    获取环境特定配置
    
    Args:
        env: 环境名称（development, production等）
        
    Returns:
        环境配置
    """
    if env is None:
        env = os.getenv('ENVIRONMENT', 'development')
    
    config_dir = Path(__file__).parent.parent.parent / 'config' / 'environments'
    env_config_path = config_dir / f'{env}.yaml'
    
    if env_config_path.exists():
        return load_config(str(env_config_path))
    else:
        logger.warning(f"环境配置文件不存在: {env_config_path}")
        return {}


def resolve_paths(config: Dict[str, Any], base_path: str = None) -> Dict[str, Any]:
    """
    解析配置中的相对路径
    
    Args:
        config: 配置字典
        base_path: 基础路径
        
    Returns:
        解析后的配置
    """
    if base_path is None:
        base_path = os.getcwd()
    
    resolved_config = config.copy()
    
    def resolve_path(value):
        if isinstance(value, str) and value.startswith('./'):
            return os.path.join(base_path, value[2:])
        return value
    
    def deep_resolve(obj):
        if isinstance(obj, dict):
            return {k: deep_resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_resolve(v) for v in obj]
        else:
            return resolve_path(obj)
    
    return deep_resolve(resolved_config) 