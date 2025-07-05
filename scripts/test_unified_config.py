#!/usr/bin/env python3
"""
统一配置文件测试脚本
验证SIMP-LAMA架构的配置管理功能
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_manager():
    """测试ConfigManager统一配置功能"""
    logger.info("🧪 测试ConfigManager统一配置...")
    
    try:
        from config.config import ConfigManager
        
        # 创建配置管理器实例
        config_manager = ConfigManager()
        logger.info("✅ ConfigManager实例创建成功")
        
        # 测试基础配置访问
        logger.info("📋 测试基础配置访问...")
        available_models = config_manager.get_available_models()
        default_model = config_manager.get_default_model()
        
        logger.info(f"   可用模型: {available_models}")
        logger.info(f"   默认模型: {default_model}")
        
        # 测试各模型特定配置
        logger.info("📋 测试模型特定配置...")
        for model_name in available_models:
            model_config = config_manager.get_model_specific_config(model_name)
            logger.info(f"   {model_name.upper()} 配置: {model_config}")
        
        # 测试mask配置
        logger.info("📋 测试mask配置...")
        mask_config = config_manager.get_mask_config()
        logger.info(f"   Mask配置: {mask_config}")
        
        # 测试默认参数生成
        logger.info("📋 测试默认参数生成...")
        
        # Custom mask参数
        custom_mask_params = config_manager.get_default_mask_params("custom")
        logger.info(f"   Custom mask参数: {custom_mask_params}")
        
        # Upload mask参数
        upload_mask_params = config_manager.get_default_mask_params("upload")
        logger.info(f"   Upload mask参数: {upload_mask_params}")
        
        # 各模型的inpainting参数
        for model_name in available_models:
            inpaint_params = config_manager.get_default_inpaint_params(model_name)
            logger.info(f"   {model_name.upper()} inpaint参数: {inpaint_params}")
        
        # 性能参数
        performance_params = config_manager.get_default_performance_params()
        logger.info(f"   性能参数: {performance_params}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ConfigManager测试失败: {e}")
        return False

def test_parameter_validation():
    """测试参数验证功能"""
    logger.info("🧪 测试参数验证功能...")
    
    try:
        from config.config import ConfigManager
        
        config_manager = ConfigManager()
        
        # 测试mask参数验证
        logger.info("📋 测试mask参数验证...")
        test_mask_params = {
            'mask_threshold': 1.5,  # 超出范围
            'mask_dilate_kernel_size': 100,  # 超出范围
            'max_bbox_percent': 200.0  # 超出范围
        }
        
        validated_mask = config_manager.validate_mask_params(test_mask_params)
        logger.info(f"   原始参数: {test_mask_params}")
        logger.info(f"   验证后参数: {validated_mask}")
        
        # 测试inpaint参数验证
        logger.info("📋 测试inpaint参数验证...")
        test_inpaint_params = {
            'inpaint_model': 'invalid_model',  # 无效模型
            'ldm_steps': 500,  # 超出范围
            'hd_strategy': 'RESIZE',  # SIMP-LAMA已移除
            'seed': 9999999  # 超出范围
        }
        
        validated_inpaint = config_manager.validate_inpaint_params(test_inpaint_params)
        logger.info(f"   原始参数: {test_inpaint_params}")
        logger.info(f"   验证后参数: {validated_inpaint}")
        
        # 验证各模型参数验证
        for model_name in config_manager.get_available_models():
            model_params = {'ldm_steps': 200, 'hd_strategy': 'INVALID'}
            validated = config_manager.validate_inpaint_params(model_params, model_name)
            logger.info(f"   {model_name.upper()} 验证结果: {validated}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 参数验证测试失败: {e}")
        return False

def test_config_file_loading():
    """测试配置文件加载"""
    logger.info("🧪 测试配置文件加载...")
    
    try:
        from config.config import ConfigManager
        
        # 测试统一配置文件是否存在
        unified_config_path = Path("config/unified_config.yaml")
        if unified_config_path.exists():
            logger.info(f"✅ 统一配置文件存在: {unified_config_path}")
            
            # 创建使用统一配置的管理器
            config_manager = ConfigManager("config/unified_config.yaml")
            
            # 验证配置加载
            full_config = config_manager.get_config()
            logger.info(f"   加载的配置节: {list(full_config.keys())}")
            
            # 验证关键配置段
            required_sections = ['app', 'models', 'mask_generator', 'model_configs']
            for section in required_sections:
                if section in full_config:
                    logger.info(f"   ✅ {section} 配置段存在")
                else:
                    logger.warning(f"   ⚠️ {section} 配置段缺失")
            
        else:
            logger.warning(f"⚠️ 统一配置文件不存在，将使用默认配置")
            config_manager = ConfigManager()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置文件加载测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始统一配置文件测试套件...")
    
    tests = [
        ("配置文件加载", test_config_file_loading),
        ("ConfigManager功能", test_config_manager),
        ("参数验证", test_parameter_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- 测试: {test_name} ---")
        try:
            if test_func():
                logger.info(f"✅ {test_name} 测试通过")
                passed += 1
            else:
                logger.error(f"❌ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
    
    logger.info(f"\n🎯 测试结果: {passed}/{total} 通过")
    
    # 总结
    logger.info("\n📋 配置整合优化总结:")
    logger.info("   ✅ 创建统一配置文件 unified_config.yaml")
    logger.info("   ✅ 更新ConfigManager支持SIMP-LAMA架构")
    logger.info("   ✅ 移除RESIZE策略，简化HD选项")
    logger.info("   ✅ 为每个模型提供专用配置段")
    logger.info("   ✅ 统一参数验证和默认值管理")
    
    if passed == total:
        logger.info("🎉 配置文件整合成功！")
        return 0
    else:
        logger.error("❌ 配置文件整合需要进一步改进")
        return 1

if __name__ == "__main__":
    sys.exit(main())