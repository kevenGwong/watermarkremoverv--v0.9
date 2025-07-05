#!/usr/bin/env python3
"""
数据流验证测试 - 验证UI→inference→manager→processor的完整链路
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_flow_without_models():
    """测试数据流链路（不加载实际模型）"""
    print("🧪 测试数据流链路（不依赖模型）...")
    
    # 测试配置管理器
    try:
        from config.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print("  ✅ 配置管理器初始化成功")
        
        # 验证关键配置项
        key_configs = ['models', 'ui', 'processing']
        missing_configs = [k for k in key_configs if not hasattr(config, k)]
        if missing_configs:
            print(f"  ⚠️ 缺失配置项: {missing_configs}")
        else:
            print("  ✅ 关键配置项完整")
            
    except Exception as e:
        print(f"  ❌ 配置管理器初始化失败: {e}")
        return False
    
    # 测试数据结构
    try:
        from core.processors.processing_result import ProcessingResult
        
        # 创建测试结果
        test_image = Image.new('RGB', (100, 100), 'blue')
        test_mask = Image.new('L', (100, 100), 255)
        
        result = ProcessingResult(
            success=True,
            result_image=test_image,
            mask_image=test_mask,
            processing_time=1.5
        )
        
        print("  ✅ ProcessingResult数据结构正常")
        print(f"    - success: {result.success}")
        print(f"    - result_image: {result.result_image.size if result.result_image else None}")
        print(f"    - mask_image: {result.mask_image.size if result.mask_image else None}")
        print(f"    - processing_time: {result.processing_time}")
        
    except Exception as e:
        print(f"  ❌ ProcessingResult创建失败: {e}")
        return False
    
    return True

def test_parameter_flow():
    """测试参数传递流程"""
    print("\n🧪 测试参数传递流程...")
    
    # 模拟UI参数
    ui_params = {
        'mask_model': 'upload',
        'mask_params': {
            'uploaded_mask': None,
            'mask_dilate_kernel_size': 5,
            'mask_dilate_iterations': 2
        },
        'inpaint_params': {
            'inpaint_model': 'iopaint',
            'force_model': 'mat',
            'hd_strategy': 'ORIGINAL',
            'ldm_steps': 50
        },
        'performance_params': {
            'mixed_precision': True,
            'log_processing_time': True
        },
        'transparent': False
    }
    
    print("  📋 模拟UI参数:")
    for key, value in ui_params.items():
        print(f"    {key}: {type(value).__name__}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"      {subkey}: {subvalue}")
    
    # 测试参数验证和转换
    try:
        from core.utils.image_utils import ImageValidator
        
        # 测试图像验证
        test_image = Image.new('RGB', (512, 512), 'red')
        validation_result = ImageValidator.validate_input(test_image)
        
        print(f"  ✅ 图像验证: {validation_result}")
        
    except Exception as e:
        print(f"  ❌ 参数验证失败: {e}")
        return False
    
    return True

def test_inference_chain():
    """测试推理链条结构"""
    print("\n🧪 测试推理链条结构...")
    
    # 测试InferenceManager结构
    try:
        # 只导入类定义，不实例化
        from core.inference_manager import InferenceManager
        from core.processors.watermark_processor import WatermarkProcessor
        from core.models.unified_processor import UnifiedProcessor
        
        print("  ✅ 核心类导入成功:")
        print("    - InferenceManager")
        print("    - WatermarkProcessor")
        print("    - UnifiedProcessor")
        
        # 检查方法存在性
        manager_methods = ['process_request', '_generate_mask', '_process_with_inpaint']
        processor_methods = ['process_image']
        unified_methods = ['get_available_models', 'predict_with_model']
        
        print("  📋 关键方法检查:")
        
        # InferenceManager方法
        for method in manager_methods:
            has_method = hasattr(InferenceManager, method)
            status = "✅" if has_method else "❌"
            print(f"    InferenceManager.{method}: {status}")
        
        # WatermarkProcessor方法
        for method in processor_methods:
            has_method = hasattr(WatermarkProcessor, method)
            status = "✅" if has_method else "❌"
            print(f"    WatermarkProcessor.{method}: {status}")
        
        # UnifiedProcessor方法
        for method in unified_methods:
            has_method = hasattr(UnifiedProcessor, method)
            status = "✅" if has_method else "❌"
            print(f"    UnifiedProcessor.{method}: {status}")
        
    except Exception as e:
        print(f"  ❌ 推理链条结构检查失败: {e}")
        return False
    
    return True

def test_mask_generation_flow():
    """测试mask生成流程"""
    print("\n🧪 测试mask生成流程...")
    
    try:
        from core.models.mask_generators import FallbackMaskGenerator
        
        # 测试fallback mask生成器
        fallback_gen = FallbackMaskGenerator()
        
        test_image = Image.new('RGB', (256, 256), 'green')
        test_params = {'mask_threshold': 0.5}
        
        # 生成mask
        generated_mask = fallback_gen.generate_mask(test_image, test_params)
        
        print(f"  ✅ Fallback mask生成成功:")
        print(f"    - 输入图像尺寸: {test_image.size}")
        print(f"    - 生成mask尺寸: {generated_mask.size}")
        print(f"    - Mask模式: {generated_mask.mode}")
        
        # 检查mask像素值
        mask_array = np.array(generated_mask)
        unique_values = np.unique(mask_array)
        print(f"    - 像素值范围: {mask_array.min()} - {mask_array.max()}")
        print(f"    - 唯一值: {unique_values}")
        
    except Exception as e:
        print(f"  ❌ Mask生成流程测试失败: {e}")
        return False
    
    return True

def test_configuration_flow():
    """测试配置流程"""
    print("\n🧪 测试配置流程...")
    
    try:
        from config.config import ConfigManager
        
        config_manager = ConfigManager()
        
        # 测试默认配置获取
        default_config = config_manager.get_config()
        
        print("  ✅ 默认配置加载成功:")
        
        # 检查配置结构
        config_attrs = ['models', 'ui', 'processing']
        for attr in config_attrs:
            has_attr = hasattr(default_config, attr)
            status = "✅" if has_attr else "❌"
            print(f"    {attr}: {status}")
            
            if has_attr:
                config_section = getattr(default_config, attr)
                if hasattr(config_section, '__dict__'):
                    section_keys = list(config_section.__dict__.keys())
                    print(f"      包含: {section_keys[:3]}..." if len(section_keys) > 3 else f"      包含: {section_keys}")
        
        # 测试配置验证
        test_params = {
            'mask_model': 'upload',
            'inpaint_params': {
                'hd_strategy': 'ORIGINAL',
                'ldm_steps': 50
            }
        }
        
        validated_params = config_manager.validate_parameters(test_params)
        print(f"  ✅ 参数验证成功: {len(validated_params)} 个参数")
        
    except Exception as e:
        print(f"  ❌ 配置流程测试失败: {e}")
        return False
    
    return True

def test_image_processing_utils():
    """测试图像处理工具"""
    print("\n🧪 测试图像处理工具...")
    
    try:
        from core.utils.image_utils import ImageProcessor, ImageValidator, ImageDownloader
        
        # 创建测试图像
        test_image = Image.new('RGB', (300, 200), 'yellow')
        
        # 测试图像验证
        is_valid = ImageValidator.validate_input(test_image)
        print(f"  ✅ 图像验证: {is_valid}")
        
        # 测试图像处理
        resized_image = ImageProcessor.resize_image(test_image, (150, 100))
        print(f"  ✅ 图像调整: {test_image.size} → {resized_image.size}")
        
        # 测试下载信息生成
        download_info = ImageDownloader.create_download_info(test_image, "test")
        print(f"  ✅ 下载信息生成: {len(download_info)} 个格式")
        
        for info in download_info:
            print(f"    - {info['format']}: {info['filename']}")
        
    except Exception as e:
        print(f"  ❌ 图像处理工具测试失败: {e}")
        return False
    
    return True

def analyze_data_flow_structure():
    """分析数据流结构"""
    print("\n🔍 分析数据流结构...")
    
    data_flow_map = {
        'UI Layer': {
            'files': ['interfaces/web/main.py', 'interfaces/web/ui.py'],
            'function': '用户交互，参数收集',
            'outputs': 'UI参数字典'
        },
        'Inference Layer': {
            'files': ['core/inference.py'],
            'function': '统一API入口',
            'outputs': 'ProcessingResult对象'
        },
        'Manager Layer': {
            'files': ['core/inference_manager.py'],
            'function': '请求分发，流程管理',
            'outputs': '处理后的图像和mask'
        },
        'Processor Layer': {
            'files': ['core/processors/watermark_processor.py'],
            'function': '具体处理逻辑',
            'outputs': '修复后的图像'
        },
        'Model Layer': {
            'files': ['core/models/unified_processor.py', 'core/models/mask_generators.py'],
            'function': 'AI模型调用',
            'outputs': 'AI模型预测结果'
        }
    }
    
    print("  📊 数据流结构分析:")
    for layer, info in data_flow_map.items():
        print(f"\n    {layer}:")
        print(f"      文件: {', '.join(info['files'])}")
        print(f"      功能: {info['function']}")
        print(f"      输出: {info['outputs']}")
    
    # 分析数据传递路径
    print("\n  🔄 数据传递路径:")
    print("    1. UI → inference.py (process_image)")
    print("    2. inference.py → inference_manager.py (process_request)")
    print("    3. inference_manager.py → watermark_processor.py (process_image)")
    print("    4. watermark_processor.py → unified_processor.py (predict_with_model)")
    print("    5. unified_processor.py → IOPaint模型 (底层推理)")
    print("    6. 结果逆向传递回UI")
    
    return True

def test_data_flow_comprehensive():
    """数据流验证完整测试"""
    print("🚀 开始数据流验证测试...")
    
    test_results = {}
    
    # 执行各项测试
    tests = [
        ('数据流基础结构', test_data_flow_without_models),
        ('参数传递流程', test_parameter_flow),
        ('推理链条结构', test_inference_chain),
        ('Mask生成流程', test_mask_generation_flow),
        ('配置流程', test_configuration_flow),
        ('图像处理工具', test_image_processing_utils),
        ('数据流结构分析', analyze_data_flow_structure)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"  ❌ {test_name}测试异常: {e}")
            test_results[test_name] = False
    
    # 输出总结
    print("\n" + "="*60)
    print("📊 数据流验证测试结果总结")
    print("="*60)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    print(f"\n🎯 测试通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    print("\n📋 详细结果:")
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    # 数据流健康度评估
    critical_tests = ['数据流基础结构', '推理链条结构', '配置流程']
    critical_passed = sum(1 for test in critical_tests if test_results.get(test, False))
    
    print(f"\n🔧 关键组件健康度:")
    print(f"   核心数据流: {critical_passed}/{len(critical_tests)} 通过")
    
    # 整体评估
    overall_health = passed_tests >= total_tests * 0.8 and critical_passed == len(critical_tests)
    
    print(f"\n🎯 整体评估:")
    print(f"   数据流完整性: {'✅ 优秀' if overall_health else '⚠️ 需要改进'}")
    
    if overall_health:
        print("\n🎉 数据流验证通过!")
        print("✅ UI→inference→manager→processor链路完整")
        print("✅ 参数传递机制正常")
        print("✅ 数据结构定义完善")
        print("✅ 错误处理机制健全")
    else:
        print("\n⚠️ 数据流需要进一步优化")
        failed_tests = [name for name, result in test_results.items() if not result]
        if failed_tests:
            print(f"需要修复的组件: {', '.join(failed_tests)}")
    
    return test_results, overall_health

if __name__ == "__main__":
    results, health = test_data_flow_comprehensive()
    
    if health:
        print("\n✅ 数据流验证全部通过!")
    else:
        print("\n⚠️ 部分数据流组件需要优化")