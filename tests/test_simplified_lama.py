#!/usr/bin/env python3
"""
测试简化LaMA处理器
验证简化后的LaMA处理器功能完整性和兼容性
"""

import sys
import os
import numpy as np
import logging
from PIL import Image
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simplified_lama_basic_functionality():
    """测试简化LaMA处理器基本功能"""
    print("🧪 测试简化LaMA处理器基本功能...")
    
    try:
        # 直接导入简化LaMA处理器，避免__init__.py中的循环导入问题
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from core.models.lama_processor_simplified import SimplifiedLamaProcessor
        
        # 测试配置
        config = {
            'ldm_steps': 50,
            'hd_strategy': 'CROP',
            'device': 'cuda'
        }
        
        # 创建处理器实例
        processor = SimplifiedLamaProcessor(config)
        
        # 验证基本属性
        assert processor.model_name == "lama"
        assert hasattr(processor, 'model_manager')
        assert hasattr(processor, 'predict')
        assert hasattr(processor, 'cleanup_resources')
        
        print("✅ 基本功能测试通过")
        return processor
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        raise

def test_model_interface_consistency():
    """测试模型接口一致性"""
    print("🧪 测试模型接口一致性...")
    
    try:
        from core.models.lama_processor_simplified import SimplifiedLamaProcessor
        from core.models.mat_processor import MatProcessor
        from core.models.base_inpainter import BaseInpainter
        
        config = {'device': 'cuda'}
        
        # 创建LaMA处理器
        lama_processor = SimplifiedLamaProcessor(config)
        
        # 验证继承关系
        assert isinstance(lama_processor, BaseInpainter)
        
        # 验证方法存在性
        required_methods = ['predict', 'cleanup_resources', 'is_loaded', 'validate_inputs']
        for method in required_methods:
            assert hasattr(lama_processor, method), f"缺少方法: {method}"
        
        print("✅ 接口一致性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 接口一致性测试失败: {e}")
        raise

def test_model_registry_integration():
    """测试模型注册表集成"""
    print("🧪 测试模型注册表集成...")
    
    try:
        from core.models.base_inpainter import ModelRegistry
        from core.models.lama_processor_simplified import SimplifiedLamaProcessor
        
        # 验证LaMA模型已注册
        available_models = ModelRegistry.get_available_models()
        assert "lama" in available_models, f"LaMA模型未注册. 可用模型: {available_models}"
        
        # 测试通过注册表创建模型
        config = {'device': 'cuda'}
        lama_processor = ModelRegistry.create_model("lama", config)
        
        assert isinstance(lama_processor, SimplifiedLamaProcessor)
        assert lama_processor.model_name == "lama"
        
        print("✅ 模型注册表集成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型注册表集成测试失败: {e}")
        raise

def test_image_processing():
    """测试图像处理功能"""
    print("🧪 测试图像处理功能...")
    
    try:
        from core.models.lama_processor_simplified import SimplifiedLamaProcessor
        
        # 创建测试图像和mask
        test_image = Image.new('RGB', (512, 512), color='red')
        test_mask = Image.new('L', (512, 512), color=0)
        
        # 在mask中添加一个白色区域
        mask_array = np.array(test_mask)
        mask_array[200:300, 200:300] = 255
        test_mask = Image.fromarray(mask_array, mode='L')
        
        config = {
            'ldm_steps': 20,  # 使用较少步数加快测试
            'hd_strategy': 'ORIGINAL',
            'device': 'cuda'
        }
        
        processor = SimplifiedLamaProcessor(config)
        
        # 验证输入验证
        assert processor.validate_inputs(test_image, test_mask)
        
        # 测试预处理
        processed_image, processed_mask = processor.preprocess_inputs(test_image, test_mask)
        assert processed_image.mode == 'RGB'
        assert processed_mask.mode == 'L'
        assert processed_image.size == processed_mask.size
        
        print("✅ 图像处理功能测试通过")
        
        # 清理资源
        processor.cleanup_resources()
        return True
        
    except Exception as e:
        print(f"❌ 图像处理功能测试失败: {e}")
        raise

def test_hd_strategy_support():
    """测试高分辨率策略支持"""
    print("🧪 测试高分辨率策略支持...")
    
    try:
        from core.models.lama_processor_simplified import SimplifiedLamaProcessor
        
        # 测试不同HD策略（根据IOPaint实际支持的策略）
        strategies = ['CROP', 'ORIGINAL', 'RESIZE']
        
        for strategy in strategies:
            config = {
                'hd_strategy': strategy,
                'device': 'cuda'
            }
            
            processor = SimplifiedLamaProcessor(config)
            
            # 验证配置构建
            test_config = processor._build_iopaint_config(config)
            assert hasattr(test_config, 'hd_strategy')
            
            processor.cleanup_resources()
        
        print("✅ 高分辨率策略支持测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 高分辨率策略支持测试失败: {e}")
        raise

def test_error_handling():
    """测试错误处理"""
    print("🧪 测试错误处理...")
    
    try:
        from core.models.lama_processor_simplified import SimplifiedLamaProcessor
        
        config = {'device': 'cuda'}
        processor = SimplifiedLamaProcessor(config)
        
        # 测试无效输入
        invalid_cases = [
            (None, Image.new('L', (100, 100))),  # 无效图像
            (Image.new('RGB', (100, 100)), None),  # 无效mask
            (Image.new('RGB', (100, 100)), Image.new('L', (200, 200))),  # 尺寸不匹配
        ]
        
        for invalid_image, invalid_mask in invalid_cases:
            try:
                if invalid_image is not None and invalid_mask is not None:
                    result = processor.validate_inputs(invalid_image, invalid_mask)
                    assert not result, "应该验证失败但返回了True"
            except (TypeError, AttributeError):
                pass  # 预期的错误
        
        print("✅ 错误处理测试通过")
        
        # 清理资源
        processor.cleanup_resources()
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        raise

def test_resource_cleanup():
    """测试资源清理"""
    print("🧪 测试资源清理...")
    
    try:
        from core.models.lama_processor_simplified import SimplifiedLamaProcessor
        
        config = {'device': 'cuda'}
        processor = SimplifiedLamaProcessor(config)
        
        # 验证模型已加载
        assert processor.is_loaded()
        
        # 清理资源
        processor.cleanup_resources()
        
        # 验证清理效果
        assert not processor.is_loaded()
        assert processor.model_manager is None
        
        print("✅ 资源清理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 资源清理测试失败: {e}")
        raise

def test_performance_comparison():
    """测试性能对比（简化vs复杂版本）"""
    print("🧪 测试性能对比...")
    
    try:
        import time
        from core.models.lama_processor_simplified import SimplifiedLamaProcessor
        
        # 测试简化版本的初始化时间
        start_time = time.time()
        config = {'device': 'cuda'}
        processor = SimplifiedLamaProcessor(config)
        init_time = time.time() - start_time
        
        print(f"📊 简化LaMA处理器初始化时间: {init_time:.3f}秒")
        
        # 验证代码行数减少
        processor_file = project_root / "core" / "models" / "lama_processor_simplified.py"
        with open(processor_file, 'r', encoding='utf-8') as f:
            lines = len(f.readlines())
        
        print(f"📊 简化LaMA处理器代码行数: {lines}行")
        assert lines < 30, f"代码行数应该小于30行，实际: {lines}行"
        
        processor.cleanup_resources()
        
        print("✅ 性能对比测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 性能对比测试失败: {e}")
        raise

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始简化LaMA处理器全面测试...")
    print("=" * 60)
    
    tests = [
        test_simplified_lama_basic_functionality,
        test_model_interface_consistency,
        test_model_registry_integration,
        test_image_processing,
        test_hd_strategy_support,
        test_error_handling,
        test_resource_cleanup,
        test_performance_comparison
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"💥 测试失败: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试通过！简化LaMA处理器工作正常")
        return True
    else:
        print("⚠️ 有测试失败，需要检查和修复")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)