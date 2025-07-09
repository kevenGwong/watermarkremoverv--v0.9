#!/usr/bin/env python3
"""
运行时问题检测工具
专门检测运行时错误，如属性缺失、方法不存在等
"""

import sys
import os
import inspect
import importlib
from typing import Dict, Any, List, Tuple
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

def check_class_interface(cls, expected_methods: List[str]) -> List[str]:
    """检查类是否实现了预期的接口方法"""
    issues = []
    
    # 获取类的所有方法
    class_methods = [name for name, _ in inspect.getmembers(cls, inspect.isfunction)]
    class_methods.extend([name for name, _ in inspect.getmembers(cls, inspect.ismethod)])
    
    # 检查是否缺少预期方法
    for method_name in expected_methods:
        if method_name not in class_methods:
            issues.append(f"类 {cls.__name__} 缺少方法: {method_name}")
    
    return issues

def check_inheritance_chain(cls) -> List[str]:
    """检查类的继承链和接口一致性"""
    issues = []
    
    # 获取所有父类
    mro = cls.__mro__
    
    # 检查每个父类的接口
    for parent in mro[1:]:  # 跳过自己
        if hasattr(parent, '__abstractmethods__'):
            abstract_methods = parent.__abstractmethods__
            for method in abstract_methods:
                if not hasattr(cls, method):
                    issues.append(f"类 {cls.__name__} 未实现抽象方法: {method}")
    
    return issues

def check_method_signatures(cls, method_name: str, expected_signature: str) -> List[str]:
    """检查方法签名是否匹配"""
    issues = []
    
    if hasattr(cls, method_name):
        method = getattr(cls, method_name)
        if inspect.isfunction(method) or inspect.ismethod(method):
            sig = inspect.signature(method)
            actual_signature = str(sig)
            
            # 简单的签名比较（可以更复杂）
            if expected_signature not in actual_signature:
                issues.append(f"方法 {method_name} 签名不匹配: 期望 {expected_signature}, 实际 {actual_signature}")
    else:
        issues.append(f"类 {cls.__name__} 缺少方法: {method_name}")
    
    return issues

def test_processor_interfaces():
    """测试处理器接口一致性"""
    print("🔍 测试处理器接口一致性...")
    
    issues = []
    
    try:
        # 导入相关模块
        from core.models.lama_processor_simplified import SimplifiedLamaProcessor
        from core.models.base_inpainter import IOPaintBaseProcessor
        from core.inference_manager import InferenceManager
        
        # 定义预期的接口方法
        expected_processor_methods = [
            'get_available_models',
            'predict_with_model',
            'process_image',
            'load_model'
        ]
        
        # 检查 SimplifiedLamaProcessor
        lama_issues = check_class_interface(SimplifiedLamaProcessor, expected_processor_methods)
        if lama_issues:
            issues.extend([f"SimplifiedLamaProcessor: {issue}" for issue in lama_issues])
            print(f"⚠️ SimplifiedLamaProcessor 接口问题: {len(lama_issues)} 个")
        else:
            print("✅ SimplifiedLamaProcessor 接口完整")
        
        # 检查基类接口
        base_issues = check_class_interface(IOPaintBaseProcessor, expected_processor_methods)
        if base_issues:
            issues.extend([f"IOPaintBaseProcessor: {issue}" for issue in base_issues])
            print(f"⚠️ IOPaintBaseProcessor 接口问题: {len(base_issues)} 个")
        else:
            print("✅ IOPaintBaseProcessor 接口完整")
        
        # 检查继承链
        inheritance_issues = check_inheritance_chain(SimplifiedLamaProcessor)
        if inheritance_issues:
            issues.extend([f"继承链: {issue}" for issue in inheritance_issues])
            print(f"⚠️ 继承链问题: {len(inheritance_issues)} 个")
        else:
            print("✅ 继承链检查通过")
        
    except Exception as e:
        issues.append(f"测试处理器接口时出错: {e}")
        print(f"❌ 测试处理器接口时出错: {e}")
    
    return issues

def test_inference_manager_integration():
    """测试推理管理器集成"""
    print("\n🔍 测试推理管理器集成...")
    
    issues = []
    
    try:
        from core.inference_manager import InferenceManager
        from config.config import ConfigManager
        
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 创建推理管理器
        inference_manager = InferenceManager(config_manager)
        
        # 测试关键方法调用
        try:
            # 测试 get_available_models 方法
            models = inference_manager.get_available_models()
            if not isinstance(models, list):
                issues.append("get_available_models 返回类型错误，应该是 list")
            else:
                print(f"✅ get_available_models 返回 {len(models)} 个模型")
        except AttributeError as e:
            issues.append(f"get_available_models 方法缺失: {e}")
        except Exception as e:
            issues.append(f"get_available_models 调用失败: {e}")
        
        # 测试其他关键方法
        try:
            status = inference_manager.get_status()
            if not isinstance(status, dict):
                issues.append("get_status 返回类型错误，应该是 dict")
            else:
                print("✅ get_status 调用成功")
        except Exception as e:
            issues.append(f"get_status 调用失败: {e}")
        
    except Exception as e:
        issues.append(f"测试推理管理器集成时出错: {e}")
        print(f"❌ 测试推理管理器集成时出错: {e}")
    
    return issues

def test_actual_processing_flow():
    """测试实际处理流程"""
    print("\n🔍 测试实际处理流程...")
    
    issues = []
    
    try:
        from core.inference_manager import InferenceManager
        from core.processors.processing_result import ProcessingResult
        from config.config import ConfigManager
        from PIL import Image
        import tempfile
        
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 创建测试图片
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # 创建推理管理器
        inference_manager = InferenceManager(config_manager)
        
        # 测试图片处理
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                test_image.save(tmp_file.name)
                
                # 测试处理流程
                result = inference_manager.process_image(
                    image_path=tmp_file.name,
                    mask_model='simple',
                    mask_params={'mask_threshold': 0.5},
                    inpaint_params={'inpaint_model': 'lama'},
                    performance_params={'max_size': 1024},
                    transparent=False
                )
                
                if isinstance(result, ProcessingResult):
                    print("✅ 图片处理流程正常")
                else:
                    issues.append("process_image 返回类型错误，应该是 ProcessingResult")
                
                # 清理临时文件
                os.unlink(tmp_file.name)
                
        except AttributeError as e:
            issues.append(f"处理方法缺失: {e}")
        except Exception as e:
            issues.append(f"处理流程失败: {e}")
    
    except Exception as e:
        issues.append(f"测试实际处理流程时出错: {e}")
        print(f"❌ 测试实际处理流程时出错: {e}")
    
    return issues

def test_config_consistency():
    """测试配置一致性"""
    print("\n🔍 测试配置一致性...")
    
    issues = []
    
    try:
        from config.config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # 检查必要的配置键
        required_keys = [
            'models',
            'processing',
            'performance'
        ]
        
        for key in required_keys:
            if key not in config:
                issues.append(f"配置缺少必要键: {key}")
            else:
                print(f"✅ 配置键 {key} 存在")
        
        # 检查模型配置
        if 'models' in config:
            models_config = config['models']
            if 'lama' not in models_config:
                issues.append("配置中缺少 LaMA 模型配置")
            else:
                print("✅ LaMA 模型配置存在")
    
    except Exception as e:
        issues.append(f"测试配置一致性时出错: {e}")
        print(f"❌ 测试配置一致性时出错: {e}")
    
    return issues

def test_dynamic_method_check():
    """动态检查方法存在性"""
    print("\n🔍 动态检查方法存在性...")
    
    issues = []
    
    try:
        # 导入相关类
        from core.models.lama_processor_simplified import SimplifiedLamaProcessor
        from core.inference_manager import InferenceManager
        from config.config import ConfigManager
        
        # 创建实例
        config = {'model_path': 'test'}
        processor = SimplifiedLamaProcessor(config)
        
        # 创建配置管理器和推理管理器
        config_manager = ConfigManager()
        inference_manager = InferenceManager(config_manager)
        
        # 检查关键方法是否存在
        critical_methods = [
            (processor, 'get_available_models'),
            (processor, 'predict_with_model'),
            (inference_manager, 'get_available_models'),
            (inference_manager, 'process_image')
        ]
        
        for obj, method_name in critical_methods:
            if not hasattr(obj, method_name):
                issues.append(f"对象 {obj.__class__.__name__} 缺少方法: {method_name}")
            else:
                method = getattr(obj, method_name)
                if not callable(method):
                    issues.append(f"对象 {obj.__class__.__name__} 的 {method_name} 不是可调用对象")
                else:
                    print(f"✅ {obj.__class__.__name__}.{method_name} 方法存在且可调用")
    
    except Exception as e:
        issues.append(f"动态检查方法存在性时出错: {e}")
        print(f"❌ 动态检查方法存在性时出错: {e}")
    
    return issues

def generate_fix_suggestions(issues: List[str]) -> List[str]:
    """生成修复建议"""
    suggestions = []
    
    for issue in issues:
        if "SimplifiedLamaProcessor" in issue and "get_available_models" in issue:
            suggestions.append("""
🔧 修复建议: 为 SimplifiedLamaProcessor 添加 get_available_models 方法

在 core/models/lama_processor_simplified.py 中添加:

def get_available_models(self) -> list:
    \"\"\"返回可用的模型列表\"\"\"
    return ['lama']  # 只支持 LaMA 模型
""")
        
        elif "AttributeError" in issue:
            suggestions.append("""
🔧 修复建议: 检查对象是否实现了预期的接口

确保所有处理器类都实现了统一的接口方法:
- get_available_models()
- predict_with_model()
- process_image()
""")
        
        elif "方法缺失" in issue:
            suggestions.append("""
🔧 修复建议: 实现缺失的方法

检查基类 IOPaintBaseProcessor 是否定义了所有必要的方法，
确保子类正确实现了这些方法。
""")
    
    return suggestions

def main():
    """主函数"""
    print("🔍 运行时问题检测工具")
    print("=" * 60)
    
    all_issues = []
    
    # 运行各项测试
    tests = [
        ("处理器接口一致性", test_processor_interfaces),
        ("推理管理器集成", test_inference_manager_integration),
        ("实际处理流程", test_actual_processing_flow),
        ("配置一致性", test_config_consistency),
        ("动态方法检查", test_dynamic_method_check)
    ]
    
    for test_name, test_func in tests:
        try:
            issues = test_func()
            all_issues.extend(issues)
        except Exception as e:
            all_issues.append(f"{test_name} 测试异常: {e}")
    
    # 输出结果
    print("\n" + "=" * 60)
    print("📊 检测报告")
    print("=" * 60)
    
    if not all_issues:
        print("🎉 没有发现运行时问题！")
    else:
        print(f"⚠️ 发现 {len(all_issues)} 个运行时问题:")
        
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        
        # 生成修复建议
        print("\n" + "=" * 60)
        print("🔧 修复建议")
        print("=" * 60)
        
        suggestions = generate_fix_suggestions(all_issues)
        for suggestion in suggestions:
            print(suggestion)
    
    print(f"\n⏱️ 检测完成")
    
    return len(all_issues) == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 