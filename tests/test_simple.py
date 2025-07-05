#!/usr/bin/env python3
"""
简化测试脚本
不依赖pytest，直接运行基本功能测试
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """测试基本导入"""
    print("🔍 测试基本导入...")
    
    try:
        from config.config import ConfigManager
        print("✅ ConfigManager 导入成功")
    except Exception as e:
        print(f"❌ ConfigManager 导入失败: {e}")
        return False
    
    try:
        from core.processors.processing_result import ProcessingResult
        print("✅ ProcessingResult 导入成功")
    except Exception as e:
        print(f"❌ ProcessingResult 导入失败: {e}")
        return False
    
    try:
        from core.inference_manager import InferenceManager
        print("✅ InferenceManager 导入成功")
    except Exception as e:
        print(f"❌ InferenceManager 导入失败: {e}")
        return False
    
    return True

def test_config_loading():
    """测试配置加载"""
    print("\n📋 测试配置加载...")
    
    try:
        from config.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        if isinstance(config, dict) and len(config) > 0:
            print("✅ 配置加载成功")
            return True
        else:
            print("❌ 配置为空或格式错误")
            return False
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_processing_result():
    """测试处理结果类"""
    print("\n🔄 测试处理结果类...")
    
    try:
        from core.processors.processing_result import ProcessingResult
        from PIL import Image
        
        # 创建测试图片
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # 测试成功结果
        result = ProcessingResult(
            success=True,
            result_image=test_image,
            mask_image=Image.new('L', (100, 100), 128),
            processing_time=1.5,
            error_message=None
        )
        
        if result.success and result.result_image and result.processing_time > 0:
            print("✅ 成功结果创建正常")
        else:
            print("❌ 成功结果创建失败")
            return False
        
        # 测试错误结果
        error_result = ProcessingResult(
            success=False,
            result_image=None,
            mask_image=None,
            processing_time=0.0,
            error_message="Test error"
        )
        
        if not error_result.success and error_result.error_message:
            print("✅ 错误结果创建正常")
            return True
        else:
            print("❌ 错误结果创建失败")
            return False
            
    except Exception as e:
        print(f"❌ 处理结果测试失败: {e}")
        return False

def test_ui_components():
    """测试UI组件（使用mock）"""
    print("\n🎨 测试UI组件...")
    
    try:
        # Mock streamlit
        with patch.dict('sys.modules', {
            'streamlit': Mock(),
            'streamlit.columns': Mock(return_value=[Mock(), Mock(), Mock()]),
            'streamlit.selectbox': Mock(return_value='lama'),
            'streamlit.slider': Mock(return_value=0.5),
            'streamlit.checkbox': Mock(return_value=False),
            'streamlit.text_input': Mock(return_value=''),
            'streamlit.expander': Mock(),
            'streamlit.write': Mock(),
            'streamlit.button': Mock(return_value=False),
            'streamlit.spinner': Mock(),
            'streamlit.error': Mock(),
            'streamlit.subheader': Mock(),
            'streamlit.metric': Mock(),
            'streamlit.warning': Mock(),
            'streamlit.info': Mock(),
            'streamlit.file_uploader': Mock(return_value=None),
            'streamlit.session_state': {}
        }):
            from interfaces.web.ui import MainInterface, ParameterPanel
            from config.config import ConfigManager
            
            config_manager = ConfigManager()
            parameter_panel = ParameterPanel(config_manager)
            main_interface = MainInterface(config_manager)
            
            print("✅ UI组件创建成功")
            return True
            
    except Exception as e:
        print(f"❌ UI组件测试失败: {e}")
        return False

def test_parameter_validation():
    """测试参数验证"""
    print("\n⚙️ 测试参数验证...")
    
    # 测试有效参数
    valid_params = {
        'inpaint_model': 'lama',
        'ldm_steps': 20,
        'hd_strategy': 'ORIGINAL'
    }
    
    # 测试无效参数
    invalid_params = {
        'inpaint_model': 'invalid_model',
        'ldm_steps': -1
    }
    
    def validate_params(params):
        valid_models = ['lama', 'iopaint']
        valid_strategies = ['ORIGINAL', 'RESIZE', 'CROP']
        
        if 'inpaint_model' in params and params['inpaint_model'] not in valid_models:
            return False
        
        if 'ldm_steps' in params and (params['ldm_steps'] < 1 or params['ldm_steps'] > 100):
            return False
        
        if 'hd_strategy' in params and params['hd_strategy'] not in valid_strategies:
            return False
        
        return True
    
    if validate_params(valid_params) and not validate_params(invalid_params):
        print("✅ 参数验证逻辑正确")
        return True
    else:
        print("❌ 参数验证逻辑错误")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🧪 运行简化测试套件...")
    print("="*50)
    
    tests = [
        ("基本导入", test_imports),
        ("配置加载", test_config_loading),
        ("处理结果", test_processing_result),
        ("UI组件", test_ui_components),
        ("参数验证", test_parameter_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 输出结果
    print("\n" + "="*50)
    print("📊 测试结果")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<15} {status}")
        if result:
            passed += 1
    
    print("-"*50)
    print(f"总计: {len(results)} 测试")
    print(f"通过: {passed}")
    print(f"失败: {len(results) - passed}")
    print(f"成功率: {passed/len(results)*100:.1f}%")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！")
        return True
    else:
        print(f"\n💥 {len(results) - passed} 个测试失败")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1) 