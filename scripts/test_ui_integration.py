#!/usr/bin/env python3
"""
UI集成测试脚本
测试实际的图片处理流程，发现并修复所有运行时错误
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
import tempfile
from PIL import Image
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_images():
    """创建测试图片和mask"""
    # 创建测试图片 (模拟手表图片)
    test_image = Image.new('RGB', (400, 400), color='white')
    # 添加一些图案
    pixels = np.array(test_image)
    pixels[100:300, 100:300] = [100, 100, 100]  # 灰色区域
    pixels[150:250, 150:250] = [200, 200, 200]  # 更亮的中心
    test_image = Image.fromarray(pixels)
    
    # 创建测试mask (模拟水印区域)
    mask_image = Image.new('L', (400, 400), color=0)  # 黑色背景
    mask_pixels = np.array(mask_image)
    mask_pixels[180:220, 180:220] = 255  # 白色水印区域
    mask_image = Image.fromarray(mask_pixels, mode='L')
    
    return test_image, mask_image

def test_core_processing():
    """测试核心处理流程"""
    print("🧪 测试核心处理流程...")
    
    try:
        from config.config import ConfigManager
        from core.inference import process_image
        
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 创建测试图片
        test_image, mask_image = create_test_images()
        
        # 准备参数
        mask_params = {'uploaded_mask': mask_image}
        inpaint_params = {
            'model_name': 'lama',
            'ldm_steps': 20,
            'hd_strategy': 'ORIGINAL'
        }
        
        print("✅ 开始处理图片...")
        
        # 执行处理
        result = process_image(
            image=test_image,
            mask_model='upload',
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            transparent=False,
            config_manager=config_manager
        )
        
        if result.success:
            print(f"✅ 处理成功! 耗时: {result.processing_time:.2f}s")
            print(f"   结果图片尺寸: {result.result_image.size}")
            return True
        else:
            print(f"❌ 处理失败: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_components():
    """测试UI组件"""
    print("\n🧪 测试UI组件...")
    
    try:
        from config.config import ConfigManager
        from interfaces.web.ui import ParameterPanel, MainInterface
        
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 测试参数面板
        print("✅ 创建参数面板...")
        parameter_panel = ParameterPanel(config_manager)
        
        # 测试主界面
        print("✅ 创建主界面...")
        main_interface = MainInterface(config_manager)
        
        print("✅ UI组件创建成功")
        return True
        
    except Exception as e:
        print(f"❌ UI测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_calls():
    """测试接口调用"""
    print("\n🧪 测试接口调用...")
    
    try:
        from config.config import ConfigManager
        from core.inference_manager import InferenceManager
        
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 创建推理管理器
        print("✅ 创建推理管理器...")
        inference_manager = InferenceManager(config_manager)
        
        # 加载处理器
        print("✅ 加载处理器...")
        if inference_manager.load_processor():
            print("✅ 处理器加载成功")
        else:
            print("⚠️ 处理器加载失败")
        
        # 测试状态获取
        status = inference_manager.get_status()
        print(f"✅ 获取状态: {status}")
        
        # 测试可用模型
        models = inference_manager.get_available_models()
        print(f"✅ 可用模型: {models}")
        
        return True
        
    except Exception as e:
        print(f"❌ 接口测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔍 UI集成测试开始...")
    print("=" * 50)
    
    # 设置环境
    os.environ['PYTHONPATH'] = str(project_root)
    
    # 运行测试
    tests = [
        ("UI组件测试", test_ui_components),
        ("接口调用测试", test_interface_calls),
        ("核心处理测试", test_core_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        success = test_func()
        results.append((test_name, success))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    success_count = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
        if success:
            success_count += 1
    
    total_tests = len(results)
    success_rate = (success_count / total_tests) * 100
    
    print(f"\n总计: {success_count}/{total_tests} 通过 ({success_rate:.1f}%)")
    
    if success_count == total_tests:
        print("🎉 所有测试通过! UI应该可以正常工作。")
    else:
        print("⚠️ 部分测试失败，需要进一步修复。")

if __name__ == "__main__":
    main()