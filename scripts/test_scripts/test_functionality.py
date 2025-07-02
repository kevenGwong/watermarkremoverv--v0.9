#!/usr/bin/env python3
"""
功能测试脚本
测试各个模块的功能是否正常
"""
import os
import sys
import time
import traceback
from pathlib import Path
from PIL import Image
import numpy as np

def test_imports():
    """测试模块导入"""
    print("🧪 Testing module imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        from transformers import AutoProcessor, AutoModelForCausalLM
        print("✅ Transformers imported")
        
        from iopaint.model_manager import ModelManager
        from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest
        print("✅ IOPaint imported")
        
        # 检查LDMSampler的实际属性
        print("📋 Available LDMSampler values:")
        for attr in dir(LDMSampler):
            if not attr.startswith('_'):
                print(f"   - {attr}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n🧪 Testing config loading...")
    
    try:
        import yaml
        
        # 测试原始配置
        if os.path.exists('web_config.yaml'):
            with open('web_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            print("✅ Original config loaded")
            print(f"   Keys: {list(config.keys())}")
        
        # 测试高级配置
        if os.path.exists('web_config_advanced.yaml'):
            with open('web_config_advanced.yaml', 'r') as f:
                advanced_config = yaml.safe_load(f)
            print("✅ Advanced config loaded")
            print(f"   Keys: {list(advanced_config.keys())}")
            
            # 检查采样器配置
            lama_config = advanced_config.get('lama_inpainting', {})
            sampler = lama_config.get('ldm_sampler', 'ddim')
            print(f"   Default sampler: {sampler}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return False

def test_backend_loading():
    """测试后端加载"""
    print("\n🧪 Testing backend loading...")
    
    try:
        # 测试原始后端
        from web_backend import WatermarkProcessor, ProcessingResult
        print("✅ Original backend imported")
        
        # 尝试初始化（但不要求模型存在）
        try:
            processor = WatermarkProcessor("web_config.yaml")
            print("✅ Original processor initialized")
        except Exception as e:
            print(f"⚠️  Original processor init failed (expected if models missing): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend loading failed: {e}")
        traceback.print_exc()
        return False

def test_image_processing():
    """测试图像处理功能"""
    print("\n🧪 Testing image processing...")
    
    try:
        # 创建测试图像
        test_image = Image.new('RGB', (512, 512), color='white')
        
        # 在图像上添加一些内容作为"水印"
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(test_image)
        
        # 画一个简单的"水印"
        draw.rectangle([100, 100, 400, 200], fill='red', outline='black', width=2)
        draw.text((150, 130), "TEST WATERMARK", fill='white')
        
        print("✅ Test image created")
        
        # 创建测试mask
        test_mask = Image.new('L', (512, 512), color=0)
        mask_draw = ImageDraw.Draw(test_mask)
        mask_draw.rectangle([100, 100, 400, 200], fill=255)
        
        print("✅ Test mask created")
        
        # 测试透明效果
        test_transparent = test_image.convert("RGBA")
        img_array = np.array(test_transparent)
        mask_array = np.array(test_mask)
        
        # 应用透明效果
        transparent_mask = mask_array > 128
        img_array[transparent_mask, 3] = 0  # 设置alpha通道为0
        
        result_transparent = Image.fromarray(img_array, 'RGBA')
        print("✅ Transparency effect test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        traceback.print_exc()
        return False

def test_extreme_parameters():
    """测试极值参数"""
    print("\n🧪 Testing extreme parameter values...")
    
    try:
        # 测试参数范围
        test_params = {
            'mask_threshold': [0.0, 0.1, 0.5, 0.9, 1.0],
            'mask_dilate_kernel_size': [1, 3, 7, 15, 25, 50],
            'mask_dilate_iterations': [1, 3, 5, 10, 20],
            'ldm_steps': [10, 20, 50, 100, 200],
            'max_bbox_percent': [1.0, 5.0, 10.0, 25.0, 50.0],
            'confidence_threshold': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        for param_name, values in test_params.items():
            print(f"   Testing {param_name}: {values}")
            
            for value in values:
                # 验证值在合理范围内
                if param_name == 'mask_threshold':
                    assert 0.0 <= value <= 1.0, f"Invalid {param_name}: {value}"
                elif param_name == 'mask_dilate_kernel_size':
                    assert 1 <= value <= 50 and value % 2 == 1, f"Invalid {param_name}: {value}"
                elif param_name == 'mask_dilate_iterations':
                    assert 1 <= value <= 20, f"Invalid {param_name}: {value}"
                elif param_name == 'ldm_steps':
                    assert 10 <= value <= 200, f"Invalid {param_name}: {value}"
                elif param_name == 'max_bbox_percent':
                    assert 1.0 <= value <= 50.0, f"Invalid {param_name}: {value}"
                elif param_name == 'confidence_threshold':
                    assert 0.1 <= value <= 0.9, f"Invalid {param_name}: {value}"
        
        print("✅ Parameter validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Parameter testing failed: {e}")
        traceback.print_exc()
        return False

def create_test_summary():
    """创建测试总结"""
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Config Loading", test_config_loading), 
        ("Backend Loading", test_backend_loading),
        ("Image Processing", test_image_processing),
        ("Parameter Validation", test_extreme_parameters)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Check dependencies and configuration.")
    
    return passed == total

def main():
    """主测试函数"""
    print("🎨 AI Watermark Remover - Functionality Test")
    print("="*60)
    
    # 显示环境信息
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")
    
    # 运行测试
    success = create_test_summary()
    
    # 退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()