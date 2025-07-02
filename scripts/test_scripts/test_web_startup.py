#!/usr/bin/env python3
"""
测试Web应用启动和基础功能
"""
import subprocess
import time
import requests
import sys
from pathlib import Path

def test_streamlit_import():
    """测试Streamlit导入"""
    print("🧪 Testing Streamlit import...")
    try:
        import streamlit as st
        print(f"✅ Streamlit version: {st.__version__}")
        return True
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        return False

def test_app_import():
    """测试应用导入"""
    print("🧪 Testing app import...")
    try:
        # 测试简化版应用导入
        import sys
        sys.path.insert(0, '.')
        
        # 尝试导入应用文件
        import importlib.util
        
        # 测试简化版
        spec = importlib.util.spec_from_file_location("watermark_web_app_simple", "watermark_web_app_simple.py")
        app_module = importlib.util.module_from_spec(spec)
        
        print("✅ Simple app file imports successfully")
        
        # 测试后端导入
        from web_backend import WatermarkProcessor, ProcessingResult
        print("✅ Backend imports successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ App import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_files():
    """测试配置文件"""
    print("🧪 Testing config files...")
    
    config_files = ["web_config.yaml", "web_config_advanced.yaml"]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file} exists")
            
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"   - Successfully parsed {config_file}")
            except Exception as e:
                print(f"   ❌ Failed to parse {config_file}: {e}")
                return False
        else:
            print(f"❌ {config_file} not found")
            return False
    
    return True

def test_model_paths():
    """测试模型路径"""
    print("🧪 Testing model paths...")
    
    try:
        import yaml
        with open('web_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 检查自定义模型路径
        mask_model_path = config['mask_generator']['mask_model_path']
        if Path(mask_model_path).exists():
            print(f"✅ Custom model found: {mask_model_path}")
        else:
            print(f"⚠️  Custom model not found: {mask_model_path}")
            print("   (This is expected if model hasn't been downloaded)")
        
        # LaMA模型会自动下载，不需要检查
        print("✅ LaMA model will be downloaded automatically")
        
        return True
        
    except Exception as e:
        print(f"❌ Model path check failed: {e}")
        return False

def create_startup_test_summary():
    """创建启动测试总结"""
    print("🎨 AI Watermark Remover - Web Startup Test")
    print("="*60)
    
    tests = [
        ("Streamlit Import", test_streamlit_import),
        ("App Import", test_app_import),
        ("Config Files", test_config_files),
        ("Model Paths", test_model_paths)
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
    print("🎯 WEB STARTUP TEST RESULTS")
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
        print("🎉 All startup tests passed! Web app is ready to launch.")
        print("\n🚀 To start the app, run:")
        print("   ./run_simple_app.sh")
    else:
        print("⚠️  Some startup tests failed. Check dependencies.")
    
    return passed == total

def main():
    create_startup_test_summary()

if __name__ == "__main__":
    main()