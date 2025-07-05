#!/usr/bin/env python3
"""
测试Streamlit UI启动是否正常
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_streamlit_imports():
    """测试Streamlit相关导入"""
    print("🧪 测试Streamlit导入...")
    
    try:
        import streamlit as st
        print("✅ Streamlit导入成功")
        
        # 测试相关库
        from streamlit_image_comparison import image_comparison
        print("✅ streamlit_image_comparison导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ Streamlit导入失败: {e}")
        return False

def test_ui_initialization():
    """测试UI组件初始化"""
    print("\n🧪 测试UI组件初始化...")
    
    try:
        from config.config import ConfigManager
        from interfaces.web.ui import MainInterface, ParameterPanel
        
        # 创建配置管理器
        config_manager = ConfigManager()
        print("✅ ConfigManager创建成功")
        
        # 创建参数面板
        parameter_panel = ParameterPanel(config_manager)
        print("✅ ParameterPanel创建成功")
        
        # 创建主界面
        main_interface = MainInterface(config_manager)
        print("✅ MainInterface创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ UI初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_initialization():
    """测试推理组件初始化"""
    print("\n🧪 测试推理组件初始化...")
    
    try:
        from config.config import ConfigManager
        from core.inference import get_inference_manager
        
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 获取推理管理器
        inference_manager = get_inference_manager(config_manager)
        print("✅ InferenceManager创建成功")
        
        if inference_manager:
            print("✅ 推理管理器正常工作")
            return True
        else:
            print("❌ 推理管理器创建失败")
            return False
            
    except Exception as e:
        print(f"❌ 推理初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔍 Streamlit UI启动测试")
    print("=" * 40)
    
    tests = [
        ("Streamlit导入", test_streamlit_imports),
        ("UI组件初始化", test_ui_initialization),
        ("推理组件初始化", test_inference_initialization)
    ]
    
    success_count = 0
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            success_count += 1
    
    total_tests = len(tests)
    success_rate = (success_count / total_tests) * 100
    
    print(f"\n{'=' * 40}")
    print(f"📊 测试结果: {success_count}/{total_tests} 通过 ({success_rate:.1f}%)")
    
    if success_count == total_tests:
        print("🎉 所有测试通过! Streamlit UI可以正常启动。")
        print("\n启动命令:")
        print("conda activate py310aiwatermark")
        print("cd /home/duolaameng/SAM_Remove/WatermarkRemover-AI")
        print("streamlit run interfaces/web/main.py")
    else:
        print("⚠️ 部分测试失败，UI可能无法正常启动。")

if __name__ == "__main__":
    main()