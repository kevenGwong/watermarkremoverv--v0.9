#!/usr/bin/env python3
"""
启动测试脚本
测试WatermarkRemover-AI项目的启动流程
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_startup():
    """测试程序启动流程"""
    print("🚀 启动测试开始...")
    print("=" * 50)
    
    try:
        # 1. 测试ConfigManager初始化
        print("1. 测试ConfigManager初始化...")
        from config.config import ConfigManager
        config_manager = ConfigManager("web_config.yaml")
        print("✅ ConfigManager initialized")
        
        # 2. 测试InferenceManager初始化
        print("2. 测试InferenceManager初始化...")
        from core.inference import get_inference_manager
        inference_manager = get_inference_manager(config_manager)
        if inference_manager is None:
            raise RuntimeError("InferenceManager returned None")
        print("✅ InferenceManager initialized")
        
        # 3. 测试模型加载
        print("3. 测试模型加载...")
        available_models = inference_manager.get_available_models()
        print(f"✅ Available models: {available_models}")
        
        # 4. 测试UI初始化
        print("4. 测试UI初始化...")
        from interfaces.web.ui import MainInterface
        main_interface = MainInterface(config_manager)
        print("✅ MainInterface initialized")
        
        # 5. 测试系统信息获取
        print("5. 测试系统信息获取...")
        from core.inference import get_system_info
        system_info = get_system_info(config_manager)
        print(f"✅ System info: {system_info}")
        
        print("\n" + "=" * 50)
        print("🎉 启动测试全部通过！")
        print("=" * 50)
        return True
        
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"❌ 启动测试失败: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_startup()
    sys.exit(0 if success else 1) 