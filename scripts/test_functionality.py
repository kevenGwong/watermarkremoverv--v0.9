#!/usr/bin/env python3
"""
功能测试脚本
测试WatermarkRemover-AI项目的基本功能
"""

import sys
import numpy as np
from PIL import Image
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 功能测试开始...")
    print("=" * 50)
    
    try:
        from config.config import ConfigManager
        from core.inference import get_inference_manager, process_image
        
        # 1. 初始化
        print("1. 初始化配置和推理管理器...")
        config_manager = ConfigManager("web_config.yaml")
        inference_manager = get_inference_manager(config_manager)
        print("✅ 初始化成功")
        
        # 2. 创建测试图像
        print("2. 创建测试图像...")
        test_image = Image.new('RGB', (512, 512), 'red')
        print("✅ 测试图像创建成功")
        
        # 3. 测试mask生成
        print("3. 测试mask生成...")
        mask_params = {
            'mask_threshold': 0.5,
            'mask_dilate_kernel_size': 3,
            'mask_dilate_iterations': 1
        }
        print("✅ Mask参数设置成功")
        
        # 4. 测试inpainting参数
        print("4. 测试inpainting参数...")
        inpaint_params = {
            'force_model': 'mat',
            'ldm_steps': 20,
            'hd_strategy': 'CROP'
        }
        print("✅ Inpainting参数设置成功")
        
        # 5. 测试图像处理
        print("5. 测试图像处理...")
        result = process_image(
            image=test_image,
            mask_model='custom',
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params={},
            transparent=False,
            config_manager=config_manager
        )
        
        if result.success:
            print("✅ 图像处理成功")
            print(f"   处理时间: {result.processing_time:.2f}秒")
            if result.result_image:
                print(f"   结果图像尺寸: {result.result_image.size}")
            if result.mask_image:
                print(f"   Mask图像尺寸: {result.mask_image.size}")
        else:
            print(f"❌ 图像处理失败: {result.error_message}")
            return False
        
        print("\n" + "=" * 50)
        print("🎉 功能测试全部通过！")
        print("=" * 50)
        return True
        
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"❌ 功能测试失败: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1) 