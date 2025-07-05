#!/usr/bin/env python3
"""
测试修复后的mask上传功能
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from core.inference import process_image
from config.config import ConfigManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mask_upload_fix():
    """测试修复后的mask上传功能"""
    print("🧪 测试修复后的mask上传功能...")
    
    # 创建测试图像
    size = (512, 512)
    test_image = Image.new('RGB', size, color='white')
    img_array = np.array(test_image)
    # 添加一些内容
    img_array[100:200, 100:200] = [255, 100, 100]  # 红色区域
    img_array[300:400, 200:300] = [100, 255, 100]  # 绿色区域
    test_image = Image.fromarray(img_array)
    
    # 创建测试mask
    test_mask = Image.new('L', size, color=0)  # 黑色背景
    mask_array = np.array(test_mask)
    # 在mask中心添加白色区域（模拟水印）
    mask_array[150:250, 150:250] = 255  # 白色水印区域
    test_mask = Image.fromarray(mask_array, mode='L')
    
    # 保存测试文件
    test_image.save("scripts/test_mask_fix_input.png")
    test_mask.save("scripts/test_mask_fix_mask.png")
    print("📁 测试文件已保存")
    
    # 检查mask内容
    mask_coverage = np.sum(mask_array > 128) / mask_array.size * 100
    print(f"📊 原始mask覆盖率: {mask_coverage:.2f}%")
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    # 设置参数
    mask_params = {
        'uploaded_mask': test_mask,  # 直接传递PIL Image
        'mask_dilate_kernel_size': 3,
        'mask_dilate_iterations': 1
    }
    
    inpaint_params = {
        'inpaint_model': 'iopaint',
        'force_model': 'fcf',  # 使用最快的模型
        'auto_model_selection': False,
        'ldm_steps': 20,
        'hd_strategy': 'ORIGINAL',
        'seed': -1
    }
    
    performance_params = {
        'mixed_precision': True,
        'log_processing_time': True
    }
    
    # 开始处理
    start_time = time.time()
    
    try:
        result = process_image(
            image=test_image,
            mask_model='upload',  # 使用upload模式
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params=performance_params,
            transparent=False,
            config_manager=config_manager
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result.success:
            # 分析结果mask
            result_mask_array = np.array(result.mask_image.convert("L"))
            result_coverage = np.sum(result_mask_array > 128) / result_mask_array.size * 100
            
            # 保存结果
            result.result_image.save("scripts/test_mask_fix_result.png")
            result.mask_image.save("scripts/test_mask_fix_result_mask.png")
            
            print(f"✅ Mask上传修复测试成功!")
            print(f"   耗时: {processing_time:.2f}秒")
            print(f"   原始mask覆盖率: {mask_coverage:.2f}%")
            print(f"   处理后mask覆盖率: {result_coverage:.2f}%")
            print(f"   Mask传递成功: {'✅' if result_coverage > 0 else '❌'}")
            print(f"   结果已保存: scripts/test_mask_fix_result.png")
            print(f"   处理后mask已保存: scripts/test_mask_fix_result_mask.png")
            
            return True, result_coverage > 0
        else:
            print(f"❌ Mask上传修复测试失败: {result.error_message}")
            return False, False
            
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"❌ Mask上传修复测试异常: {str(e)}")
        return False, False

if __name__ == "__main__":
    success, mask_working = test_mask_upload_fix()
    if success and mask_working:
        print("\n🎉 Mask上传功能修复成功!")
    else:
        print("\n❌ Mask上传功能仍有问题，需要进一步调试")