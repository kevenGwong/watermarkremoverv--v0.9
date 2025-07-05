#!/usr/bin/env python3
"""
测试ZITS、MAT、FCF三个模型的完整处理流程
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

def create_test_image_and_mask():
    """创建测试图像和mask"""
    # 创建测试图像 (512x512)
    image_size = (512, 512)
    test_image = Image.new('RGB', image_size, color='white')
    
    # 在图像中添加一些内容（模拟真实场景）
    import numpy as np
    img_array = np.array(test_image)
    # 添加一些颜色区域
    img_array[100:200, 100:200] = [255, 100, 100]  # 红色区域
    img_array[300:400, 200:300] = [100, 255, 100]  # 绿色区域
    img_array[200:300, 350:450] = [100, 100, 255]  # 蓝色区域
    test_image = Image.fromarray(img_array)
    
    # 创建测试mask - 模拟水印区域
    test_mask = Image.new('L', image_size, color=0)  # 黑色背景
    mask_array = np.array(test_mask)
    
    # 在mask中心添加一个白色区域（模拟水印）
    center_x, center_y = image_size[0] // 2, image_size[1] // 2
    mask_array[center_y-50:center_y+50, center_x-100:center_x+100] = 255  # 白色水印区域
    test_mask = Image.fromarray(mask_array, mode='L')
    
    return test_image, test_mask

def test_model(model_name, test_image, test_mask):
    """测试指定模型"""
    print(f"\n🧪 测试 {model_name.upper()} 模型...")
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    # 设置参数
    mask_params = {
        'uploaded_mask': test_mask,
        'mask_dilate_kernel_size': 3,
        'mask_dilate_iterations': 1
    }
    
    inpaint_params = {
        'inpaint_model': 'iopaint',
        'force_model': model_name,
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
            mask_model='upload',
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params=performance_params,
            transparent=False,
            config_manager=config_manager
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ {model_name.upper()} 处理完成，耗时: {processing_time:.2f}秒")
        print(f"   处理成功: {result.success}")
        
        if result.success:
            print(f"   结果图像尺寸: {result.result_image.size}")
            print(f"   Mask 图像尺寸: {result.mask_image.size}")
            
            # 检查mask覆盖率
            mask_array = np.array(result.mask_image)
            coverage = np.sum(mask_array > 128) / mask_array.size * 100
            print(f"   Mask 覆盖率: {coverage:.2f}%")
            
            # 保存结果
            result_path = f"scripts/test_{model_name}_result.png"
            result.result_image.save(result_path)
            print(f"   结果已保存到: {result_path}")
            
            return True, processing_time
        else:
            print(f"❌ {model_name.upper()} 处理失败: {result.error_message}")
            return False, processing_time
            
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"❌ {model_name.upper()} 测试异常: {str(e)}")
        logger.error(f"Model {model_name} test failed with exception: {e}")
        return False, processing_time

def test_all_models():
    """测试所有三个模型"""
    print("🚀 开始测试所有模型...")
    
    # 创建测试数据
    test_image, test_mask = create_test_image_and_mask()
    print(f"📏 测试图像尺寸: {test_image.size}")
    print(f"📏 测试mask尺寸: {test_mask.size}")
    
    # 保存测试图像和mask
    test_image.save("scripts/test_input_image.png")
    test_mask.save("scripts/test_input_mask.png")
    print("📁 测试图像和mask已保存")
    
    # 测试三个模型
    models = ['zits', 'mat', 'fcf']
    results = {}
    
    for model_name in models:
        success, processing_time = test_model(model_name, test_image, test_mask)
        results[model_name] = {'success': success, 'time': processing_time}
    
    # 输出总结
    print("\n📊 测试结果总结:")
    print("=" * 50)
    for model_name, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"{model_name.upper():>8}: {status} (耗时: {result['time']:.2f}秒)")
    
    # 统计成功率
    successful_models = [m for m, r in results.items() if r['success']]
    success_rate = len(successful_models) / len(models) * 100
    print(f"\n🎯 总体成功率: {success_rate:.1f}% ({len(successful_models)}/{len(models)})")
    
    if successful_models:
        print(f"✅ 成功的模型: {', '.join(successful_models).upper()}")
    
    failed_models = [m for m, r in results.items() if not r['success']]
    if failed_models:
        print(f"❌ 失败的模型: {', '.join(failed_models).upper()}")
    
    return results

if __name__ == "__main__":
    results = test_all_models()