#!/usr/bin/env python3
"""
快速测试脚本 - 模拟用户上传图片和mask的处理流程
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image
from config.config import ConfigManager
from core.inference import process_image

def create_test_images():
    """创建测试图片 - 模拟手表图片和水印mask"""
    # 创建模拟手表图片
    watch_image = Image.new('RGB', (300, 300), color='white')
    watch_pixels = np.array(watch_image)
    
    # 添加手表样式
    watch_pixels[50:250, 50:250] = [50, 50, 50]  # 表盘
    watch_pixels[140:160, 140:160] = [200, 200, 200]  # 中心
    watch_pixels[150, 80:220] = [255, 255, 255]  # 指针1
    watch_pixels[80:220, 150] = [255, 255, 255]  # 指针2
    
    watch_image = Image.fromarray(watch_pixels)
    
    # 创建水印mask - 模拟邮箱水印位置
    mask_image = Image.new('L', (300, 300), color=0)  # 黑色背景
    mask_pixels = np.array(mask_image)
    
    # 水印区域 (模拟右下角邮箱水印)
    mask_pixels[200:250, 50:250] = 255  # 白色水印区域
    
    mask_image = Image.fromarray(mask_pixels, mode='L')
    
    return watch_image, mask_image

def main():
    print("🔍 快速处理测试...")
    
    # 创建测试图片
    test_image, mask_image = create_test_images()
    
    # 保存测试图片以便查看
    test_image.save("/tmp/test_watch.jpg")
    mask_image.save("/tmp/test_mask.png")
    print("✅ 测试图片已保存到 /tmp/test_watch.jpg 和 /tmp/test_mask.png")
    
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 准备处理参数
    mask_params = {'uploaded_mask': mask_image}
    inpaint_params = {
        'model_name': 'lama',
        'ldm_steps': 20,
        'hd_strategy': 'ORIGINAL'
    }
    
    print("🎨 开始处理...")
    
    try:
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
            print(f"✅ 处理成功!")
            print(f"   耗时: {result.processing_time:.2f}s")
            print(f"   输入尺寸: {test_image.size}")
            print(f"   输出尺寸: {result.result_image.size}")
            
            # 保存结果
            result.result_image.save("/tmp/result_image.jpg")
            print("   结果已保存到: /tmp/result_image.jpg")
            
            if result.mask_image:
                result.mask_image.save("/tmp/used_mask.png")
                print("   使用的mask已保存到: /tmp/used_mask.png")
                
            return True
        else:
            print(f"❌ 处理失败: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ 处理异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 测试通过! 核心功能正常工作。")
        print("现在可以启动 Streamlit UI 进行完整测试。")
    else:
        print("\n⚠️ 测试失败，需要进一步调试。")