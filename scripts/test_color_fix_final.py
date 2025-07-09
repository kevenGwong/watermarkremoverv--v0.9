#!/usr/bin/env python3
"""
最终颜色修复验证测试
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_color_test_image():
    """创建颜色测试图像：明确的红色和蓝色区域"""
    image = Image.new('RGB', (100, 100), (255, 255, 255))  # 白色背景
    
    # 左侧红色区域
    for x in range(0, 50):
        for y in range(0, 100):
            image.putpixel((x, y), (255, 0, 0))  # 纯红色
    
    # 右侧蓝色区域  
    for x in range(50, 100):
        for y in range(0, 100):
            image.putpixel((x, y), (0, 0, 255))  # 纯蓝色
    
    return image

def create_center_mask():
    """创建中心mask：只处理中心部分"""
    mask = Image.new('L', (100, 100), 0)  # 黑色背景
    mask_array = np.array(mask)
    mask_array[25:75, 25:75] = 255  # 中心50x50白色区域
    return Image.fromarray(mask_array, mode='L')

def test_color_fix():
    """测试颜色修复效果"""
    print("🎨 最终颜色修复验证测试")
    print("=" * 50)
    
    # 创建测试数据
    test_image = create_color_test_image()
    test_mask = create_center_mask()
    
    print("📸 测试图像:")
    red_pixel = test_image.getpixel((25, 50))   # 左侧红色区域
    blue_pixel = test_image.getpixel((75, 50))  # 右侧蓝色区域
    print(f"   左侧红色区域: RGB{red_pixel}")
    print(f"   右侧蓝色区域: RGB{blue_pixel}")
    
    try:
        from config.config import ConfigManager
        from core.inference import process_image
        
        config_manager = ConfigManager()
        
        # 测试MAT模型
        print(f"\n🚀 测试MAT模型处理...")
        
        result = process_image(
            image=test_image,
            mask_model='upload',
            mask_params={'uploaded_mask': test_mask},
            inpaint_params={
                'model_name': 'mat',
                'ldm_steps': 5,  # 快速测试
                'hd_strategy': 'ORIGINAL'
            },
            performance_params={},
            transparent=False,
            config_manager=config_manager
        )
        
        if result.success:
            # 检查结果颜色
            result_red = result.result_image.getpixel((10, 50))   # 左侧红色区域（未被mask覆盖）
            result_blue = result.result_image.getpixel((90, 50))  # 右侧蓝色区域（未被mask覆盖）
            
            print(f"✅ MAT处理成功")
            print(f"📊 结果颜色:")
            print(f"   左侧结果: RGB{result_red}")
            print(f"   右侧结果: RGB{result_blue}")
            
            # 颜色匹配检查
            red_match = abs(result_red[0] - red_pixel[0]) < 10 and result_red[1] < 10 and result_red[2] < 10
            blue_match = abs(result_blue[2] - blue_pixel[2]) < 10 and result_blue[0] < 10 and result_blue[1] < 10
            
            print(f"\n🔍 颜色匹配检查:")
            print(f"   红色区域匹配: {'✅ 正确' if red_match else '❌ 错误'}")
            print(f"   蓝色区域匹配: {'✅ 正确' if blue_match else '❌ 错误'}")
            
            if red_match and blue_match:
                print(f"\n🎉 颜色修复验证成功！红蓝通道正确。")
                return True
            else:
                print(f"\n❌ 颜色修复失败，仍有红蓝通道问题。")
                return False
                
        else:
            print(f"❌ MAT处理失败: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🔬 最终颜色修复验证")
    print("=" * 60)
    
    success = test_color_fix()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！颜色通道修复成功！")
        print("✅ 用户在UI中选择不同模型时，颜色将正确显示。")
    else:
        print("❌ 测试失败，需要进一步调试。")

if __name__ == "__main__":
    main()