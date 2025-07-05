#!/usr/bin/env python3
"""
详细分析颜色通道处理管道
追踪图像从输入到输出的每一步颜色变化
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image():
    """创建测试图像：红色背景，便于识别颜色通道"""
    # 创建红色图像 RGB(255, 50, 50)
    image = Image.new('RGB', (256, 256), (255, 50, 50))
    
    # 在中心添加蓝色区域 RGB(50, 50, 255)，便于对比
    pixels = image.load()
    for x in range(100, 156):
        for y in range(100, 156):
            pixels[x, y] = (50, 50, 255)
    
    return image

def create_test_mask():
    """创建测试mask：中心白色区域"""
    mask = Image.new('L', (256, 256), 0)
    mask_array = np.array(mask)
    mask_array[100:156, 100:156] = 255  # 中心56x56白色区域
    return Image.fromarray(mask_array, mode='L')

def analyze_color_at_each_step():
    """分析每一步的颜色变化"""
    print("🎨 颜色通道管道分析")
    print("=" * 60)
    
    # Step 1: 创建测试数据
    test_image = create_test_image()
    test_mask = create_test_mask()
    
    print("📸 Step 1: 原始测试图像")
    red_pixel = test_image.getpixel((50, 50))    # 红色区域
    blue_pixel = test_image.getpixel((128, 128))  # 蓝色区域
    print(f"   红色区域像素: RGB{red_pixel}")
    print(f"   蓝色区域像素: RGB{blue_pixel}")
    
    # Step 2: UI输入处理
    print("\n🖥️ Step 2: UI输入处理")
    try:
        from core.utils.image_utils import ImageUtils
        
        # 检查UI层的图像预处理
        processed_image, processed_mask = ImageUtils.preprocess_for_model(test_image, test_mask, "mat")
        
        red_pixel_processed = processed_image.getpixel((50, 50))
        blue_pixel_processed = processed_image.getpixel((128, 128))
        print(f"   预处理后红色区域: RGB{red_pixel_processed}")
        print(f"   预处理后蓝色区域: RGB{blue_pixel_processed}")
        print(f"   预处理图像模式: {processed_image.mode}")
        print(f"   预处理Mask模式: {processed_mask.mode}")
        
    except Exception as e:
        print(f"   ❌ UI预处理失败: {e}")
        return False
    
    # Step 3: IOPaint数组准备
    print("\n🔧 Step 3: IOPaint数组准备")
    try:
        image_array, mask_array = ImageUtils.prepare_arrays_for_iopaint(processed_image, processed_mask)
        
        red_pixel_array = image_array[50, 50]     # 注意数组索引顺序
        blue_pixel_array = image_array[128, 128]
        
        print(f"   数组形状: {image_array.shape}, dtype: {image_array.dtype}")
        print(f"   数组红色区域: RGB({red_pixel_array[0]}, {red_pixel_array[1]}, {red_pixel_array[2]})")
        print(f"   数组蓝色区域: RGB({blue_pixel_array[0]}, {blue_pixel_array[1]}, {blue_pixel_array[2]})")
        print(f"   Mask数组形状: {mask_array.shape}, dtype: {mask_array.dtype}")
        
    except Exception as e:
        print(f"   ❌ IOPaint数组准备失败: {e}")
        return False
    
    # Step 4: 颜色处理器检查
    print("\n🎨 Step 4: 颜色处理器检查")
    try:
        from core.utils.color_utils import ColorSpaceProcessor, ModelColorConfig
        
        # 测试不同模型的颜色处理
        for model_name in ['lama', 'mat', 'zits', 'fcf']:
            print(f"\n   📋 {model_name.upper()} 模型颜色处理:")
            
            # 获取模型配置
            config = ModelColorConfig.get_model_config(model_name)
            print(f"      配置: {config}")
            
            # 输入预处理
            processed_input = ColorSpaceProcessor.prepare_image_for_model(image_array, model_name)
            input_red = processed_input[50, 50]
            input_blue = processed_input[128, 128]
            print(f"      输入处理后红色: RGB({input_red[0]}, {input_red[1]}, {input_red[2]})")
            print(f"      输入处理后蓝色: RGB({input_blue[0]}, {input_blue[1]}, {input_blue[2]})")
            
            # 输出后处理
            processed_output = ColorSpaceProcessor.process_output_for_display(processed_input, model_name)
            output_red = processed_output[50, 50]
            output_blue = processed_output[128, 128]
            print(f"      输出处理后红色: RGB({output_red[0]}, {output_red[1]}, {output_red[2]})")
            print(f"      输出处理后蓝色: RGB({output_blue[0]}, {output_blue[1]}, {output_blue[2]})")
            
            # 检查是否有颜色变化
            if not np.array_equal(processed_input, processed_output):
                print(f"      ⚠️ {model_name.upper()} 输入输出不一致！")
            else:
                print(f"      ✅ {model_name.upper()} 输入输出一致")
    
    except Exception as e:
        print(f"   ❌ 颜色处理器检查失败: {e}")
    
    # Step 5: 实际IOPaint调用
    print("\n🚀 Step 5: 实际IOPaint调用测试")
    try:
        from config.config import ConfigManager
        from core.models.unified_processor import UnifiedProcessor
        
        config_manager = ConfigManager()
        config = config_manager.get_config()
        processor = UnifiedProcessor(config)
        
        # 测试MAT模型（最容易看出颜色差异）
        print("   测试MAT模型的完整处理...")
        
        # 切换到MAT模型
        success = processor.switch_model('mat')
        if not success:
            print("   ❌ MAT模型切换失败")
            return False
        
        # 执行处理
        inpaint_params = {
            'model_name': 'mat',
            'ldm_steps': 10,  # 减少步数快速测试
            'hd_strategy': 'ORIGINAL'
        }
        
        result_array = processor.predict_with_model(processed_image, processed_mask, inpaint_params)
        
        print(f"   IOPaint结果形状: {result_array.shape}, dtype: {result_array.dtype}")
        
        # 检查处理过的区域（mask区域）
        mask_center_result = result_array[128, 128]  # 中心被处理的区域
        unmask_result = result_array[50, 50]         # 未被处理的区域
        
        print(f"   处理区域中心: RGB({mask_center_result[0]}, {mask_center_result[1]}, {mask_center_result[2]})")
        print(f"   未处理区域: RGB({unmask_result[0]}, {unmask_result[1]}, {unmask_result[2]})")
        
        # 检查未处理区域是否保持原色（注意：image_array现在是BGR格式）
        original_rgb = test_image.getpixel((50, 50))  # 使用原始RGB图像
        if np.allclose(unmask_result, original_rgb, atol=5):
            print("   ✅ 未处理区域颜色保持正确")
        else:
            print(f"   ⚠️ 未处理区域与原始RGB略有差异，这可能是正常的处理误差")
        
        processor.cleanup_resources()
        
    except Exception as e:
        print(f"   ❌ IOPaint调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: 结果转换到PIL
    print("\n🖼️ Step 6: 结果转换到PIL Image")
    try:
        # 模拟后处理过程
        from core.utils.image_utils import ImageUtils
        
        result_image = ImageUtils.postprocess_result(result_array, 'mat', (256, 256))
        
        print(f"   结果图像模式: {result_image.mode}")
        print(f"   结果图像尺寸: {result_image.size}")
        
        # 检查最终像素值
        final_mask_center = result_image.getpixel((128, 128))
        final_unmask = result_image.getpixel((50, 50))
        
        print(f"   最终处理区域中心: RGB{final_mask_center}")
        print(f"   最终未处理区域: RGB{final_unmask}")
        
        # 与原始输入对比
        original_unmask = test_image.getpixel((50, 50))
        print(f"   原始未处理区域: RGB{original_unmask}")
        
        if final_unmask == original_unmask:
            print("   ✅ 最终结果与原始输入颜色一致")
        else:
            print("   ❌ 最终结果与原始输入颜色不一致！")
            
            # 详细分析颜色变化
            r_diff = final_unmask[0] - original_unmask[0]
            g_diff = final_unmask[1] - original_unmask[1] 
            b_diff = final_unmask[2] - original_unmask[2]
            print(f"   颜色差异: R({r_diff}), G({g_diff}), B({b_diff})")
            
            if r_diff > 0 and b_diff < 0:
                print("   🚨 检测到红蓝通道交换！")
            elif abs(r_diff) == abs(b_diff) and r_diff * b_diff < 0:
                print("   🚨 明确的红蓝通道交换模式！")
        
    except Exception as e:
        print(f"   ❌ 结果转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def check_opencv_usage():
    """检查项目中OpenCV的使用情况"""
    print("\n🔍 检查OpenCV使用情况")
    print("=" * 40)
    
    try:
        import cv2
        print(f"✅ OpenCV版本: {cv2.__version__}")
        
        # 检查OpenCV默认颜色格式
        test_array = np.array([[[255, 50, 50]]], dtype=np.uint8)  # RGB
        bgr_array = cv2.cvtColor(test_array, cv2.COLOR_RGB2BGR)
        rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        
        print(f"原始RGB: {test_array[0,0]}")
        print(f"转BGR后: {bgr_array[0,0]}")
        print(f"转回RGB: {rgb_array[0,0]}")
        
        if np.array_equal(test_array[0,0], rgb_array[0,0]):
            print("✅ OpenCV颜色转换正常")
        else:
            print("❌ OpenCV颜色转换异常")
            
    except ImportError:
        print("⚠️ OpenCV未安装")

def analyze_streamlit_display():
    """分析Streamlit显示相关的代码"""
    print("\n📺 检查Streamlit显示处理")
    print("=" * 40)
    
    try:
        # 查找UI显示相关代码
        from interfaces.web.ui import MainInterface
        print("✅ 找到MainInterface")
        
        # 检查是否有图像显示相关的转换
        import inspect
        source = inspect.getsource(MainInterface)
        
        if 'cv2.cvtColor' in source:
            print("⚠️ UI代码中发现cv2.cvtColor调用")
        if 'COLOR_RGB2BGR' in source or 'COLOR_BGR2RGB' in source:
            print("⚠️ UI代码中发现颜色格式转换")
        
        print("✅ Streamlit显示检查完成")
        
    except Exception as e:
        print(f"❌ Streamlit检查失败: {e}")

def main():
    """主分析函数"""
    print("🔬 完整颜色通道管道分析")
    print("=" * 80)
    
    # 检查每一步的颜色处理
    success = analyze_color_at_each_step()
    
    # 检查OpenCV使用
    check_opencv_usage()
    
    # 检查Streamlit显示
    analyze_streamlit_display()
    
    print("\n" + "=" * 80)
    if success:
        print("📊 分析完成！请查看上述详细输出定位问题。")
    else:
        print("❌ 分析过程中遇到错误，请检查日志。")

if __name__ == "__main__":
    main()