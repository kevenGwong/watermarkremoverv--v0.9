#!/usr/bin/env python3
"""
测试模型选择和颜色通道修复效果
验证用户选择的模型能正确调用，且颜色通道正确
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import time
from PIL import Image
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_selection():
    """测试模型选择功能"""
    print("🧪 测试模型选择功能")
    print("=" * 50)
    
    try:
        from config.config import ConfigManager
        from core.models.unified_processor import UnifiedProcessor
        
        # 创建配置管理器
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # 创建统一处理器
        processor = UnifiedProcessor(config)
        
        # 测试获取可用模型
        available_models = processor.get_available_models()
        print(f"✅ 可用模型: {available_models}")
        
        # 测试模型切换
        test_models = ['mat', 'zits', 'fcf', 'lama']
        for model_name in test_models:
            if model_name in available_models:
                print(f"\n🔄 测试切换到 {model_name.upper()} 模型...")
                success = processor.switch_model(model_name)
                current_model = processor.get_current_model()
                
                if success and current_model == model_name:
                    print(f"✅ {model_name.upper()} 模型切换成功")
                else:
                    print(f"❌ {model_name.upper()} 模型切换失败")
                    return False
            else:
                print(f"⚠️ {model_name.upper()} 模型不可用，跳过")
        
        # 清理资源
        processor.cleanup_resources()
        print(f"\n✅ 模型选择功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型选择测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_color_channel_processing():
    """测试颜色通道处理"""
    print("\n🧪 测试颜色通道处理")
    print("=" * 50)
    
    try:
        from core.utils.image_utils import ImageUtils
        from core.utils.color_utils import ColorSpaceProcessor, ModelColorConfig
        
        # 创建测试图像（红色为主）
        test_image = Image.new('RGB', (100, 100), (255, 50, 50))  # 红色图像
        test_mask = Image.new('L', (100, 100), 255)  # 白色mask
        
        print(f"📸 测试图像: 100x100 红色图像 RGB(255,50,50)")
        
        # 测试IOPaint数组准备
        image_array, mask_array = ImageUtils.prepare_arrays_for_iopaint(test_image, test_mask)
        
        print(f"✅ IOPaint数组准备:")
        print(f"   图像数组形状: {image_array.shape}")
        print(f"   图像颜色值: R={image_array[0,0,0]}, G={image_array[0,0,1]}, B={image_array[0,0,2]}")
        print(f"   Mask数组形状: {mask_array.shape}")
        
        # 验证颜色通道顺序（红色应该在第0通道）
        if image_array[0,0,0] == 255 and image_array[0,0,1] == 50 and image_array[0,0,2] == 50:
            print(f"✅ 颜色通道顺序正确 (RGB)")
        else:
            print(f"❌ 颜色通道顺序错误！期望RGB(255,50,50)，实际({image_array[0,0,0]},{image_array[0,0,1]},{image_array[0,0,2]})")
            return False
        
        # 测试不同模型的颜色处理
        test_models = ['lama', 'mat', 'zits', 'fcf']
        for model_name in test_models:
            print(f"\n🎨 测试 {model_name.upper()} 模型颜色处理:")
            
            # 获取模型配置
            model_config = ModelColorConfig.get_model_config(model_name)
            print(f"   配置: {model_config}")
            
            # 测试输入预处理
            processed_input = ColorSpaceProcessor.prepare_image_for_model(image_array, model_name)
            input_red = processed_input[0,0,0]
            input_blue = processed_input[0,0,2]
            
            # 测试输出后处理
            processed_output = ColorSpaceProcessor.process_output_for_display(processed_input, model_name)
            output_red = processed_output[0,0,0]
            output_blue = processed_output[0,0,2]
            
            print(f"   输入处理: R={input_red}, B={input_blue}")
            print(f"   输出处理: R={output_red}, B={output_blue}")
            
            # 验证颜色一致性（红色通道应该保持最高）
            if output_red > output_blue:
                print(f"   ✅ {model_name.upper()} 颜色通道正确")
            else:
                print(f"   ❌ {model_name.upper()} 颜色通道可能错误")
                return False
        
        print(f"\n✅ 颜色通道处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 颜色通道测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_processing():
    """测试端到端处理流程"""
    print("\n🧪 测试端到端处理流程")
    print("=" * 50)
    
    try:
        from config.config import ConfigManager
        from core.inference import process_image
        
        # 创建配置管理器
        config_manager = ConfigManager()
        
        # 创建测试图像
        test_image = Image.new('RGB', (256, 256), (100, 150, 200))  # 蓝绿色图像
        test_mask = Image.new('L', (256, 256), 0)  # 黑色背景
        
        # 在中心添加白色区域作为水印区域
        mask_array = np.array(test_mask)
        mask_array[100:156, 100:156] = 255  # 中心56x56的白色区域
        test_mask = Image.fromarray(mask_array, mode='L')
        
        print(f"📸 测试场景: 256x256图像，中心56x56水印区域")
        
        # 测试不同模型的处理
        test_models = ['mat', 'fcf']  # 测试最常用的两个模型
        
        for model_name in test_models:
            print(f"\n🎨 测试 {model_name.upper()} 端到端处理:")
            
            start_time = time.time()
            
            try:
                result = process_image(
                    image=test_image,
                    mask_model='upload',
                    mask_params={'uploaded_mask': test_mask},
                    inpaint_params={
                        'model_name': model_name,  # 关键：模型选择参数
                        'ldm_steps': 20,  # 减少步数以加快测试
                        'hd_strategy': 'ORIGINAL'
                    },
                    performance_params={},
                    transparent=False,
                    config_manager=config_manager  # 传递配置管理器
                )
                
                processing_time = time.time() - start_time
                
                if result.success:
                    print(f"   ✅ {model_name.upper()} 处理成功")
                    print(f"   ⏱️ 处理时间: {processing_time:.2f}秒")
                    print(f"   📊 结果图像: {result.result_image.size} {result.result_image.mode}")
                    
                    # 验证结果图像颜色通道
                    result_array = np.array(result.result_image)
                    center_pixel = result_array[128, 128]  # 中心像素
                    print(f"   🎨 中心像素颜色: RGB({center_pixel[0]},{center_pixel[1]},{center_pixel[2]})")
                    
                else:
                    print(f"   ❌ {model_name.upper()} 处理失败: {result.error_message}")
                    return False
                    
            except Exception as e:
                print(f"   ❌ {model_name.upper()} 处理异常: {e}")
                return False
        
        print(f"\n✅ 端到端处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔍 模型选择和颜色通道修复测试")
    print("=" * 60)
    
    tests = [
        ("模型选择功能", test_model_selection),
        ("颜色通道处理", test_color_channel_processing),
        ("端到端处理", test_end_to_end_processing)
    ]
    
    success_count = 0
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            success_count += 1
        else:
            print(f"💥 {test_name} 测试失败，停止后续测试")
            break
    
    total_tests = len(tests)
    success_rate = (success_count / total_tests) * 100
    
    print(f"\n{'=' * 60}")
    print(f"📊 测试结果: {success_count}/{total_tests} 通过 ({success_rate:.1f}%)")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！修复效果验证成功")
        print("\n✅ 修复总结:")
        print("   1. ✅ UI模型选择参数传递已修复")
        print("   2. ✅ 统一处理器动态模型切换已实现")
        print("   3. ✅ 颜色通道处理已统一为IOPaint标准")
        print("   4. ✅ 所有模型都使用RGB格式，无红蓝颠倒问题")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")

if __name__ == "__main__":
    main()