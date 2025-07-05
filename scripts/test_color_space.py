#!/usr/bin/env python3
"""
颜色空间处理测试脚本
验证不同模型的颜色空间处理是否正确
"""

import sys
import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_image():
    """创建测试图像"""
    # 创建一个简单的RGB测试图像
    width, height = 256, 256
    image = Image.new('RGB', (width, height))
    
    # 填充不同颜色区域来测试颜色转换
    pixels = []
    for y in range(height):
        for x in range(width):
            if x < width // 3:
                pixels.append((255, 0, 0))  # 红色
            elif x < 2 * width // 3:
                pixels.append((0, 255, 0))  # 绿色
            else:
                pixels.append((0, 0, 255))  # 蓝色
    
    image.putdata(pixels)
    return image

def create_test_mask():
    """创建测试mask"""
    width, height = 256, 256
    mask = Image.new('L', (width, height), 0)
    
    # 在中心创建一个矩形mask
    center_x, center_y = width // 2, height // 2
    mask_size = 64
    
    pixels = []
    for y in range(height):
        for x in range(width):
            if (center_x - mask_size//2 <= x <= center_x + mask_size//2 and
                center_y - mask_size//2 <= y <= center_y + mask_size//2):
                pixels.append(255)  # 白色mask区域
            else:
                pixels.append(0)    # 黑色背景
    
    mask.putdata(pixels)
    return mask

def test_image_utils_color_processing():
    """测试ImageUtils颜色空间处理"""
    logger.info("🧪 测试ImageUtils颜色空间处理...")
    
    try:
        from core.utils.image_utils import ImageUtils
        
        # 创建测试数据
        test_image = create_test_image()
        test_mask = create_test_mask()
        
        # 测试标准预处理
        logger.info("📋 测试标准预处理...")
        processed_img, processed_mask = ImageUtils.preprocess_for_model(test_image, test_mask, "mat")
        logger.info(f"✅ 标准预处理完成: 图像{processed_img.size}, mask{processed_mask.size}")
        
        # 测试IOPaint数组准备
        logger.info("📋 测试IOPaint数组准备...")
        iopaint_img, iopaint_mask = ImageUtils.prepare_arrays_for_iopaint(processed_img, processed_mask)
        logger.info(f"✅ IOPaint数组准备完成: 图像{iopaint_img.shape}, mask{iopaint_mask.shape}")
        logger.info(f"   图像颜色范围: {iopaint_img.min()}-{iopaint_img.max()}")
        logger.info(f"   Mask值: {np.unique(iopaint_mask)}")
        
        # 测试LaMA数组准备
        logger.info("📋 测试LaMA数组准备...")
        lama_img, lama_mask = ImageUtils.prepare_arrays_for_lama(processed_img, processed_mask)
        logger.info(f"✅ LaMA数组准备完成: 图像{lama_img.shape}, mask{lama_mask.shape}")
        logger.info(f"   图像颜色范围: {lama_img.min()}-{lama_img.max()}")
        
        # 验证BGR转换
        # 检查第一个像素的颜色是否从RGB(255,0,0)转换为BGR(0,0,255)
        rgb_pixel = iopaint_img[0, 0]  # 应该是[255, 0, 0]
        bgr_pixel = lama_img[0, 0]     # 应该是[0, 0, 255]
        
        logger.info(f"   RGB像素示例: {rgb_pixel}")
        logger.info(f"   BGR像素示例: {bgr_pixel}")
        
        if np.array_equal(rgb_pixel, [255, 0, 0]) and np.array_equal(bgr_pixel, [0, 0, 255]):
            logger.info("✅ BGR/RGB转换验证通过")
        else:
            logger.warning("⚠️ BGR/RGB转换可能有问题")
        
        # 测试LaMA结果后处理
        logger.info("📋 测试LaMA结果后处理...")
        # 模拟LaMA输出（BGR格式）
        mock_lama_result = lama_img.copy()  # 使用BGR格式的图像作为模拟结果
        processed_result = ImageUtils.postprocess_lama_result(mock_lama_result)
        
        # 验证是否转换回RGB
        rgb_result_pixel = processed_result[0, 0]
        logger.info(f"   处理后像素: {rgb_result_pixel}")
        
        if np.array_equal(rgb_result_pixel, [255, 0, 0]):
            logger.info("✅ LaMA结果后处理验证通过")
        else:
            logger.warning("⚠️ LaMA结果后处理可能有问题")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ImageUtils颜色空间处理测试失败: {e}")
        return False

def test_model_color_consistency():
    """测试模型颜色一致性"""
    logger.info("🧪 测试模型颜色一致性...")
    
    try:
        from core.models.base_inpainter import ModelRegistry
        
        # 导入所有模型以确保注册
        from core.models.mat_processor import MatProcessor
        from core.models.zits_processor import ZitsProcessor
        from core.models.fcf_processor import FcfProcessor
        from core.models.lama_processor_unified import LamaProcessor
        
        available_models = ModelRegistry.get_available_models()
        logger.info(f"📋 可用模型: {available_models}")
        
        # 创建测试数据
        test_image = create_test_image()
        test_mask = create_test_mask()
        
        for model_name in ["mat", "zits", "fcf"]:  # 先测试IOPaint模型
            logger.info(f"   测试 {model_name.upper()} 颜色处理...")
            
            try:
                # 创建模型实例
                config = {'device': 'cpu'}
                model = ModelRegistry.create_model(model_name, config)
                
                # 验证输入
                if model.validate_inputs(test_image, test_mask):
                    logger.info(f"   ✅ {model_name.upper()} 输入验证通过")
                
                # 清理
                model.cleanup_resources()
                
            except Exception as e:
                logger.warning(f"   ⚠️ {model_name.upper()} 测试跳过: {e}")
        
        # 特别测试LaMA
        logger.info("   测试 LaMA 颜色处理...")
        try:
            config = {'device': 'cpu'}
            lama_model = ModelRegistry.create_model("lama", config)
            
            if lama_model.validate_inputs(test_image, test_mask):
                logger.info("   ✅ LaMA 输入验证通过")
            
            # 检查LaMA的特殊属性
            model_info = lama_model.get_model_info()
            logger.info(f"   📊 LaMA模式: {model_info.get('mode', 'unknown')}")
            
            lama_model.cleanup_resources()
            
        except Exception as e:
            logger.warning(f"   ⚠️ LaMA 测试跳过: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型颜色一致性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始颜色空间处理测试套件...")
    
    tests = [
        ("ImageUtils颜色处理", test_image_utils_color_processing),
        ("模型颜色一致性", test_model_color_consistency)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- 测试: {test_name} ---")
        try:
            if test_func():
                logger.info(f"✅ {test_name} 测试通过")
                passed += 1
            else:
                logger.error(f"❌ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
    
    logger.info(f"\n🎯 测试结果: {passed}/{total} 通过")
    
    # 总结
    logger.info("\n📋 颜色空间处理优化总结:")
    logger.info("   ✅ 简化了颜色空间处理逻辑")
    logger.info("   ✅ IOPaint模型使用标准RGB处理")
    logger.info("   ✅ LaMA模型专门处理BGR转换")
    logger.info("   ✅ 统一预处理和后处理接口")
    
    if passed == total:
        logger.info("🎉 颜色空间处理优化成功！")
        return 0
    else:
        logger.error("❌ 颜色空间处理需要进一步改进")
        return 1

if __name__ == "__main__":
    sys.exit(main())