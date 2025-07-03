#!/usr/bin/env python3
"""
Mask Debug Test Script
测试mask传递和参数传递的完整流程
"""

import sys
import os
import numpy as np
from PIL import Image
import cv2
import logging

# 添加项目路径
sys.path.insert(0, '/home/duolaameng/SAM_Remove/WatermarkRemover-AI')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_mask_loading():
    """测试mask加载和验证"""
    
    # 测试图片路径
    image_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/input/IMG_0308-3.jpg"
    mask_path = "/home/duolaameng/SAM_Remove/Watermark_sam/mask/watermark_2000x1500.png"
    
    logger.info("=== Mask Loading Test ===")
    
    # 1. 加载图片
    if not os.path.exists(image_path):
        logger.error(f"测试图片不存在: {image_path}")
        return False
    
    image = Image.open(image_path).convert("RGB")
    logger.info(f"✅ 图片加载: size={image.size}, mode={image.mode}")
    
    # 2. 加载mask
    if not os.path.exists(mask_path):
        logger.error(f"测试mask不存在: {mask_path}")
        return False
    
    mask = Image.open(mask_path).convert("L")
    logger.info(f"✅ Mask加载: size={mask.size}, mode={mask.mode}")
    
    # 3. 检查尺寸匹配
    if mask.size != image.size:
        logger.info(f"📏 调整mask尺寸: {mask.size} → {image.size}")
        mask = mask.resize(image.size, Image.LANCZOS)
    
    # 4. 验证mask内容
    mask_array = np.array(mask)
    white_pixels = np.sum(mask_array > 128)
    total_pixels = mask_array.size
    coverage = white_pixels / total_pixels * 100
    
    logger.info(f"🔍 Mask验证:")
    logger.info(f"   总像素: {total_pixels}")
    logger.info(f"   白色像素: {white_pixels}")
    logger.info(f"   黑色像素: {total_pixels - white_pixels}")
    logger.info(f"   覆盖率: {coverage:.2f}%")
    logger.info(f"   值范围: min={mask_array.min()}, max={mask_array.max()}")
    
    if white_pixels == 0:
        logger.warning("⚠️ WARNING: Mask中没有白色像素！")
        return False
    
    # 5. 测试膨胀处理
    dilate_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    dilated_mask = cv2.dilate(mask_array, kernel, iterations=1)
    
    white_pixels_after = np.sum(dilated_mask > 128)
    coverage_after = white_pixels_after / total_pixels * 100
    
    logger.info(f"🔍 膨胀后验证:")
    logger.info(f"   白色像素: {white_pixels_after}")
    logger.info(f"   覆盖率: {coverage_after:.2f}%")
    
    return True

def test_backend_integration():
    """测试后端集成"""
    
    logger.info("=== Backend Integration Test ===")
    
    try:
        from web_backend import WatermarkProcessor
        
        # 初始化处理器
        processor = WatermarkProcessor("web_config.yaml")
        logger.info("✅ 处理器初始化成功")
        
        # 测试图片路径
        image_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/input/IMG_0308-3.jpg"
        
        # 测试处理
        result = processor.process_image(
            image=image_path,
            transparent=False,
            max_bbox_percent=10.0,
            force_format="PNG",
            custom_inpaint_config={
                'ldm_steps': 25,  # 减少步数快速测试
                'ldm_sampler': 'ddim',
                'hd_strategy': 'CROP'
            }
        )
        
        if result.success:
            logger.info("✅ 后端处理成功")
            logger.info(f"   结果图像: {result.result_image.size}")
            logger.info(f"   Mask图像: {result.mask_image.size}")
            logger.info(f"   处理时间: {result.processing_time:.2f}秒")
            return True
        else:
            logger.error(f"❌ 后端处理失败: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 后端集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    
    logger.info("🚀 开始Mask调试测试...")
    
    # 测试1: Mask加载
    if not test_mask_loading():
        logger.error("❌ Mask加载测试失败")
        return
    
    # 测试2: 后端集成
    if not test_backend_integration():
        logger.error("❌ 后端集成测试失败")
        return
    
    logger.info("✅ 所有测试通过！")

if __name__ == "__main__":
    main()