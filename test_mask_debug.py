#!/usr/bin/env python3
"""
测试 mask 处理逻辑的调试脚本
"""

import numpy as np
from PIL import Image
import cv2

def test_mask_processing():
    """测试 mask 处理逻辑"""
    print("🧪 测试 mask 处理逻辑...")
    
    # 创建一个测试图像和 mask
    image_size = (2000, 1500)
    mask_size = (2000, 1500)
    
    # 创建测试图像
    test_image = Image.new('RGB', image_size, color='white')
    print(f"📏 测试图像尺寸: {test_image.size}")
    
    # 创建测试 mask - 模拟水印区域
    test_mask = Image.new('L', mask_size, color=0)  # 黑色背景
    
    # 在 mask 中心添加一个白色区域（模拟水印）
    mask_array = np.array(test_mask)
    center_x, center_y = mask_size[0] // 2, mask_size[1] // 2
    mask_array[center_y-100:center_y+100, center_x-200:center_x+200] = 255  # 白色水印区域
    test_mask = Image.fromarray(mask_array, mode='L')
    
    print(f"📏 测试 mask 尺寸: {test_mask.size}")
    print(f"🎨 测试 mask 模式: {test_mask.mode}")
    
    # 检查 mask 内容
    mask_array = np.array(test_mask)
    white_pixels = np.sum(mask_array > 128)
    total_pixels = mask_array.size
    mask_coverage = white_pixels / total_pixels * 100
    print(f"🔍 Mask 内容分析: 白色像素={white_pixels}, 总像素={total_pixels}, 覆盖率={mask_coverage:.2f}%")
    print(f"📊 Mask 像素值范围: {mask_array.min()} - {mask_array.max()}")
    
    # 测试轮廓检测
    print("\n🔍 测试轮廓检测...")
    binary_mask = (mask_array > 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"🎯 找到轮廓数量: {len(contours)}")
    
    if contours:
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            print(f"📦 轮廓 {i+1}: 边界框=({x}, {y}, {w}, {h})")
    
    # 测试 crop 策略
    print("\n🔍 测试 crop 策略...")
    crop_size = 512
    margin = 64
    
    # 模拟 _find_mask_regions 逻辑
    boxes = []
    height, width = binary_mask.shape
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expand box with margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + w + margin)
        y2 = min(height, y + h + margin)
        
        # Ensure minimum crop size
        if x2 - x1 < crop_size or y2 - y1 < crop_size:
            # Center the crop around the contour
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            half_size = crop_size // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(width, x1 + crop_size)
            y2 = min(height, y1 + crop_size)
            
            # Adjust if we hit boundaries
            if x2 - x1 < crop_size:
                x1 = max(0, x2 - crop_size)
            if y2 - y1 < crop_size:
                y1 = max(0, y2 - crop_size)
        
        boxes.append((x1, y1, x2, y2))
        print(f"📐 边界框 {i+1}: ({x1}, {y1}, {x2}, {y2})")
    
    print(f"🔗 总边界框数量: {len(boxes)}")
    
    # 测试 crop 区域
    if boxes:
        x1, y1, x2, y2 = boxes[0]
        crop_image = test_image.crop((x1, y1, x2, y2))
        crop_mask = test_mask.crop((x1, y1, x2, y2))
        print(f"📦 Crop 区域尺寸: {crop_image.size}")
        print(f"📦 Crop mask 尺寸: {crop_mask.size}")
        
        # 检查 crop mask 内容
        crop_mask_array = np.array(crop_mask)
        crop_white_pixels = np.sum(crop_mask_array > 128)
        crop_total_pixels = crop_mask_array.size
        crop_coverage = crop_white_pixels / crop_total_pixels * 100
        print(f"🔍 Crop mask 覆盖率: {crop_coverage:.2f}%")

if __name__ == "__main__":
    test_mask_processing() 