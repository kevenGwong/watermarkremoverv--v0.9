#!/usr/bin/env python3
"""
分析mask质量和修复效果的脚本
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

def analyze_mask(mask_path, name="Mask"):
    """分析mask质量"""
    print(f"\n🔍 分析{name}: {mask_path}")
    
    try:
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        print(f"   尺寸: {mask.size}")
        print(f"   模式: {mask.mode}")
        print(f"   数据类型: {mask_array.dtype}")
        print(f"   数值范围: {mask_array.min()} - {mask_array.max()}")
        
        # 计算覆盖率
        white_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        coverage = (white_pixels / total_pixels) * 100
        
        print(f"   白色像素: {white_pixels}")
        print(f"   总像素: {total_pixels}")
        print(f"   覆盖率: {coverage:.2f}%")
        
        # 分析连通区域
        if mask.mode == 'L':
            # 二值化
            binary = (mask_array > 128).astype(np.uint8)
            
            # 查找连通区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            print(f"   连通区域数量: {num_labels - 1}")  # 减去背景
            
            if num_labels > 1:
                # 分析每个连通区域
                for i in range(1, num_labels):  # 跳过背景
                    area = stats[i, cv2.CC_STAT_AREA]
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    
                    print(f"     区域{i}: 面积={area}, 位置=({x},{y}), 尺寸={w}x{h}")
        
        return coverage
        
    except Exception as e:
        print(f"   ❌ 分析失败: {e}")
        return 0

def compare_images(original_path, result_path, name="结果"):
    """比较原始图像和结果图像"""
    print(f"\n🖼️  分析{name}: {result_path}")
    
    try:
        original = Image.open(original_path)
        result = Image.open(result_path)
        
        print(f"   原始图像: {original.size}, {original.mode}")
        print(f"   结果图像: {result.size}, {result.mode}")
        
        # 转换为numpy数组进行比较
        orig_array = np.array(original.convert('RGB'))
        result_array = np.array(result.convert('RGB'))
        
        # 计算差异
        diff = np.abs(orig_array.astype(np.float32) - result_array.astype(np.float32))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        print(f"   平均差异: {mean_diff:.2f}")
        print(f"   最大差异: {max_diff:.2f}")
        
        # 计算PSNR (Peak Signal-to-Noise Ratio)
        mse = np.mean((orig_array.astype(np.float32) - result_array.astype(np.float32)) ** 2)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            print(f"   PSNR: {psnr:.2f} dB")
        
        # 检查是否有明显变化
        if mean_diff < 5:
            print("   ⚠️  警告: 图像变化很小，可能修复效果不明显")
        elif mean_diff > 50:
            print("   ✅ 图像有明显变化，修复效果显著")
        else:
            print("   ✅ 图像有适度变化，修复效果正常")
            
    except Exception as e:
        print(f"   ❌ 比较失败: {e}")

def analyze_original_mask():
    """分析原始mask文件"""
    original_mask_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/scripts/完美1500*2000mask透明通道.png"
    
    if Path(original_mask_path).exists():
        print("\n📸 分析原始mask文件...")
        analyze_mask(original_mask_path, "原始Mask")
    else:
        print(f"\n❌ 原始mask文件不存在: {original_mask_path}")

def main():
    """主分析函数"""
    print("🔬 Mask质量和修复效果分析")
    print("=" * 60)
    
    # 分析原始mask
    analyze_original_mask()
    
    # 分析生成的mask
    test_masks = [
        "test_mask_passing_mask.png",
        "test_enhanced_mask.png"
    ]
    
    for mask_path in test_masks:
        if Path(mask_path).exists():
            analyze_mask(mask_path, f"生成Mask ({mask_path})")
        else:
            print(f"\n❌ Mask文件不存在: {mask_path}")
    
    # 分析结果图像
    original_image_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/scripts/IMG_0001-3.jpg"
    
    test_results = [
        "test_mask_passing_result.png",
        "test_enhanced_result.png"
    ]
    
    for result_path in test_results:
        if Path(result_path).exists():
            compare_images(original_image_path, result_path, f"修复结果 ({result_path})")
        else:
            print(f"\n❌ 结果文件不存在: {result_path}")
    
    print("\n" + "=" * 60)
    print("📊 分析完成")

if __name__ == "__main__":
    main() 