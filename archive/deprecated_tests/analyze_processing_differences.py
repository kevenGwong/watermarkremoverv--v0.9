#!/usr/bin/env python3
"""
分析test_iopaint.py与UI程序处理差异
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from web_backend import WatermarkProcessor

def analyze_test_iopaint_method():
    """分析test_iopaint.py的处理方法"""
    print("🔬 分析 test_iopaint.py 的处理方法")
    print("="*60)
    
    print("📋 test_iopaint.py 的关键特点:")
    print("1. 直接使用cv2.imread()读取图像 (BGR格式)")
    print("2. 直接使用cv2.imread()读取mask (灰度)")
    print("3. 使用固定的膨胀参数: kernel_size=5, iterations=1")
    print("4. 使用固定的LaMA配置:")
    print("   - ldm_steps=50")
    print("   - ldm_sampler=ddim") 
    print("   - hd_strategy=CROP")
    print("   - hd_strategy_crop_margin=64")
    print("   - hd_strategy_crop_trigger_size=800")
    print("   - hd_strategy_resize_limit=1600")
    print("5. 直接调用 model_manager(image, mask, config)")
    print("6. 结果直接用cv2.imwrite()保存")
    
    print("\n🎯 关键点: 使用预制的高质量mask文件!")
    print("   mask_path = '/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/mask/IMG_0095-4_dilated_mask.png'")

def analyze_ui_processing_method():
    """分析UI程序的处理方法"""
    print("\n🔬 分析 UI程序 的处理方法")
    print("="*60)
    
    print("📋 UI程序的处理流程:")
    print("1. 图像格式转换: PIL Image -> RGB")
    print("2. mask生成过程:")
    print("   a. Custom模型: FPN+MIT-B5推理生成mask")
    print("   b. Florence-2模型: 目标检测生成bbox")
    print("3. mask后处理:")
    print("   - 阈值二值化 (threshold=0.5)")
    print("   - 形态学膨胀 (kernel_size=3, iterations=1)")
    print("4. 图像转换链:")
    print("   PIL Image -> numpy (RGB) -> LaMA处理 -> numpy -> PIL Image")
    print("5. 颜色空间转换: cv2.cvtColor(result, cv2.COLOR_BGR2RGB)")
    
    print("\n🎯 关键差异: UI程序需要AI生成mask，而test_iopaint.py使用预制mask!")

def create_test_comparison():
    """创建对比测试"""
    print("\n🧪 创建对比测试")
    print("="*60)
    
    try:
        # 测试1: 使用test_iopaint.py的方法
        print("\n📋 测试1: 复现test_iopaint.py的处理方法")
        
        input_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/input/IMG_0001-3.jpg"
        mask_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/mask/IMG_0095-4_dilated_mask.png"
        
        if not Path(input_path).exists() or not Path(mask_path).exists():
            print("❌ 测试文件不存在，跳过对比测试")
            return
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 方法1: test_iopaint.py的方式
        print("\n🔧 方法1: test_iopaint.py方式")
        
        model_manager = ModelManager(name="lama", device=device)
        
        # 读取 (重要: cv2直接读取)
        image_cv = cv2.imread(input_path)
        mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"原始图像shape: {image_cv.shape}")
        print(f"原始mask shape: {mask_cv.shape}")
        print(f"Mask dtype: {mask_cv.dtype}")
        print(f"Mask 值范围: {mask_cv.min()} - {mask_cv.max()}")
        
        # 膨胀处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask_cv = cv2.dilate(mask_cv, kernel, iterations=1)
        
        # LaMA配置
        config = Config(
            ldm_steps=50,
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.CROP,
            hd_strategy_crop_margin=64,
            hd_strategy_crop_trigger_size=800,
            hd_strategy_resize_limit=1600,
        )
        
        # 处理
        result1 = model_manager(image_cv, dilated_mask_cv, config)
        
        if result1.dtype in [np.float64, np.float32]:
            result1 = np.clip(result1, 0, 255).astype(np.uint8)
        
        # 保存结果1
        cv2.imwrite("comparison_method1_test_iopaint.png", result1)
        cv2.imwrite("comparison_mask1_test_iopaint.png", dilated_mask_cv)
        
        print("✅ 方法1完成: comparison_method1_test_iopaint.png")
        
        # 方法2: UI程序的方式
        print("\n🔧 方法2: UI程序方式")
        
        # 使用UI的处理器
        processor = WatermarkProcessor("web_config.yaml")
        
        # PIL方式读取
        image_pil = Image.open(input_path)
        mask_pil = Image.open(mask_path).convert('L')
        
        print(f"PIL图像模式: {image_pil.mode}, 尺寸: {image_pil.size}")
        print(f"PIL mask模式: {mask_pil.mode}, 尺寸: {mask_pil.size}")
        
        # 转换为numpy (UI程序的方式)
        image_np = np.array(image_pil.convert("RGB"))
        mask_np = np.array(mask_pil.convert("L"))
        
        print(f"转换后图像shape: {image_np.shape}")
        print(f"转换后mask shape: {mask_np.shape}")
        print(f"转换后mask 值范围: {mask_np.min()} - {mask_np.max()}")
        
        # 使用UI的LaMA处理
        result2 = processor.model_manager(image_np, mask_np, config)
        
        if result2.dtype in [np.float64, np.float32]:
            result2 = np.clip(result2, 0, 255).astype(np.uint8)
        
        # 颜色空间转换 (UI程序的方式)
        result2_rgb = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)
        
        # 保存结果2
        result2_pil = Image.fromarray(result2_rgb)
        result2_pil.save("comparison_method2_ui_program.png")
        
        # 也保存numpy版本用于对比
        cv2.imwrite("comparison_method2_ui_program_raw.png", result2)
        
        print("✅ 方法2完成: comparison_method2_ui_program.png")
        
        # 分析差异
        print("\n📊 结果分析:")
        print("1. comparison_method1_test_iopaint.png - test_iopaint.py方式")
        print("2. comparison_method2_ui_program.png - UI程序方式")
        print("3. comparison_method2_ui_program_raw.png - UI程序原始结果")
        
        # 计算像素差异
        diff = cv2.absdiff(result1, result2)
        diff_mean = np.mean(diff)
        print(f"\n📈 平均像素差异: {diff_mean:.2f}")
        
        if diff_mean > 1.0:
            print("⚠️  存在显著差异!")
            cv2.imwrite("comparison_difference.png", diff)
            print("差异图保存为: comparison_difference.png")
        else:
            print("✅ 结果基本一致")
        
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()

def analyze_mask_quality():
    """分析mask质量差异"""
    print("\n🎭 分析mask质量差异")
    print("="*60)
    
    mask_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/mask/IMG_0095-4_dilated_mask.png"
    
    if not Path(mask_path).exists():
        print("❌ 预制mask文件不存在")
        return
    
    # 读取预制mask
    mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_pil = Image.open(mask_path).convert('L')
    
    print("📊 预制mask分析:")
    print(f"尺寸: {mask_cv.shape}")
    print(f"数据类型: {mask_cv.dtype}")
    print(f"值范围: {mask_cv.min()} - {mask_cv.max()}")
    print(f"唯一值: {np.unique(mask_cv)}")
    
    # 统计白色像素
    white_pixels = np.sum(mask_cv > 128)
    total_pixels = mask_cv.size
    coverage = (white_pixels / total_pixels) * 100
    
    print(f"白色像素数: {white_pixels}")
    print(f"总像素数: {total_pixels}")
    print(f"覆盖率: {coverage:.2f}%")
    
    # 分析mask形状
    contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"检测到轮廓数: {len(contours)}")
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        print(f"最大轮廓面积: {area:.0f}")
        
        # 保存轮廓可视化
        contour_img = cv2.cvtColor(mask_cv, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 2)
        cv2.imwrite("mask_contour_analysis.png", contour_img)
        print("轮廓分析保存为: mask_contour_analysis.png")

def main():
    """主函数"""
    print("🔍 分析 test_iopaint.py 与 UI程序 处理差异")
    print("="*80)
    
    analyze_test_iopaint_method()
    analyze_ui_processing_method()
    analyze_mask_quality()
    create_test_comparison()
    
    print("\n" + "="*80)
    print("🎯 总结分析")
    print("="*80)
    
    print("💡 主要差异:")
    print("1. 📁 输入源: test_iopaint.py使用预制的高质量mask")
    print("2. 🎯 mask生成: UI程序需要AI实时生成mask")
    print("3. 🖼️  图像格式: cv2 vs PIL的处理链差异")
    print("4. 🔄 颜色空间: BGR/RGB转换可能影响结果")
    
    print("\n🔧 改进建议:")
    print("1. 检查AI生成的mask质量是否足够")
    print("2. 对比预制mask与AI生成mask的差异")
    print("3. 优化mask后处理参数")
    print("4. 检查图像格式转换是否正确")
    
    print("\n📁 生成的对比文件:")
    print("- comparison_method1_test_iopaint.png")
    print("- comparison_method2_ui_program.png") 
    print("- comparison_difference.png (如果有差异)")
    print("- mask_contour_analysis.png")

if __name__ == "__main__":
    main()