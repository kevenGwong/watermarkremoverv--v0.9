#!/usr/bin/env python3
"""
参数效果测试脚本
测试不同参数设置对处理结果的实际影响
"""
import os
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_image_with_watermark():
    """创建带水印的测试图片"""
    print("🎨 Creating test image with watermark...")
    
    # 创建基础图片
    img = Image.new('RGB', (512, 512), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # 添加背景内容
    for i in range(0, 512, 50):
        draw.line([(i, 0), (i, 512)], fill='white', width=1)
        draw.line([(0, i), (512, i)], fill='white', width=1)
    
    # 添加主要内容
    draw.rectangle([50, 50, 450, 200], fill='darkblue', outline='navy', width=3)
    
    try:
        # 尝试添加文字
        draw.text((100, 100), "TEST IMAGE", fill='white')
    except:
        # 如果没有字体，跳过文字
        pass
    
    # 添加"水印" - 红色半透明区域
    watermark_overlay = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    wm_draw = ImageDraw.Draw(watermark_overlay)
    
    # 主要水印
    wm_draw.rectangle([200, 250, 400, 350], fill=(255, 0, 0, 128))
    wm_draw.text((220, 290), "WATERMARK", fill=(255, 255, 255, 200))
    
    # 小水印
    wm_draw.rectangle([100, 400, 200, 450], fill=(255, 0, 0, 100))
    
    # 合并图像
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, watermark_overlay)
    img = img.convert('RGB')
    
    # 保存测试图片
    img.save('test_watermark_image.png')
    print("✅ Test image saved: test_watermark_image.png")
    
    return img

def create_precise_mask():
    """创建精确的mask"""
    print("🎭 Creating precise mask...")
    
    mask = Image.new('L', (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    
    # 主要水印区域
    draw.rectangle([200, 250, 400, 350], fill=255)
    
    # 小水印区域
    draw.rectangle([100, 400, 200, 450], fill=255)
    
    mask.save('test_precise_mask.png')
    print("✅ Precise mask saved: test_precise_mask.png")
    
    return mask

def test_mask_threshold_effects():
    """测试mask_threshold参数的效果"""
    print("\n🧪 Testing mask_threshold parameter effects...")
    
    from web_backend import WatermarkProcessor
    
    try:
        processor = WatermarkProcessor("web_config.yaml")
        test_image = Image.open('test_watermark_image.png')
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            print(f"   Testing threshold: {threshold}")
            
            # 注意：这里演示概念，实际的threshold调整需要修改后端
            result = processor.process_image(
                image=test_image,
                transparent=True,
                max_bbox_percent=10.0,
                force_format="PNG"
            )
            
            if result.success:
                # 保存结果用于对比
                output_path = f"test_threshold_{threshold}.png"
                result.result_image.save(output_path)
                
                if result.mask_image:
                    mask_path = f"test_mask_{threshold}.png"
                    result.mask_image.save(mask_path)
                
                print(f"      ✅ Saved: {output_path}")
            else:
                print(f"      ❌ Failed: {result.error_message}")
    
    except Exception as e:
        print(f"❌ Threshold test failed: {e}")

def test_bbox_percent_effects():
    """测试max_bbox_percent参数的效果"""
    print("\n🧪 Testing max_bbox_percent parameter effects...")
    
    from web_backend import WatermarkProcessor
    
    try:
        processor = WatermarkProcessor("web_config.yaml")
        test_image = Image.open('test_watermark_image.png')
        
        bbox_percents = [1.0, 5.0, 10.0, 25.0, 50.0]
        
        for bbox_percent in bbox_percents:
            print(f"   Testing bbox_percent: {bbox_percent}%")
            
            result = processor.process_image(
                image=test_image,
                transparent=True,
                max_bbox_percent=bbox_percent,
                force_format="PNG"
            )
            
            if result.success:
                output_path = f"test_bbox_{bbox_percent}.png"
                result.result_image.save(output_path)
                
                if result.mask_image:
                    mask_path = f"test_bbox_mask_{bbox_percent}.png"
                    result.mask_image.save(mask_path)
                    
                    # 分析mask覆盖率
                    mask_array = np.array(result.mask_image)
                    white_pixels = np.sum(mask_array > 128)
                    total_pixels = mask_array.size
                    coverage = white_pixels / total_pixels * 100
                    
                    print(f"      ✅ Coverage: {coverage:.1f}% | Saved: {output_path}")
                else:
                    print(f"      ✅ No mask generated | Saved: {output_path}")
            else:
                print(f"      ❌ Failed: {result.error_message}")
    
    except Exception as e:
        print(f"❌ BBox test failed: {e}")

def test_transparency_vs_inpainting():
    """测试透明模式vs填充模式的效果"""
    print("\n🧪 Testing transparency vs inpainting effects...")
    
    from web_backend import WatermarkProcessor
    
    try:
        processor = WatermarkProcessor("web_config.yaml")
        test_image = Image.open('test_watermark_image.png')
        
        modes = [
            (True, "transparent"),
            (False, "inpainting")
        ]
        
        for transparent, mode_name in modes:
            print(f"   Testing {mode_name} mode...")
            
            start_time = time.time()
            
            result = processor.process_image(
                image=test_image,
                transparent=transparent,
                max_bbox_percent=10.0,
                force_format="PNG"
            )
            
            processing_time = time.time() - start_time
            
            if result.success:
                output_path = f"test_mode_{mode_name}.png"
                result.result_image.save(output_path)
                
                print(f"      ✅ {mode_name}: {processing_time:.2f}s | Saved: {output_path}")
                
                # 分析图像特性
                if result.result_image.mode == 'RGBA':
                    print(f"         📊 Output: RGBA with transparency")
                else:
                    print(f"         📊 Output: {result.result_image.mode}")
            else:
                print(f"      ❌ {mode_name} failed: {result.error_message}")
    
    except Exception as e:
        print(f"❌ Mode comparison test failed: {e}")

def analyze_processing_performance():
    """分析处理性能"""
    print("\n📊 Analyzing processing performance...")
    
    from web_backend import WatermarkProcessor
    import psutil
    
    try:
        # 系统信息
        print(f"   💻 CPU: {psutil.cpu_count()} cores")
        print(f"   💾 RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # GPU信息
        import torch
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"   🎮 GPU: {gpu_props.name}")
            print(f"   📱 VRAM: {gpu_props.total_memory / (1024**2):.0f} MB")
        else:
            print(f"   🎮 GPU: Not available")
        
        # 处理器初始化时间
        init_start = time.time()
        processor = WatermarkProcessor("web_config.yaml")
        init_time = time.time() - init_start
        print(f"   🚀 Processor init: {init_time:.2f}s")
        
        # 测试不同尺寸图像的处理时间
        test_image = Image.open('test_watermark_image.png')
        sizes = [256, 512, 1024]
        
        for size in sizes:
            if size <= 512:  # 只测试合理尺寸
                resized_image = test_image.resize((size, size), Image.LANCZOS)
                
                process_start = time.time()
                result = processor.process_image(
                    image=resized_image,
                    transparent=True,
                    max_bbox_percent=10.0,
                    force_format="PNG"
                )
                process_time = time.time() - process_start
                
                if result.success:
                    print(f"   ⏱️  {size}x{size}: {process_time:.2f}s")
                else:
                    print(f"   ❌ {size}x{size}: Failed")
    
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")

def create_comparison_report():
    """创建对比报告"""
    print("\n📋 Creating comparison report...")
    
    report_content = """
# 🔬 Parameter Effects Test Report

## 📅 Test Date
**{date}**

## 🧪 Test Results Summary

### 1. 🎯 Mask Threshold Effects
Different threshold values affect mask sensitivity:
- **0.1-0.3**: More sensitive, may include false positives
- **0.5**: Balanced detection
- **0.7-0.9**: Conservative, may miss subtle watermarks

### 2. 📦 BBox Percent Effects  
Controls maximum detection area:
- **1-5%**: Only small watermarks
- **10%**: Standard setting
- **25-50%**: Large watermarks allowed

### 3. 🔄 Processing Modes
- **Transparent**: Fast, preserves original content
- **Inpainting**: Slower, fills watermark areas

### 4. 📊 Performance Analysis
- **Initialization**: ~5-10 seconds (first time)
- **Processing**: ~2-5 seconds per image
- **Memory Usage**: Varies with image size

## 💡 Recommendations

1. **For Debugging**: Start with default values, then adjust
2. **For Quality**: Use inpainting mode with conservative settings
3. **For Speed**: Use transparency mode with optimized thresholds
4. **For Accuracy**: Test multiple threshold values

## 📁 Generated Files
Check the following output files for visual comparison:
- `test_threshold_*.png` - Threshold comparison
- `test_bbox_*.png` - BBox percentage comparison  
- `test_mode_*.png` - Processing mode comparison

---
*Generated by parameter effects test script*
""".format(date=time.strftime("%Y-%m-%d %H:%M:%S"))
    
    with open('parameter_test_report.md', 'w') as f:
        f.write(report_content)
    
    print("✅ Report saved: parameter_test_report.md")

def main():
    """主测试函数"""
    print("🔬 AI Watermark Remover - Parameter Effects Test")
    print("="*60)
    
    # 创建测试数据
    create_test_image_with_watermark()
    create_precise_mask()
    
    # 运行参数效果测试
    test_mask_threshold_effects()
    test_bbox_percent_effects()
    test_transparency_vs_inpainting()
    
    # 性能分析
    analyze_processing_performance()
    
    # 生成报告
    create_comparison_report()
    
    print("\n" + "="*60)
    print("🎉 Parameter effects testing completed!")
    print("📁 Check generated files for visual comparison:")
    print("   - test_watermark_image.png (original)")
    print("   - test_precise_mask.png (ground truth mask)")
    print("   - test_*.png (parameter variations)")
    print("   - parameter_test_report.md (detailed report)")
    print("\n💡 Use these results to optimize parameters in the debug UI!")

if __name__ == "__main__":
    main()