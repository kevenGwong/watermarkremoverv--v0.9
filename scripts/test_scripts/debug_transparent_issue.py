"""
Debug script to diagnose and fix transparent functionality issues
"""
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import os
from web_backend import WatermarkProcessor

def create_test_image_with_known_watermark():
    """创建一个已知水印位置的测试图片"""
    # 创建基础图片
    img = Image.new('RGB', (400, 300), 'lightblue')
    draw = ImageDraw.Draw(img)
    
    # 绘制一些背景内容
    draw.rectangle([50, 50, 150, 100], fill='lightgreen')
    draw.rectangle([200, 150, 300, 200], fill='lightyellow')
    
    # 添加明显的"水印" - 黑色矩形
    draw.rectangle([250, 50, 350, 100], fill='black', outline='red', width=2)
    draw.text((260, 65), "WATERMARK", fill='white')
    
    return img

def create_perfect_mask():
    """创建完美的mask（已知水印位置）"""
    mask = Image.new('L', (400, 300), 0)  # 黑色背景
    draw = ImageDraw.Draw(mask)
    
    # 水印区域设为白色
    draw.rectangle([250, 50, 350, 100], fill=255)
    
    return mask

def test_transparent_function_original(image, mask):
    """测试原版透明函数"""
    print("🔍 Testing original transparent function...")
    
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:  # 白色区域
                transparent_image.putpixel((x, y), (0, 0, 0, 0))  # 设为透明
            else:  # 黑色区域
                transparent_image.putpixel((x, y), image.getpixel((x, y)))  # 保持原样
    
    return transparent_image

def test_transparent_function_numpy(image, mask):
    """测试numpy优化版透明函数"""
    print("🔍 Testing numpy optimized transparent function...")
    
    # 转换格式
    image = image.convert("RGBA")
    mask = mask.convert("L")
    
    # 转换为numpy数组
    img_array = np.array(image)
    mask_array = np.array(mask)
    
    # 创建结果数组
    result_array = img_array.copy()
    
    # 水印区域设为透明
    transparent_mask = mask_array > 128
    result_array[transparent_mask, 3] = 0  # 设置alpha通道为0
    
    return Image.fromarray(result_array, 'RGBA')

def analyze_mask_properties(mask):
    """分析mask的属性"""
    print("\n📊 Mask Analysis:")
    mask_array = np.array(mask)
    
    print(f"  Shape: {mask_array.shape}")
    print(f"  Data type: {mask_array.dtype}")
    print(f"  Min value: {mask_array.min()}")
    print(f"  Max value: {mask_array.max()}")
    print(f"  Unique values: {np.unique(mask_array)}")
    
    white_pixels = np.sum(mask_array > 128)
    total_pixels = mask_array.size
    print(f"  White pixels (>128): {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
    
    return {
        'white_pixels': white_pixels,
        'total_pixels': total_pixels,
        'white_ratio': white_pixels / total_pixels
    }

def create_visualization(original, mask, transparent_result, save_path="debug_transparent.png"):
    """创建可视化对比图"""
    print(f"\n🎨 Creating visualization: {save_path}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title("Mask (White=Watermark)")
    axes[0, 1].axis('off')
    
    # 透明结果 - 白色背景
    if transparent_result.mode == 'RGBA':
        # 创建白色背景
        white_bg = Image.new('RGB', transparent_result.size, (255, 255, 255))
        white_bg.paste(transparent_result, mask=transparent_result.split()[-1])
        axes[0, 2].imshow(white_bg)
        axes[0, 2].set_title("Transparent Result (White BG)")
    else:
        axes[0, 2].imshow(transparent_result)
        axes[0, 2].set_title("Result")
    axes[0, 2].axis('off')
    
    # 透明结果 - 黑色背景
    if transparent_result.mode == 'RGBA':
        black_bg = Image.new('RGB', transparent_result.size, (0, 0, 0))
        black_bg.paste(transparent_result, mask=transparent_result.split()[-1])
        axes[1, 0].imshow(black_bg)
        axes[1, 0].set_title("Transparent Result (Black BG)")
    else:
        axes[1, 0].imshow(transparent_result)
        axes[1, 0].set_title("Result")
    axes[1, 0].axis('off')
    
    # 透明结果 - 棋盘背景
    if transparent_result.mode == 'RGBA':
        # 创建棋盘格背景
        checkered = Image.new('RGB', transparent_result.size, (255, 255, 255))
        for x in range(0, transparent_result.width, 20):
            for y in range(0, transparent_result.height, 20):
                if (x//20 + y//20) % 2:
                    for i in range(min(20, transparent_result.width - x)):
                        for j in range(min(20, transparent_result.height - y)):
                            checkered.putpixel((x + i, y + j), (200, 200, 200))
        checkered.paste(transparent_result, mask=transparent_result.split()[-1])
        axes[1, 1].imshow(checkered)
        axes[1, 1].set_title("Transparent Result (Checkered BG)")
    else:
        axes[1, 1].imshow(transparent_result)
        axes[1, 1].set_title("Result")
    axes[1, 1].axis('off')
    
    # Alpha通道可视化
    if transparent_result.mode == 'RGBA':
        alpha_channel = np.array(transparent_result.split()[-1])
        axes[1, 2].imshow(alpha_channel, cmap='gray')
        axes[1, 2].set_title("Alpha Channel (Black=Transparent)")
    else:
        axes[1, 2].text(0.5, 0.5, "No Alpha Channel", ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title("No Alpha Channel")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def test_with_processor():
    """使用实际的processor进行测试"""
    print("\n🤖 Testing with actual WatermarkProcessor...")
    
    try:
        processor = WatermarkProcessor("web_config.yaml")
        
        # 创建测试图片
        test_image = create_test_image_with_known_watermark()
        
        # 测试透明模式
        result = processor.process_image(
            image=test_image,
            transparent=True,
            max_bbox_percent=50.0  # 增大阈值确保检测到
        )
        
        if result.success:
            print("✅ Processor test successful!")
            print(f"   Processing time: {result.processing_time:.2f}s")
            
            if result.mask_image:
                mask_stats = analyze_mask_properties(result.mask_image)
                if mask_stats['white_ratio'] < 0.01:
                    print("⚠️  Warning: Very few white pixels in mask - may not detect watermark")
            
            return test_image, result.mask_image, result.result_image
        else:
            print(f"❌ Processor test failed: {result.error_message}")
            return None, None, None
            
    except Exception as e:
        print(f"❌ Processor test error: {e}")
        return None, None, None

def main():
    """主测试函数"""
    print("🔧 Watermark Transparent Function Debug Tool")
    print("=" * 50)
    
    # 测试1: 已知正确的图片和mask
    print("\n📌 Test 1: Known correct image and mask")
    test_image = create_test_image_with_known_watermark()
    perfect_mask = create_perfect_mask()
    
    print("💾 Saving test image and mask...")
    os.makedirs("debug_output", exist_ok=True)
    test_image.save("debug_output/test_image.png")
    perfect_mask.save("debug_output/perfect_mask.png")
    
    # 分析mask
    mask_stats = analyze_mask_properties(perfect_mask)
    
    # 测试原版函数
    transparent_original = test_transparent_function_original(test_image, perfect_mask)
    transparent_original.save("debug_output/transparent_original.png")
    
    # 测试numpy优化版
    transparent_numpy = test_transparent_function_numpy(test_image, perfect_mask)
    transparent_numpy.save("debug_output/transparent_numpy.png")
    
    # 创建可视化
    create_visualization(test_image, perfect_mask, transparent_original, 
                        "debug_output/debug_visualization_original.png")
    create_visualization(test_image, perfect_mask, transparent_numpy, 
                        "debug_output/debug_visualization_numpy.png")
    
    print("✅ Test 1 completed. Check debug_output/ folder for results.")
    
    # 测试2: 使用实际的processor
    print("\n📌 Test 2: Actual processor test")
    proc_image, proc_mask, proc_result = test_with_processor()
    
    if proc_image is not None:
        proc_image.save("debug_output/processor_test_image.png")
        if proc_mask:
            proc_mask.save("debug_output/processor_mask.png")
            analyze_mask_properties(proc_mask)
        if proc_result:
            proc_result.save("debug_output/processor_result.png")
            create_visualization(proc_image, proc_mask, proc_result,
                               "debug_output/processor_visualization.png")
        print("✅ Test 2 completed.")
    
    # 生成诊断报告
    print("\n📋 Diagnostic Report")
    print("-" * 30)
    
    if mask_stats['white_ratio'] > 0.05:
        print("✅ Perfect mask has reasonable white pixel ratio")
    else:
        print("⚠️  Perfect mask has very low white pixel ratio")
    
    # 检查透明效果
    if transparent_original.mode == 'RGBA':
        alpha_array = np.array(transparent_original.split()[-1])
        transparent_pixels = np.sum(alpha_array == 0)
        print(f"✅ Transparent pixels created: {transparent_pixels}")
        
        if transparent_pixels == mask_stats['white_pixels']:
            print("✅ Transparent pixel count matches mask white pixels")
        else:
            print("⚠️  Transparent pixel count doesn't match mask")
    
    print("\n🎯 Recommendations:")
    print("1. Check if mask generation is working correctly")
    print("2. Verify mask format: white=watermark, black=background")
    print("3. Use PNG format for transparent images")
    print("4. Check browser compatibility for transparent image display")
    print("5. Consider using checkered background for transparent preview")
    
    print(f"\n📁 All debug files saved to: {os.path.abspath('debug_output')}")

if __name__ == "__main__":
    main()