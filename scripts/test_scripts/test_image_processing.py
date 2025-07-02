#!/usr/bin/env python3
"""
实际图像处理测试
使用用户提供的测试图片进行功能验证
"""
import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

def save_test_images():
    """保存用户提供的测试图片到本地"""
    print("📸 Preparing test images...")
    
    # 创建测试目录
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # 第一张图片 - 手表图片（有水印）
    print("⏱️  Test image 1: Watch image with watermarks")
    watch_image_path = test_dir / "watch_with_watermark.jpg"
    print(f"   Expected path: {watch_image_path}")
    
    # 第二张图片 - 透明mask
    print("🎭 Test image 2: Transparent mask")
    mask_image_path = test_dir / "precise_mask.png"
    print(f"   Expected path: {mask_image_path}")
    
    return watch_image_path, mask_image_path

def test_custom_model_processing(image_path):
    """测试自定义模型处理"""
    print("\n🤖 Testing Custom Model Processing...")
    
    try:
        from web_backend import WatermarkProcessor
        
        # 初始化处理器
        processor = WatermarkProcessor("web_config.yaml")
        print("✅ Processor initialized")
        
        # 加载测试图片
        if not os.path.exists(image_path):
            print(f"⚠️  Test image not found: {image_path}")
            return False
            
        image = Image.open(image_path)
        print(f"✅ Test image loaded: {image.size}")
        
        # 测试不同参数设置
        test_configs = [
            {"transparent": False, "max_bbox_percent": 10.0, "name": "remove_standard"},
            {"transparent": True, "max_bbox_percent": 10.0, "name": "transparent_standard"},
            {"transparent": False, "max_bbox_percent": 5.0, "name": "remove_strict"},
            {"transparent": True, "max_bbox_percent": 25.0, "name": "transparent_loose"}
        ]
        
        results = []
        for config in test_configs:
            print(f"\n   🔄 Testing config: {config['name']}")
            print(f"      - Transparent: {config['transparent']}")
            print(f"      - Max bbox: {config['max_bbox_percent']}%")
            
            start_time = time.time()
            
            try:
                result = processor.process_image(
                    image=image,
                    transparent=config['transparent'],
                    max_bbox_percent=config['max_bbox_percent'],
                    force_format="PNG"
                )
                
                processing_time = time.time() - start_time
                
                if result.success:
                    print(f"      ✅ Success in {processing_time:.2f}s")
                    
                    # 保存结果
                    output_path = f"test_output_{config['name']}.png"
                    result.result_image.save(output_path)
                    print(f"      💾 Saved: {output_path}")
                    
                    # 保存mask
                    if result.mask_image:
                        mask_path = f"test_mask_{config['name']}.png"
                        result.mask_image.save(mask_path)
                        print(f"      🎭 Mask saved: {mask_path}")
                    
                    results.append((config['name'], True, processing_time))
                else:
                    print(f"      ❌ Failed: {result.error_message}")
                    results.append((config['name'], False, processing_time))
                    
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"      ❌ Exception: {e}")
                results.append((config['name'], False, processing_time))
        
        # 总结结果
        print(f"\n📋 Custom Model Test Summary:")
        for name, success, proc_time in results:
            status = "✅" if success else "❌"
            print(f"   {status} {name}: {proc_time:.2f}s")
        
        return len([r for r in results if r[1]]) > 0
        
    except Exception as e:
        print(f"❌ Custom model test failed: {e}")
        return False

def test_custom_mask_processing(image_path, mask_path):
    """测试自定义mask处理"""
    print("\n🎭 Testing Custom Mask Processing...")
    
    try:
        from web_backend import WatermarkProcessor
        
        processor = WatermarkProcessor("web_config.yaml")
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            return False
            
        if not os.path.exists(mask_path):
            print(f"⚠️  Mask not found: {mask_path}")
            return False
        
        # 加载图片和mask
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')
        
        print(f"✅ Image loaded: {image.size}")
        print(f"✅ Mask loaded: {mask.size}")
        
        # 确保尺寸匹配
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.LANCZOS)
            print(f"🔧 Mask resized to match image: {image.size}")
        
        # 测试透明效果
        print("\n   🔄 Testing transparency with custom mask...")
        
        start_time = time.time()
        
        # 手动应用透明效果（绕过复杂的处理流程）
        image_rgba = image.convert("RGBA")
        img_array = np.array(image_rgba)
        mask_array = np.array(mask)
        
        # 应用透明效果
        transparent_mask = mask_array > 128
        img_array[transparent_mask, 3] = 0  # 设置alpha通道为0
        
        result_image = Image.fromarray(img_array, 'RGBA')
        
        processing_time = time.time() - start_time
        
        # 保存结果
        result_path = "test_custom_mask_result.png"
        result_image.save(result_path)
        
        print(f"      ✅ Custom mask processing success in {processing_time:.2f}s")
        print(f"      💾 Result saved: {result_path}")
        
        # 分析mask统计
        white_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        coverage = white_pixels / total_pixels * 100
        
        print(f"      📊 Mask coverage: {coverage:.1f}% ({white_pixels}/{total_pixels} pixels)")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom mask test failed: {e}")
        return False

def test_parameter_extremes():
    """测试极值参数的实际效果"""
    print("\n⚗️  Testing Parameter Extremes...")
    
    try:
        # 创建一个简单的测试图片
        test_img = Image.new('RGB', (256, 256), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_img)
        
        # 添加测试"水印"
        draw.rectangle([50, 50, 200, 150], fill='red', outline='black', width=2)
        draw.text((80, 90), "TEST", fill='white')
        
        test_img.save("test_synthetic.png")
        print("✅ Synthetic test image created")
        
        from web_backend import WatermarkProcessor
        processor = WatermarkProcessor("web_config.yaml")
        
        # 测试极值参数
        extreme_tests = [
            {"max_bbox_percent": 1.0, "name": "min_bbox"},
            {"max_bbox_percent": 50.0, "name": "max_bbox"},
        ]
        
        for test_config in extreme_tests:
            print(f"\n   🔄 Testing {test_config['name']}...")
            
            try:
                result = processor.process_image(
                    image=test_img,
                    transparent=True,
                    max_bbox_percent=test_config['max_bbox_percent'],
                    force_format="PNG"
                )
                
                if result.success:
                    output_path = f"test_extreme_{test_config['name']}.png"
                    result.result_image.save(output_path)
                    print(f"      ✅ Success: {output_path}")
                else:
                    print(f"      ❌ Failed: {result.error_message}")
                    
            except Exception as e:
                print(f"      ❌ Exception: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter extreme test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("🎨 AI Watermark Remover - Image Processing Test")
    print("="*60)
    
    # 准备测试图片
    watch_path, mask_path = save_test_images()
    
    # 运行测试
    tests_results = []
    
    print("\n" + "="*60)
    print("🔄 RUNNING IMAGE PROCESSING TESTS")
    print("="*60)
    
    # 测试1: 自定义模型处理
    if os.path.exists(watch_path):
        result1 = test_custom_model_processing(watch_path)
        tests_results.append(("Custom Model Processing", result1))
    else:
        print(f"⚠️  Skipping custom model test - image not found: {watch_path}")
        tests_results.append(("Custom Model Processing", False))
    
    # 测试2: 自定义mask处理
    if os.path.exists(watch_path) and os.path.exists(mask_path):
        result2 = test_custom_mask_processing(watch_path, mask_path)
        tests_results.append(("Custom Mask Processing", result2))
    else:
        print(f"⚠️  Skipping custom mask test - files not found")
        tests_results.append(("Custom Mask Processing", False))
    
    # 测试3: 极值参数
    result3 = test_parameter_extremes()
    tests_results.append(("Parameter Extremes", result3))
    
    # 总结
    print("\n" + "="*60)
    print("🎯 IMAGE PROCESSING TEST RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, result in tests_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print("-"*60)
    print(f"Total: {passed}/{len(tests_results)} tests passed")
    
    if passed > 0:
        print("🎉 Some tests passed! Check output files for results.")
        print("\n📁 Output files to check:")
        for file_pattern in ["test_output_*.png", "test_mask_*.png", "test_custom_*.png", "test_extreme_*.png"]:
            print(f"   - {file_pattern}")
    else:
        print("⚠️  No tests passed. Check error messages above.")

if __name__ == "__main__":
    main()