#!/usr/bin/env python3
"""
验证IOPaint官方标准对齐 - 确认输入图像格式和预处理流程
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from core.inference import process_image
from config.config import ConfigManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image_format_handling():
    """测试图像格式处理"""
    print("\n🧪 测试图像格式处理...")
    
    size = (512, 512)
    
    # 测试不同图像格式
    formats_to_test = [
        ('RGB', 'rgb_image'),
        ('RGBA', 'rgba_image'), 
        ('L', 'grayscale_image'),
        ('P', 'palette_image')
    ]
    
    config_manager = ConfigManager()
    
    # 创建简单mask
    test_mask = Image.new('L', size, color=0)
    mask_array = np.array(test_mask)
    mask_array[200:300, 200:300] = 255
    test_mask = Image.fromarray(mask_array, mode='L')
    
    results = {}
    
    for mode, name in formats_to_test:
        print(f"  📋 测试 {mode} 格式 ({name})...")
        
        # 创建对应格式的测试图像
        if mode == 'RGB':
            test_image = Image.new('RGB', size, color=(255, 200, 100))
        elif mode == 'RGBA':
            test_image = Image.new('RGBA', size, color=(255, 200, 100, 255))
        elif mode == 'L':
            test_image = Image.new('L', size, color=128)
        elif mode == 'P':
            test_image = Image.new('P', size, color=100)
        
        # 添加一些内容
        img_array = np.array(test_image)
        if len(img_array.shape) == 3:  # RGB/RGBA
            channels = img_array.shape[2]
            if channels == 3:  # RGB
                img_array[100:150, 100:150] = [255, 0, 0]
            elif channels == 4:  # RGBA
                img_array[100:150, 100:150] = [255, 0, 0, 255]
        else:  # L/P
            img_array[100:150, 100:150] = 255
        
        if mode in ['RGB', 'RGBA']:
            test_image = Image.fromarray(img_array, mode=mode)
        else:
            test_image = Image.fromarray(img_array, mode=mode)
        
        # 测试处理
        try:
            result = process_image(
                image=test_image,
                mask_model='upload',
                mask_params={'uploaded_mask': test_mask},
                inpaint_params={
                    'inpaint_model': 'iopaint',
                    'force_model': 'fcf',
                    'hd_strategy': 'ORIGINAL'
                },
                config_manager=config_manager
            )
            
            if result.success:
                # 检查输出格式
                output_mode = result.result_image.mode
                input_size = test_image.size
                output_size = result.result_image.size
                
                results[name] = {
                    'input_mode': mode,
                    'output_mode': output_mode,
                    'input_size': input_size,
                    'output_size': output_size,
                    'success': True,
                    'size_preserved': input_size == output_size
                }
                
                print(f"    ✅ {mode} → {output_mode}, 尺寸: {input_size} → {output_size}")
                
                # 保存结果
                result.result_image.save(f"scripts/iopaint_format_test_{name}.png")
                
            else:
                results[name] = {
                    'input_mode': mode,
                    'success': False,
                    'error': result.error_message
                }
                print(f"    ❌ {mode} 处理失败: {result.error_message}")
                
        except Exception as e:
            results[name] = {
                'input_mode': mode,
                'success': False,
                'error': str(e)
            }
            print(f"    ❌ {mode} 处理异常: {e}")
    
    return results

def test_preprocessing_pipeline():
    """测试预处理管道"""
    print("\n🧪 测试预处理管道...")
    
    config_manager = ConfigManager()
    
    # 创建不同尺寸的测试图像
    test_sizes = [
        (256, 256),    # 小尺寸
        (512, 512),    # 标准尺寸
        (1024, 768),   # 中等尺寸
        (1920, 1080),  # 大尺寸
        (1000, 1500),  # 非标准比例
    ]
    
    results = {}
    
    for size in test_sizes:
        print(f"  📋 测试尺寸 {size[0]}x{size[1]}...")
        
        # 创建测试图像和mask
        test_image = Image.new('RGB', size, color=(200, 150, 100))
        img_array = np.array(test_image)
        h, w = size[1], size[0]
        img_array[h//4:h//2, w//4:w//2] = [255, 100, 100]
        test_image = Image.fromarray(img_array)
        
        # 创建对应mask
        test_mask = Image.new('L', size, color=0)
        mask_array = np.array(test_mask)
        mask_array[h//3:2*h//3, w//3:2*w//3] = 255
        test_mask = Image.fromarray(mask_array, mode='L')
        
        try:
            start_time = time.time()
            
            result = process_image(
                image=test_image,
                mask_model='upload',
                mask_params={'uploaded_mask': test_mask},
                inpaint_params={
                    'inpaint_model': 'iopaint',
                    'force_model': 'fcf',
                    'hd_strategy': 'ORIGINAL',  # 确保无resize
                    'ldm_steps': 20
                },
                config_manager=config_manager
            )
            
            processing_time = time.time() - start_time
            
            if result.success:
                input_size = test_image.size
                output_size = result.result_image.size
                size_preserved = input_size == output_size
                
                # 检查像素值范围
                result_array = np.array(result.result_image)
                pixel_range = (result_array.min(), result_array.max())
                
                results[f"{size[0]}x{size[1]}"] = {
                    'input_size': input_size,
                    'output_size': output_size,
                    'size_preserved': size_preserved,
                    'processing_time': processing_time,
                    'pixel_range': pixel_range,
                    'success': True
                }
                
                print(f"    ✅ 处理成功: {input_size} → {output_size}, 耗时: {processing_time:.2f}s")
                print(f"    📊 像素值范围: {pixel_range[0]} - {pixel_range[1]}")
                
                # 保存结果
                result.result_image.save(f"scripts/iopaint_preprocess_test_{size[0]}x{size[1]}.png")
                
            else:
                results[f"{size[0]}x{size[1]}"] = {
                    'input_size': size,
                    'success': False,
                    'error': result.error_message,
                    'processing_time': processing_time
                }
                print(f"    ❌ 处理失败: {result.error_message}")
                
        except Exception as e:
            results[f"{size[0]}x{size[1]}"] = {
                'input_size': size,
                'success': False,
                'error': str(e)
            }
            print(f"    ❌ 处理异常: {e}")
    
    return results

def test_tensor_format_validation():
    """测试tensor格式验证"""
    print("\n🧪 测试tensor格式验证...")
    
    try:
        # 检查模型输入要求
        import torch
        from core.models.unified_processor import UnifiedProcessor
        from config.config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # 创建unified processor实例
        processor = UnifiedProcessor(config)
        available_models = processor.get_available_models()
        
        print(f"  📋 可用模型: {available_models}")
        
        if available_models:
            # 测试第一个可用模型
            test_model = available_models[0]
            print(f"  🧪 测试模型: {test_model}")
            
            # 创建标准输入
            test_image = Image.new('RGB', (512, 512), color=(200, 150, 100))
            test_mask = Image.new('L', (512, 512), color=0)
            mask_array = np.array(test_mask)
            mask_array[200:300, 200:300] = 255
            test_mask = Image.fromarray(mask_array, mode='L')
            
            # 检查输入格式转换
            print("  📊 输入格式分析:")
            print(f"    原始图像: {test_image.mode}, {test_image.size}")
            print(f"    原始mask: {test_mask.mode}, {test_mask.size}")
            
            # 转换为numpy数组（模拟模型输入前的状态）
            image_array = np.array(test_image.convert("RGB"))
            mask_array = np.array(test_mask.convert("L"))
            
            print(f"    转换后图像: {image_array.shape}, dtype: {image_array.dtype}, 范围: {image_array.min()}-{image_array.max()}")
            print(f"    转换后mask: {mask_array.shape}, dtype: {mask_array.dtype}, 范围: {mask_array.min()}-{mask_array.max()}")
            
            # 测试处理
            result = processor.predict_with_model(test_model, test_image, test_mask)
            
            print(f"    ✅ 模型 {test_model} 输入格式验证成功")
            print(f"    📊 输出: {result.shape}, dtype: {result.dtype}")
            
            return True, {
                'input_image_format': f"{image_array.shape}, {image_array.dtype}",
                'input_mask_format': f"{mask_array.shape}, {mask_array.dtype}",
                'output_format': f"{result.shape}, {result.dtype}",
                'model_tested': test_model
            }
        else:
            print("  ❌ 没有可用模型进行测试")
            return False, {'error': 'No models available'}
            
    except Exception as e:
        print(f"  ❌ Tensor格式验证失败: {e}")
        return False, {'error': str(e)}

def test_iopaint_alignment():
    """IOPaint官方标准对齐完整测试"""
    print("🚀 开始IOPaint官方标准对齐验证...")
    
    results = {
        'image_formats': test_image_format_handling(),
        'preprocessing': test_preprocessing_pipeline(),
        'tensor_validation': test_tensor_format_validation()
    }
    
    # 输出总结
    print("\n" + "="*60)
    print("📊 IOPaint官方标准对齐验证结果总结")
    print("="*60)
    
    # 图像格式测试总结
    print("\n🎨 图像格式处理:")
    for name, result in results['image_formats'].items():
        if result['success']:
            status = "✅"
            size_status = "✅" if result.get('size_preserved', False) else "⚠️"
            print(f"   {name:>15}: {status} ({result['input_mode']} → {result['output_mode']}, 尺寸保持: {size_status})")
        else:
            print(f"   {name:>15}: ❌ ({result.get('error', 'Unknown error')})")
    
    # 预处理管道测试总结
    print("\n⚙️ 预处理管道:")
    for size, result in results['preprocessing'].items():
        if result['success']:
            status = "✅"
            size_status = "✅" if result['size_preserved'] else "⚠️"
            print(f"   {size:>10}: {status} (尺寸保持: {size_status}, 耗时: {result['processing_time']:.2f}s)")
        else:
            print(f"   {size:>10}: ❌ ({result.get('error', 'Unknown error')})")
    
    # Tensor验证总结
    print("\n🔢 Tensor格式验证:")
    tensor_success, tensor_result = results['tensor_validation']
    if tensor_success:
        print("   ✅ Tensor格式验证成功")
        print(f"   📊 输入图像格式: {tensor_result['input_image_format']}")
        print(f"   📊 输入mask格式: {tensor_result['input_mask_format']}")
        print(f"   📊 输出格式: {tensor_result['output_format']}")
        print(f"   🎯 测试模型: {tensor_result['model_tested']}")
    else:
        print(f"   ❌ Tensor格式验证失败: {tensor_result['error']}")
    
    # 整体评估
    format_success_rate = sum(1 for r in results['image_formats'].values() if r['success']) / len(results['image_formats']) * 100
    preprocess_success_rate = sum(1 for r in results['preprocessing'].values() if r['success']) / len(results['preprocessing']) * 100
    
    print(f"\n🎯 整体评估:")
    print(f"   图像格式兼容性: {format_success_rate:.1f}%")
    print(f"   预处理管道成功率: {preprocess_success_rate:.1f}%")
    print(f"   Tensor格式验证: {'✅' if tensor_success else '❌'}")
    
    # IOPaint标准符合性评估
    iopaint_compliance = format_success_rate >= 75 and preprocess_success_rate >= 80 and tensor_success
    print(f"\n📋 IOPaint标准符合性: {'✅ 符合' if iopaint_compliance else '⚠️ 需要改进'}")
    
    return results, iopaint_compliance

if __name__ == "__main__":
    results, compliance = test_iopaint_alignment()
    
    if compliance:
        print("\n🎉 IOPaint官方标准对齐验证通过!")
    else:
        print("\n⚠️ IOPaint标准对齐需要进一步优化")