#!/usr/bin/env python3
"""
测试不同图像尺寸和格式的处理能力
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

def create_test_image_with_mask(size, color='white'):
    """创建指定尺寸的测试图像和mask"""
    # 创建测试图像
    test_image = Image.new('RGB', size, color=color)
    
    # 添加一些内容
    img_array = np.array(test_image)
    h, w = size[1], size[0]
    
    # 添加颜色区域
    img_array[h//4:h//2, w//4:w//2] = [255, 100, 100]  # 红色
    img_array[h//2:3*h//4, w//2:3*w//4] = [100, 255, 100]  # 绿色
    test_image = Image.fromarray(img_array)
    
    # 创建mask
    test_mask = Image.new('L', size, color=0)  # 黑色背景
    mask_array = np.array(test_mask)
    
    # 在mask中心添加白色区域（模拟水印）
    center_x, center_y = w // 2, h // 2
    mask_w, mask_h = min(100, w//4), min(50, h//4)
    mask_array[center_y-mask_h:center_y+mask_h, center_x-mask_w:center_x+mask_w] = 255
    test_mask = Image.fromarray(mask_array, mode='L')
    
    return test_image, test_mask

def test_image_size_and_format(size, format_ext, model='mat'):
    """测试指定尺寸和格式"""
    print(f"\n🧪 测试 {size[0]}x{size[1]} {format_ext.upper()} 格式...")
    
    # 创建测试数据
    test_image, test_mask = create_test_image_with_mask(size)
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    # 设置参数
    mask_params = {
        'uploaded_mask': test_mask,
        'mask_dilate_kernel_size': 3,
        'mask_dilate_iterations': 1
    }
    
    inpaint_params = {
        'inpaint_model': 'iopaint',
        'force_model': model,
        'auto_model_selection': False,
        'ldm_steps': 20,
        'hd_strategy': 'ORIGINAL',  # 确保无resize
        'seed': -1
    }
    
    performance_params = {
        'mixed_precision': True,
        'log_processing_time': True
    }
    
    # 开始处理
    start_time = time.time()
    
    try:
        result = process_image(
            image=test_image,
            mask_model='upload',
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params=performance_params,
            transparent=False,
            config_manager=config_manager
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result.success:
            # 验证尺寸保持不变
            size_preserved = result.result_image.size == test_image.size
            
            # 保存结果
            filename = f"scripts/test_{size[0]}x{size[1]}_{model}_result.{format_ext}"
            if format_ext.lower() == 'jpg':
                result.result_image.convert('RGB').save(filename, 'JPEG', quality=95)
            else:
                result.result_image.save(filename, format_ext.upper())
            
            print(f"✅ {size[0]}x{size[1]} {format_ext.upper()} 处理成功")
            print(f"   耗时: {processing_time:.2f}秒")
            print(f"   尺寸保持: {'✅' if size_preserved else '❌'}")
            print(f"   输入尺寸: {test_image.size}")
            print(f"   输出尺寸: {result.result_image.size}")
            print(f"   结果已保存: {filename}")
            
            return True, processing_time, size_preserved
        else:
            print(f"❌ {size[0]}x{size[1]} {format_ext.upper()} 处理失败: {result.error_message}")
            return False, processing_time, False
            
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"❌ {size[0]}x{size[1]} {format_ext.upper()} 测试异常: {str(e)}")
        return False, processing_time, False

def test_all_formats_and_sizes():
    """测试所有格式和尺寸组合"""
    print("🚀 开始测试不同图像尺寸和格式...")
    
    # 测试尺寸列表（精简版，重点测试）
    test_sizes = [
        (512, 512),      # 标准尺寸
        (1024, 768),     # 中等尺寸
        (1920, 1080),    # 高分辨率 
        (2048, 1536),    # 大尺寸
    ]
    
    # 测试格式列表
    test_formats = ['png', 'jpg', 'webp']
    
    # 测试模型（选择最快的FCF）
    test_model = 'fcf'
    
    results = {}
    
    print(f"📋 将测试 {len(test_sizes)} 种尺寸 × {len(test_formats)} 种格式 = {len(test_sizes) * len(test_formats)} 种组合")
    
    for size in test_sizes:
        for format_ext in test_formats:
            test_key = f"{size[0]}x{size[1]}_{format_ext}"
            
            success, processing_time, size_preserved = test_image_size_and_format(
                size, format_ext, test_model
            )
            
            results[test_key] = {
                'success': success,
                'time': processing_time,
                'size_preserved': size_preserved,
                'size': size,
                'format': format_ext
            }
    
    # 输出总结
    print("\n" + "="*60)
    print("📊 图像尺寸和格式测试结果总结")
    print("="*60)
    
    # 按尺寸分组统计
    for size in test_sizes:
        print(f"\n📏 {size[0]}x{size[1]} 尺寸:")
        for format_ext in test_formats:
            test_key = f"{size[0]}x{size[1]}_{format_ext}"
            result = results[test_key]
            
            status = "✅" if result['success'] else "❌"
            size_status = "✅" if result['size_preserved'] else "❌"
            
            print(f"   {format_ext.upper():>4}: {status} (耗时: {result['time']:.2f}s, 尺寸保持: {size_status})")
    
    # 总体统计
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r['success'])
    size_preserved_tests = sum(1 for r in results.values() if r['size_preserved'])
    
    success_rate = successful_tests / total_tests * 100
    size_preservation_rate = size_preserved_tests / total_tests * 100
    
    print(f"\n🎯 总体统计:")
    print(f"   总测试数: {total_tests}")
    print(f"   成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    print(f"   尺寸保持率: {size_preservation_rate:.1f}% ({size_preserved_tests}/{total_tests})")
    
    # 性能分析
    if successful_tests > 0:
        successful_results = [r for r in results.values() if r['success']]
        avg_time = sum(r['time'] for r in successful_results) / len(successful_results)
        max_time = max(r['time'] for r in successful_results)
        min_time = min(r['time'] for r in successful_results)
        
        print(f"\n⏱️ 处理时间分析:")
        print(f"   平均时间: {avg_time:.2f}秒")
        print(f"   最快时间: {min_time:.2f}秒")
        print(f"   最慢时间: {max_time:.2f}秒")
        
        # 找出最慢的测试
        slowest = max(successful_results, key=lambda x: x['time'])
        print(f"   最慢测试: {slowest['size'][0]}x{slowest['size'][1]} {slowest['format'].upper()} ({slowest['time']:.2f}s)")
    
    # 失败分析
    failed_tests = [k for k, r in results.items() if not r['success']]
    if failed_tests:
        print(f"\n❌ 失败的测试:")
        for test_key in failed_tests:
            print(f"   {test_key}")
    
    # 尺寸问题分析
    size_issues = [k for k, r in results.items() if r['success'] and not r['size_preserved']]
    if size_issues:
        print(f"\n⚠️ 尺寸被改变的测试:")
        for test_key in size_issues:
            print(f"   {test_key}")
    
    return results

if __name__ == "__main__":
    results = test_all_formats_and_sizes()