#!/usr/bin/env python3
"""
测试mask生成和上传的各种场景
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import io

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

def create_test_image():
    """创建复杂的测试图像"""
    size = (768, 768)
    image = Image.new('RGB', size, color='white')
    img_array = np.array(image)
    
    # 创建复杂场景
    h, w = size[1], size[0]
    
    # 背景纹理
    for i in range(h):
        for j in range(w):
            img_array[i, j] = [
                int(200 + 50 * np.sin(i/50) * np.cos(j/50)),
                int(180 + 40 * np.sin(i/30)),
                int(220 + 30 * np.cos(j/40))
            ]
    
    # 添加一些对象
    img_array[100:200, 100:250] = [255, 100, 100]  # 红色区域
    img_array[300:450, 200:400] = [100, 255, 100]  # 绿色区域
    img_array[500:600, 400:550] = [100, 100, 255]  # 蓝色区域
    
    # 添加"水印"区域（我们要去除的）
    img_array[200:300, 300:500] = [50, 50, 50]     # 深色水印
    img_array[450:500, 500:650] = [240, 240, 240]  # 浅色水印
    
    return Image.fromarray(img_array)

def create_various_masks():
    """创建各种类型的mask"""
    size = (768, 768)
    masks = {}
    
    # 1. 简单矩形mask
    simple_mask = Image.new('L', size, color=0)
    mask_array = np.array(simple_mask)
    mask_array[200:300, 300:500] = 255  # 矩形水印区域
    masks['simple_rectangle'] = Image.fromarray(mask_array, mode='L')
    
    # 2. 复杂形状mask
    complex_mask = Image.new('L', size, color=0)
    mask_array = np.array(complex_mask)
    # 圆形区域
    center_x, center_y = 400, 250
    for i in range(size[1]):
        for j in range(size[0]):
            if (i - center_y)**2 + (j - center_x)**2 < 50**2:
                mask_array[i, j] = 255
    # 不规则区域
    mask_array[450:500, 500:650] = 255
    masks['complex_shape'] = Image.fromarray(mask_array, mode='L')
    
    # 3. 多个分离区域mask
    multi_mask = Image.new('L', size, color=0)
    mask_array = np.array(multi_mask)
    mask_array[100:150, 100:200] = 255  # 区域1
    mask_array[300:350, 400:500] = 255  # 区域2
    mask_array[600:650, 200:300] = 255  # 区域3
    masks['multiple_regions'] = Image.fromarray(mask_array, mode='L')
    
    # 4. 大面积mask
    large_mask = Image.new('L', size, color=0)
    mask_array = np.array(large_mask)
    mask_array[100:600, 100:600] = 255  # 大面积水印
    masks['large_area'] = Image.fromarray(mask_array, mode='L')
    
    # 5. 边缘mask
    edge_mask = Image.new('L', size, color=0)
    mask_array = np.array(edge_mask)
    mask_array[0:100, :] = 255      # 顶部边缘
    mask_array[:, 650:768] = 255    # 右侧边缘
    masks['edge_regions'] = Image.fromarray(mask_array, mode='L')
    
    # 6. 精细线条mask
    line_mask = Image.new('L', size, color=0)
    mask_array = np.array(line_mask)
    mask_array[200:210, 100:600] = 255  # 水平线
    mask_array[100:600, 300:310] = 255  # 垂直线
    masks['thin_lines'] = Image.fromarray(mask_array, mode='L')
    
    return masks

def create_mock_uploaded_file(image):
    """创建模拟的上传文件对象"""
    class MockUploadedFile:
        def __init__(self, image):
            self.image = image
            self._buffer = None
        
        def seek(self, pos):
            pass
        
        def read(self, size=None):
            if self._buffer is None:
                self._buffer = io.BytesIO()
                self.image.save(self._buffer, format='PNG')
                self._buffer.seek(0)
            return self._buffer.getvalue()
    
    return MockUploadedFile(image)

def test_mask_scenario(mask_model, mask_name, test_image, test_mask=None, inpaint_model='mat'):
    """测试特定mask场景"""
    print(f"\n🧪 测试 {mask_model} - {mask_name}...")
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    # 设置mask参数
    if mask_model == 'upload':
        mask_params = {
            'uploaded_mask': create_mock_uploaded_file(test_mask),
            'mask_dilate_kernel_size': 3,
            'mask_dilate_iterations': 1
        }
    elif mask_model == 'custom':
        mask_params = {
            'mask_threshold': 0.5,
            'mask_dilate_kernel_size': 5,
            'mask_dilate_iterations': 2
        }
    elif mask_model == 'florence':
        mask_params = {
            'detection_prompt': 'watermark',
            'max_bbox_percent': 15.0,
            'confidence_threshold': 0.3
        }
    else:
        mask_params = {}
    
    # 设置inpainting参数
    inpaint_params = {
        'inpaint_model': 'iopaint',
        'force_model': inpaint_model,
        'auto_model_selection': False,
        'ldm_steps': 30,
        'hd_strategy': 'ORIGINAL',
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
            mask_model=mask_model,
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params=performance_params,
            transparent=False,
            config_manager=config_manager
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result.success:
            # 分析mask质量
            mask_array = np.array(result.mask_image.convert("L"))
            mask_coverage = np.sum(mask_array > 128) / mask_array.size * 100
            mask_quality = analyze_mask_quality(mask_array)
            
            # 保存结果
            filename = f"scripts/test_mask_{mask_model}_{mask_name}_{inpaint_model}_result.png"
            result.result_image.save(filename)
            
            # 保存mask
            mask_filename = f"scripts/test_mask_{mask_model}_{mask_name}_mask.png"
            result.mask_image.save(mask_filename)
            
            print(f"✅ {mask_model} - {mask_name} 处理成功")
            print(f"   耗时: {processing_time:.2f}秒")
            print(f"   Mask覆盖率: {mask_coverage:.2f}%")
            print(f"   Mask质量: {mask_quality}")
            print(f"   结果已保存: {filename}")
            print(f"   Mask已保存: {mask_filename}")
            
            return True, processing_time, mask_coverage, mask_quality
        else:
            print(f"❌ {mask_model} - {mask_name} 处理失败: {result.error_message}")
            return False, processing_time, 0, "failed"
            
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"❌ {mask_model} - {mask_name} 测试异常: {str(e)}")
        return False, processing_time, 0, "error"

def analyze_mask_quality(mask_array):
    """分析mask质量"""
    total_pixels = mask_array.size
    white_pixels = np.sum(mask_array > 128)
    
    if white_pixels == 0:
        return "empty"
    elif white_pixels < total_pixels * 0.01:
        return "very_small"
    elif white_pixels < total_pixels * 0.05:
        return "small"
    elif white_pixels < total_pixels * 0.20:
        return "medium"
    elif white_pixels < total_pixels * 0.50:
        return "large"
    else:
        return "very_large"

def test_all_mask_scenarios():
    """测试所有mask场景"""
    print("🚀 开始测试所有mask生成和上传场景...")
    
    # 创建测试图像
    test_image = create_test_image()
    test_image.save("scripts/test_mask_input_image.png")
    print("📁 测试图像已保存: scripts/test_mask_input_image.png")
    
    # 创建各种mask
    test_masks = create_various_masks()
    
    # 保存测试mask
    for mask_name, mask in test_masks.items():
        mask.save(f"scripts/test_mask_{mask_name}.png")
    print(f"📁 {len(test_masks)} 个测试mask已保存")
    
    results = {}
    
    # 1. 测试upload mask场景
    print("\n" + "="*50)
    print("📋 测试 UPLOAD MASK 场景")
    print("="*50)
    
    for mask_name, mask in test_masks.items():
        success, time_cost, coverage, quality = test_mask_scenario(
            'upload', mask_name, test_image, mask, 'mat'
        )
        results[f'upload_{mask_name}'] = {
            'success': success,
            'time': time_cost,
            'coverage': coverage,
            'quality': quality,
            'mask_model': 'upload',
            'mask_name': mask_name
        }
    
    # 2. 测试custom mask场景
    print("\n" + "="*50)
    print("📋 测试 CUSTOM MASK 场景")
    print("="*50)
    
    success, time_cost, coverage, quality = test_mask_scenario(
        'custom', 'auto_detection', test_image, None, 'mat'
    )
    results['custom_auto'] = {
        'success': success,
        'time': time_cost,
        'coverage': coverage,
        'quality': quality,
        'mask_model': 'custom',
        'mask_name': 'auto_detection'
    }
    
    # 3. 测试florence mask场景
    print("\n" + "="*50)
    print("📋 测试 FLORENCE MASK 场景")
    print("="*50)
    
    success, time_cost, coverage, quality = test_mask_scenario(
        'florence', 'watermark_detection', test_image, None, 'mat'
    )
    results['florence_watermark'] = {
        'success': success,
        'time': time_cost,
        'coverage': coverage,
        'quality': quality,
        'mask_model': 'florence',
        'mask_name': 'watermark_detection'
    }
    
    # 输出总结
    print("\n" + "="*60)
    print("📊 Mask生成和上传测试结果总结")
    print("="*60)
    
    # 按mask模型分组统计
    for mask_model in ['upload', 'custom', 'florence']:
        print(f"\n🎭 {mask_model.upper()} MASK 模式:")
        
        model_results = [r for k, r in results.items() if r['mask_model'] == mask_model]
        if not model_results:
            continue
            
        for result in model_results:
            status = "✅" if result['success'] else "❌"
            print(f"   {result['mask_name']:>20}: {status} (耗时: {result['time']:.2f}s, 覆盖率: {result['coverage']:.1f}%, 质量: {result['quality']})")
    
    # 总体统计
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r['success'])
    success_rate = successful_tests / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 总体统计:")
    print(f"   总测试数: {total_tests}")
    print(f"   成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    
    # 性能分析
    if successful_tests > 0:
        successful_results = [r for r in results.values() if r['success']]
        avg_time = sum(r['time'] for r in successful_results) / len(successful_results)
        avg_coverage = sum(r['coverage'] for r in successful_results) / len(successful_results)
        
        print(f"\n⏱️ 性能分析:")
        print(f"   平均处理时间: {avg_time:.2f}秒")
        print(f"   平均mask覆盖率: {avg_coverage:.1f}%")
        
        # 质量分析
        quality_counts = {}
        for result in successful_results:
            quality = result['quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        print(f"\n📊 Mask质量分布:")
        for quality, count in quality_counts.items():
            print(f"   {quality}: {count} 个")
    
    # 失败分析
    failed_tests = [k for k, r in results.items() if not r['success']]
    if failed_tests:
        print(f"\n❌ 失败的测试:")
        for test_key in failed_tests:
            result = results[test_key]
            print(f"   {result['mask_model']} - {result['mask_name']}")
    
    return results

if __name__ == "__main__":
    results = test_all_mask_scenarios()