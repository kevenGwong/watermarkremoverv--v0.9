#!/usr/bin/env python3
"""
高清修复验证脚本
测试PowerPaint Object Removal的分辨率保持功能
"""

import logging
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from core.inference import InferenceManager
from config.config import ConfigManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_image_and_mask(width: int, height: int):
    """创建测试图像和mask"""
    # 创建彩色测试图像
    image = Image.new('RGB', (width, height), color=(100, 150, 200))
    
    # 在图像中央添加一些细节
    pixels = image.load()
    for x in range(width // 4, 3 * width // 4):
        for y in range(height // 4, 3 * height // 4):
            pixels[x, y] = (200, 100, 50)
    
    # 创建简单mask（中央区域）
    mask = Image.new('L', (width, height), color=0)
    mask_pixels = mask.load()
    mask_w, mask_h = width // 6, height // 6
    start_x, start_y = width // 2 - mask_w, height // 2 - mask_h
    
    for x in range(start_x, start_x + 2 * mask_w):
        for y in range(start_y, start_y + 2 * mask_h):
            if 0 <= x < width and 0 <= y < height:
                mask_pixels[x, y] = 255
    
    return image, mask

def test_resolution_preservation():
    """测试分辨率保持功能"""
    logger.info("🧪 开始高清修复测试")
    
    # 测试不同分辨率
    test_resolutions = [
        (800, 600),    # 中等分辨率
        (1200, 800),   # 高分辨率
        (1920, 1080),  # Full HD
        (2048, 1536),  # 2K+
    ]
    
    try:
        # 初始化推理管理器
        config_manager = ConfigManager()
        inference_manager = InferenceManager(config_manager.get_config())
        
        results = []
        
        for original_width, original_height in test_resolutions:
            logger.info(f"\n🎯 测试分辨率: {original_width}x{original_height}")
            
            # 创建测试数据
            image, mask = create_test_image_and_mask(original_width, original_height)
            
            # 保存测试输入
            input_dir = Path("data/temp/test_input")
            input_dir.mkdir(parents=True, exist_ok=True)
            
            image.save(input_dir / f"test_{original_width}x{original_height}.png")
            mask.save(input_dir / f"test_mask_{original_width}x{original_height}.png")
            
            # 测试参数
            test_params = {
                'inpaint_model': 'powerpaint',
                'task': 'object-removal',
                'prompt': '',  # Object removal不需要prompt
                'num_inference_steps': 20,  # 快速测试
                'crop_trigger_size': 1024,
                'preserve_original_resolution': True,
                'high_quality_resize': True
            }
            
            try:
                # 执行处理
                logger.info("🚀 开始PowerPaint处理...")
                result = inference_manager.process_image(
                    image=np.array(image),
                    mask=np.array(mask),
                    custom_config=test_params
                )
                
                # 检查结果分辨率
                result_image = Image.fromarray(result)
                result_width, result_height = result_image.size
                
                resolution_preserved = (
                    result_width == original_width and 
                    result_height == original_height
                )
                
                test_result = {
                    'original_size': (original_width, original_height),
                    'result_size': (result_width, result_height),
                    'resolution_preserved': resolution_preserved,
                    'success': True
                }
                
                # 保存结果
                output_dir = Path("data/temp/test_output")
                output_dir.mkdir(parents=True, exist_ok=True)
                result_image.save(
                    output_dir / f"result_{original_width}x{original_height}.png"
                )
                
                status = "✅ 成功" if resolution_preserved else "❌ 分辨率改变"
                logger.info(f"{status}: {original_width}x{original_height} -> {result_width}x{result_height}")
                
            except Exception as e:
                logger.error(f"❌ 处理失败: {e}")
                test_result = {
                    'original_size': (original_width, original_height),
                    'result_size': None,
                    'resolution_preserved': False,
                    'success': False,
                    'error': str(e)
                }
            
            results.append(test_result)
        
        # 总结结果
        logger.info("\n📊 测试总结:")
        successful_tests = sum(1 for r in results if r['success'])
        preserved_resolutions = sum(1 for r in results if r['resolution_preserved'])
        
        logger.info(f"总测试数: {len(results)}")
        logger.info(f"成功执行: {successful_tests}")
        logger.info(f"分辨率保持: {preserved_resolutions}")
        
        for result in results:
            if result['success']:
                status = "✅" if result['resolution_preserved'] else "⚠️"
                logger.info(f"{status} {result['original_size']} -> {result['result_size']}")
            else:
                logger.info(f"❌ {result['original_size']}: {result.get('error', 'Unknown error')}")
        
        # 判断修复是否成功
        fix_successful = preserved_resolutions == successful_tests and successful_tests > 0
        
        if fix_successful:
            logger.info("\n🎉 高清修复验证成功！所有处理都保持了原始分辨率")
            return True
        else:
            logger.warning("\n⚠️ 高清修复可能仍有问题，部分测试未保持原始分辨率")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    success = test_resolution_preservation()
    sys.exit(0 if success else 1)