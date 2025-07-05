#!/usr/bin/env python3
"""
HD Strategy 快速验证脚本
快速测试三种HD策略的基本功能
"""

import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.inference import process_image
from config.config import ConfigManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickHDTester:
    """快速HD策略测试器"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.output_dir = Path("scripts/hd_quick_test")
        self.output_dir.mkdir(exist_ok=True)
        
        # 简化的测试配置
        self.test_sizes = [
            (800, 600),      # 中等尺寸 - 应该触发CROP
            (1024, 768),     # 标准尺寸 - 应该触发CROP
            (1920, 1080),    # 高分辨率 - 应该触发CROP
            (2048, 1536)     # 超高分辨率 - 应该触发CROP
        ]
        
        self.strategies = ['ORIGINAL', 'CROP', 'RESIZE']
        
    def create_simple_test_image(self, size: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
        """创建简单的测试图像和mask"""
        width, height = size
        
        # 创建测试图像
        image = Image.new('RGB', size, color=(200, 200, 200))
        draw = ImageDraw.Draw(image)
        
        # 添加简单的图案
        draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], 
                      fill=(100, 150, 200), outline=(50, 100, 150))
        
        # 添加水印区域（右下角）
        wm_w, wm_h = width//8, height//10
        wm_x, wm_y = width - wm_w - 50, height - wm_h - 50
        draw.rectangle([wm_x, wm_y, wm_x + wm_w, wm_y + wm_h], 
                      fill=(255, 255, 255), outline=(0, 0, 0))
        
        # 创建mask
        mask = Image.new('L', size, color=0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([wm_x, wm_y, wm_x + wm_w, wm_y + wm_h], fill=255)
        
        return image, mask
    
    def test_strategy(self, size: Tuple[int, int], strategy: str) -> Dict:
        """测试单个策略"""
        logger.info(f"🧪 测试 {size[0]}x{size[1]} - {strategy}")
        
        # 创建测试数据
        test_image, test_mask = self.create_simple_test_image(size)
        
        # 保存输入
        input_path = self.output_dir / f"input_{size[0]}x{size[1]}_{strategy}.png"
        mask_path = self.output_dir / f"mask_{size[0]}x{size[1]}_{strategy}.png"
        test_image.save(input_path)
        test_mask.save(mask_path)
        
        # 配置参数
        mask_params = {
            'uploaded_mask': test_mask,
            'mask_dilate_kernel_size': 1,
            'mask_dilate_iterations': 1
        }
        
        # 根据策略调整参数
        if strategy == 'ORIGINAL':
            # ORIGINAL应该保持原始尺寸
            crop_trigger = 99999
            resize_limit = 99999
        elif strategy == 'CROP':
            # CROP应该在图像超过阈值时分块处理
            crop_trigger = 600  # 较低阈值确保触发
            resize_limit = 2048
        else:  # RESIZE
            # RESIZE应该限制最大尺寸
            crop_trigger = 99999
            resize_limit = 1024  # 较低限制确保触发
        
        inpaint_params = {
            'inpaint_model': 'iopaint',
            'force_model': 'lama',  # 使用最快的模型
            'auto_model_selection': False,
            'ldm_steps': 10,  # 减少步数加快测试
            'hd_strategy': strategy,
            'hd_strategy_crop_margin': 64,
            'hd_strategy_crop_trigger_size': crop_trigger,
            'hd_strategy_resize_limit': resize_limit,
            'seed': 42
        }
        
        performance_params = {
            'mixed_precision': True,
            'log_processing_time': True
        }
        
        # 执行测试
        start_time = time.time()
        
        try:
            result = process_image(
                image=test_image,
                mask_model='upload',
                mask_params=mask_params,
                inpaint_params=inpaint_params,
                performance_params=performance_params,
                transparent=False,
                config_manager=self.config_manager
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result.success:
                # 保存结果
                result_path = self.output_dir / f"result_{size[0]}x{size[1]}_{strategy}.png"
                result.result_image.save(result_path)
                
                # 分析结果
                size_preserved = result.result_image.size == test_image.size
                
                test_result = {
                    'strategy': strategy,
                    'original_size': test_image.size,
                    'result_size': result.result_image.size,
                    'size_preserved': size_preserved,
                    'processing_time': processing_time,
                    'success': True,
                    'expected_behavior': self._get_expected_behavior(strategy, size, crop_trigger, resize_limit)
                }
                
                logger.info(f"✅ {strategy} 成功: {test_image.size} -> {result.result_image.size} ({processing_time:.2f}s)")
                
            else:
                test_result = {
                    'strategy': strategy,
                    'original_size': test_image.size,
                    'result_size': (0, 0),
                    'size_preserved': False,
                    'processing_time': processing_time,
                    'success': False,
                    'error': result.error_message
                }
                
                logger.error(f"❌ {strategy} 失败: {result.error_message}")
                
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            test_result = {
                'strategy': strategy,
                'original_size': test_image.size,
                'result_size': (0, 0),
                'size_preserved': False,
                'processing_time': processing_time,
                'success': False,
                'error': str(e)
            }
            
            logger.error(f"❌ {strategy} 异常: {str(e)}")
        
        return test_result
    
    def _get_expected_behavior(self, strategy: str, size: Tuple[int, int], crop_trigger: int, resize_limit: int) -> str:
        """获取期望行为描述"""
        width, height = size
        max_dim = max(width, height)
        
        if strategy == 'ORIGINAL':
            return f"保持原始尺寸 {size}"
        elif strategy == 'CROP':
            if max_dim > crop_trigger:
                return f"分块处理，最终合成为原始尺寸 {size}"
            else:
                return f"直接处理，保持原始尺寸 {size}"
        else:  # RESIZE
            if max_dim > resize_limit:
                scale = resize_limit / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                return f"缩放到 {new_width}x{new_height} 以内"
            else:
                return f"保持原始尺寸 {size}"
    
    def run_quick_test(self) -> Dict:
        """运行快速测试"""
        logger.info("🚀 开始HD策略快速测试")
        
        results = {}
        
        for size in self.test_sizes:
            results[f"{size[0]}x{size[1]}"] = {}
            
            for strategy in self.strategies:
                test_result = self.test_strategy(size, strategy)
                results[f"{size[0]}x{size[1]}"][strategy] = test_result
        
        return results
    
    def analyze_results(self, results: Dict) -> None:
        """分析测试结果"""
        logger.info("\n📊 分析测试结果")
        
        print("\n" + "="*80)
        print("HD Strategy 快速测试结果")
        print("="*80)
        
        for size_key, size_results in results.items():
            print(f"\n📏 {size_key}:")
            
            for strategy, result in size_results.items():
                if result['success']:
                    status = "✅"
                    size_status = "✅" if result['size_preserved'] else "⚠️"
                    details = f"{result['original_size']} -> {result['result_size']}"
                    time_info = f"({result['processing_time']:.2f}s)"
                    expected = result['expected_behavior']
                    
                    print(f"  {strategy:>8}: {status} {details} {size_status} {time_info}")
                    print(f"           期望: {expected}")
                else:
                    print(f"  {strategy:>8}: ❌ {result.get('error', 'Unknown error')}")
        
        print("\n💡 策略行为分析:")
        print("-"*40)
        
        # 分析ORIGINAL策略
        original_results = []
        for size_results in results.values():
            if 'ORIGINAL' in size_results:
                original_results.append(size_results['ORIGINAL'])
        
        original_preserved = sum(1 for r in original_results if r['success'] and r['size_preserved'])
        if original_preserved == len(original_results):
            print("✅ ORIGINAL策略正确：所有测试都保持了原始尺寸")
        else:
            print(f"❌ ORIGINAL策略异常：{original_preserved}/{len(original_results)} 保持了原始尺寸")
        
        # 分析CROP策略
        crop_results = []
        for size_results in results.values():
            if 'CROP' in size_results:
                crop_results.append(size_results['CROP'])
        
        crop_preserved = sum(1 for r in crop_results if r['success'] and r['size_preserved'])
        if crop_preserved == len(crop_results):
            print("✅ CROP策略正确：所有测试都保持了原始尺寸（分块处理后合成）")
        else:
            print(f"❌ CROP策略异常：{crop_preserved}/{len(crop_results)} 保持了原始尺寸")
        
        # 分析RESIZE策略
        resize_results = []
        for size_results in results.values():
            if 'RESIZE' in size_results:
                resize_results.append(size_results['RESIZE'])
        
        resize_changed = sum(1 for r in resize_results if r['success'] and not r['size_preserved'])
        if resize_changed > 0:
            print(f"✅ RESIZE策略正确：{resize_changed}/{len(resize_results)} 改变了尺寸")
        else:
            print(f"⚠️  RESIZE策略可能异常：没有测试改变了尺寸")
        
        print("\n"+"="*80)

def main():
    """主函数"""
    logger.info("🔍 HD策略快速验证开始")
    
    # 创建测试器
    tester = QuickHDTester()
    
    # 运行测试
    try:
        results = tester.run_quick_test()
        
        # 分析结果
        tester.analyze_results(results)
        
        # 判断是否通过
        total_tests = 0
        successful_tests = 0
        
        for size_results in results.values():
            for result in size_results.values():
                total_tests += 1
                if result['success']:
                    successful_tests += 1
        
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.8:
            logger.info("🎉 HD策略快速测试通过！")
            return True
        else:
            logger.warning(f"⚠️  HD策略测试成功率较低: {success_rate:.2%}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)