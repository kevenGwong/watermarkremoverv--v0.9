#!/usr/bin/env python3
"""
HD Strategy 高清处理策略全面测试脚本
验证ORIGINAL、CROP、RESIZE三种模式的正确性
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.inference import process_image
from config.config import ConfigManager
from core.utils.image_utils import ImageValidator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    original_size: Tuple[int, int]
    result_size: Tuple[int, int]
    hd_strategy: str
    processing_time: float
    success: bool
    size_preserved: bool
    quality_score: float
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'original_size': self.original_size,
            'result_size': self.result_size,
            'hd_strategy': self.hd_strategy,
            'processing_time': self.processing_time,
            'success': self.success,
            'size_preserved': self.size_preserved,
            'quality_score': self.quality_score,
            'error_message': self.error_message
        }

class HDStrategyTester:
    """HD Strategy测试器"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.test_results: List[TestResult] = []
        self.output_dir = Path("scripts/hd_strategy_test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 测试策略配置
        self.strategies = ['ORIGINAL', 'CROP', 'RESIZE']
        self.test_sizes = [
            (512, 512),      # 小尺寸
            (800, 600),      # 中等尺寸
            (1024, 768),     # 标准分辨率
            (1280, 720),     # 720p
            (1920, 1080),    # 1080p
            (2048, 1536),    # 2K
            (2560, 1440),    # 1440p
            (3840, 2160)     # 4K
        ]
        
        # 测试模型（选择处理速度较快的）
        self.test_models = ['fcf', 'lama']
        
    def create_test_image_with_watermark(self, size: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
        """创建带水印的测试图像和对应的mask"""
        width, height = size
        
        # 创建复杂的测试图像
        image = Image.new('RGB', size, color=(240, 240, 240))
        draw = ImageDraw.Draw(image)
        
        # 添加渐变背景
        for y in range(height):
            color_val = int(200 + 55 * (y / height))
            draw.line([(0, y), (width, y)], fill=(color_val, color_val - 20, color_val - 40))
        
        # 添加几何图形
        draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], 
                      fill=(100, 150, 200), outline=(50, 100, 150), width=3)
        
        # 添加圆形
        circle_size = min(width, height) // 8
        draw.ellipse([width//2 - circle_size, height//2 - circle_size,
                     width//2 + circle_size, height//2 + circle_size],
                    fill=(255, 100, 100), outline=(200, 50, 50), width=2)
        
        # 添加文本
        try:
            font = ImageFont.load_default()
            text = f"Test Image {width}x{height}"
            draw.text((20, 20), text, fill=(0, 0, 0), font=font)
        except:
            # 如果字体加载失败，使用默认字体
            draw.text((20, 20), f"Test {width}x{height}", fill=(0, 0, 0))
        
        # 创建水印区域（右下角）
        watermark_w, watermark_h = width // 6, height // 8
        watermark_x = width - watermark_w - 20
        watermark_y = height - watermark_h - 20
        
        # 半透明水印
        watermark = Image.new('RGBA', (watermark_w, watermark_h), (255, 255, 255, 128))
        watermark_draw = ImageDraw.Draw(watermark)
        watermark_draw.rectangle([0, 0, watermark_w-1, watermark_h-1], 
                               fill=(200, 200, 200, 180), outline=(100, 100, 100, 255))
        try:
            watermark_draw.text((10, 10), "WATERMARK", fill=(0, 0, 0, 200))
        except:
            pass
        
        # 粘贴水印
        image.paste(watermark, (watermark_x, watermark_y), watermark)
        
        # 创建对应的mask
        mask = Image.new('L', size, color=0)
        mask_draw = ImageDraw.Draw(mask)
        # 水印区域设为白色
        mask_draw.rectangle([watermark_x, watermark_y, 
                           watermark_x + watermark_w, watermark_y + watermark_h],
                          fill=255)
        
        return image, mask
    
    def calculate_image_quality_score(self, original: Image.Image, result: Image.Image) -> float:
        """计算图像质量分数（基于PSNR）"""
        try:
            # 确保两个图像尺寸相同
            if original.size != result.size:
                result = result.resize(original.size, Image.LANCZOS)
            
            # 转换为numpy数组
            orig_array = np.array(original.convert('RGB'))
            result_array = np.array(result.convert('RGB'))
            
            # 计算MSE
            mse = np.mean((orig_array - result_array) ** 2)
            
            # 避免除零错误
            if mse == 0:
                return 100.0
            
            # 计算PSNR
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # 归一化到0-100分数
            quality_score = min(100.0, max(0.0, psnr))
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"无法计算图像质量分数: {e}")
            return 0.0
    
    def test_single_configuration(self, 
                                 size: Tuple[int, int], 
                                 strategy: str, 
                                 model: str) -> TestResult:
        """测试单一配置"""
        test_name = f"{size[0]}x{size[1]}_{strategy}_{model}"
        logger.info(f"🧪 测试配置: {test_name}")
        
        # 创建测试数据
        original_image, test_mask = self.create_test_image_with_watermark(size)
        
        # 保存原始图像（用于对比）
        original_path = self.output_dir / f"{test_name}_original.png"
        original_image.save(original_path)
        
        # 保存mask
        mask_path = self.output_dir / f"{test_name}_mask.png"
        test_mask.save(mask_path)
        
        # 设置处理参数
        mask_params = {
            'uploaded_mask': test_mask,
            'mask_dilate_kernel_size': 1,
            'mask_dilate_iterations': 1
        }
        
        # 根据策略设置参数
        if strategy == 'ORIGINAL':
            crop_trigger = 99999  # 足够大，永远不会触发
            resize_limit = 99999
        elif strategy == 'CROP':
            crop_trigger = 800    # 较小的触发尺寸
            resize_limit = 2048
        else:  # RESIZE
            crop_trigger = 99999  # 不触发crop
            resize_limit = 1024   # 较小的resize限制
        
        inpaint_params = {
            'inpaint_model': 'iopaint',
            'force_model': model,
            'auto_model_selection': False,
            'ldm_steps': 20,
            'hd_strategy': strategy,
            'hd_strategy_crop_margin': 64,
            'hd_strategy_crop_trigger_size': crop_trigger,
            'hd_strategy_resize_limit': resize_limit,
            'seed': 42  # 固定随机种子
        }
        
        performance_params = {
            'mixed_precision': True,
            'log_processing_time': True
        }
        
        # 执行处理
        start_time = time.time()
        
        try:
            result = process_image(
                image=original_image,
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
                result_path = self.output_dir / f"{test_name}_result.png"
                result.result_image.save(result_path)
                
                # 检查尺寸保持
                size_preserved = result.result_image.size == original_image.size
                
                # 计算质量分数
                quality_score = self.calculate_image_quality_score(original_image, result.result_image)
                
                test_result = TestResult(
                    test_name=test_name,
                    original_size=original_image.size,
                    result_size=result.result_image.size,
                    hd_strategy=strategy,
                    processing_time=processing_time,
                    success=True,
                    size_preserved=size_preserved,
                    quality_score=quality_score
                )
                
                logger.info(f"✅ {test_name} 成功 - 尺寸保持: {size_preserved}, 质量分数: {quality_score:.2f}")
                
            else:
                test_result = TestResult(
                    test_name=test_name,
                    original_size=original_image.size,
                    result_size=(0, 0),
                    hd_strategy=strategy,
                    processing_time=processing_time,
                    success=False,
                    size_preserved=False,
                    quality_score=0.0,
                    error_message=result.error_message
                )
                
                logger.error(f"❌ {test_name} 失败: {result.error_message}")
                
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            test_result = TestResult(
                test_name=test_name,
                original_size=original_image.size,
                result_size=(0, 0),
                hd_strategy=strategy,
                processing_time=processing_time,
                success=False,
                size_preserved=False,
                quality_score=0.0,
                error_message=str(e)
            )
            
            logger.error(f"❌ {test_name} 异常: {str(e)}")
        
        return test_result
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """运行全面的HD策略测试"""
        logger.info("🚀 开始HD策略全面测试")
        logger.info(f"测试配置: {len(self.test_sizes)} 种尺寸 × {len(self.strategies)} 种策略 × {len(self.test_models)} 种模型")
        
        # 创建测试矩阵
        test_configurations = []
        for size in self.test_sizes:
            for strategy in self.strategies:
                for model in self.test_models:
                    test_configurations.append((size, strategy, model))
        
        logger.info(f"总测试数量: {len(test_configurations)}")
        
        # 执行测试
        for i, (size, strategy, model) in enumerate(test_configurations, 1):
            logger.info(f"\n📊 进度: {i}/{len(test_configurations)}")
            
            test_result = self.test_single_configuration(size, strategy, model)
            self.test_results.append(test_result)
            
            # 内存清理
            if i % 10 == 0:
                logger.info("🧹 执行内存清理...")
                try:
                    import gc
                    gc.collect()
                    if 'torch' in sys.modules:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except:
                    pass
        
        # 分析结果
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析测试结果"""
        logger.info("\n📈 分析测试结果...")
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        # 按策略分组分析
        strategy_stats = {}
        for strategy in self.strategies:
            strategy_results = [r for r in self.test_results if r.hd_strategy == strategy]
            successful_strategy = [r for r in strategy_results if r.success]
            
            if successful_strategy:
                avg_time = sum(r.processing_time for r in successful_strategy) / len(successful_strategy)
                avg_quality = sum(r.quality_score for r in successful_strategy) / len(successful_strategy)
                size_preserved_rate = sum(1 for r in successful_strategy if r.size_preserved) / len(successful_strategy)
            else:
                avg_time = 0
                avg_quality = 0
                size_preserved_rate = 0
            
            strategy_stats[strategy] = {
                'total_tests': len(strategy_results),
                'successful_tests': len(successful_strategy),
                'success_rate': len(successful_strategy) / len(strategy_results) if strategy_results else 0,
                'avg_processing_time': avg_time,
                'avg_quality_score': avg_quality,
                'size_preserved_rate': size_preserved_rate
            }
        
        # 按尺寸分组分析
        size_stats = {}
        for size in self.test_sizes:
            size_results = [r for r in self.test_results if r.original_size == size]
            successful_size = [r for r in size_results if r.success]
            
            if successful_size:
                avg_time = sum(r.processing_time for r in successful_size) / len(successful_size)
                avg_quality = sum(r.quality_score for r in successful_size) / len(successful_size)
                size_preserved_rate = sum(1 for r in successful_size if r.size_preserved) / len(successful_size)
            else:
                avg_time = 0
                avg_quality = 0
                size_preserved_rate = 0
            
            size_stats[f"{size[0]}x{size[1]}"] = {
                'total_tests': len(size_results),
                'successful_tests': len(successful_size),
                'success_rate': len(successful_size) / len(size_results) if size_results else 0,
                'avg_processing_time': avg_time,
                'avg_quality_score': avg_quality,
                'size_preserved_rate': size_preserved_rate
            }
        
        # 问题分析
        failed_tests = [r for r in self.test_results if not r.success]
        size_changed_tests = [r for r in self.test_results if r.success and not r.size_preserved]
        
        analysis = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'strategy_stats': strategy_stats,
            'size_stats': size_stats,
            'failed_tests': len(failed_tests),
            'size_changed_tests': len(size_changed_tests),
            'test_results': [r.to_dict() for r in self.test_results]
        }
        
        # 保存分析结果
        analysis_path = self.output_dir / "analysis_results.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 分析结果已保存到: {analysis_path}")
        
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """生成测试报告"""
        logger.info("📝 生成测试报告...")
        
        report = []
        report.append("=" * 80)
        report.append("HD Strategy 高清处理策略测试报告")
        report.append("=" * 80)
        report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试配置: {len(self.test_sizes)} 种尺寸 × {len(self.strategies)} 种策略 × {len(self.test_models)} 种模型")
        report.append("")
        
        # 总体统计
        report.append("📊 总体统计")
        report.append("-" * 40)
        report.append(f"总测试数: {analysis['total_tests']}")
        report.append(f"成功测试数: {analysis['successful_tests']}")
        report.append(f"成功率: {analysis['success_rate']:.2%}")
        report.append(f"失败测试数: {analysis['failed_tests']}")
        report.append(f"尺寸变化测试数: {analysis['size_changed_tests']}")
        report.append("")
        
        # 策略对比
        report.append("🎯 HD策略对比")
        report.append("-" * 40)
        for strategy, stats in analysis['strategy_stats'].items():
            report.append(f"{strategy} 策略:")
            report.append(f"  成功率: {stats['success_rate']:.2%} ({stats['successful_tests']}/{stats['total_tests']})")
            report.append(f"  平均处理时间: {stats['avg_processing_time']:.2f}秒")
            report.append(f"  平均质量分数: {stats['avg_quality_score']:.2f}")
            report.append(f"  尺寸保持率: {stats['size_preserved_rate']:.2%}")
            report.append("")
        
        # 尺寸分析
        report.append("📏 尺寸分析")
        report.append("-" * 40)
        for size_key, stats in analysis['size_stats'].items():
            report.append(f"{size_key}:")
            report.append(f"  成功率: {stats['success_rate']:.2%} ({stats['successful_tests']}/{stats['total_tests']})")
            report.append(f"  平均处理时间: {stats['avg_processing_time']:.2f}秒")
            report.append(f"  平均质量分数: {stats['avg_quality_score']:.2f}")
            report.append(f"  尺寸保持率: {stats['size_preserved_rate']:.2%}")
            report.append("")
        
        # 问题分析
        if analysis['failed_tests'] > 0:
            report.append("❌ 失败测试分析")
            report.append("-" * 40)
            failed_results = [r for r in self.test_results if not r.success]
            for result in failed_results:
                report.append(f"  {result.test_name}: {result.error_message}")
            report.append("")
        
        if analysis['size_changed_tests'] > 0:
            report.append("⚠️  尺寸变化测试分析")
            report.append("-" * 40)
            size_changed_results = [r for r in self.test_results if r.success and not r.size_preserved]
            for result in size_changed_results:
                report.append(f"  {result.test_name}: {result.original_size} -> {result.result_size}")
            report.append("")
        
        # 建议
        report.append("💡 建议")
        report.append("-" * 40)
        
        # 分析最佳策略
        best_strategy = max(analysis['strategy_stats'].items(), 
                          key=lambda x: x[1]['success_rate'] * x[1]['size_preserved_rate'])
        report.append(f"最佳策略: {best_strategy[0]} (成功率: {best_strategy[1]['success_rate']:.2%}, 尺寸保持: {best_strategy[1]['size_preserved_rate']:.2%})")
        
        # ORIGINAL策略分析
        original_stats = analysis['strategy_stats'].get('ORIGINAL', {})
        if original_stats.get('size_preserved_rate', 0) < 1.0:
            report.append("⚠️  ORIGINAL策略未能100%保持尺寸，需要检查实现")
        
        # CROP策略分析
        crop_stats = analysis['strategy_stats'].get('CROP', {})
        if crop_stats.get('success_rate', 0) < 0.9:
            report.append("⚠️  CROP策略成功率偏低，需要优化分块逻辑")
        
        # RESIZE策略分析
        resize_stats = analysis['strategy_stats'].get('RESIZE', {})
        if resize_stats.get('size_preserved_rate', 0) > 0.1:
            report.append("⚠️  RESIZE策略应该会改变尺寸，但部分测试保持了原尺寸")
        
        report.append("")
        report.append("=" * 80)
        
        report_content = "\n".join(report)
        
        # 保存报告
        report_path = self.output_dir / "test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📄 测试报告已保存到: {report_path}")
        
        return report_content

def main():
    """主函数"""
    logger.info("🔍 HD Strategy 高清处理策略测试开始")
    
    # 检查CUDA可用性
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA可用，GPU: {torch.cuda.get_device_name()}")
        else:
            logger.warning("⚠️  CUDA不可用，将使用CPU处理")
    except ImportError:
        logger.warning("⚠️  PyTorch未安装，某些功能可能受限")
    
    # 创建测试器
    tester = HDStrategyTester()
    
    # 运行测试
    try:
        analysis = tester.run_comprehensive_tests()
        
        # 生成报告
        report = tester.generate_report(analysis)
        
        # 打印报告
        print("\n" + report)
        
        # 判断测试是否通过
        success_rate = analysis['success_rate']
        size_preservation_rate = analysis['strategy_stats'].get('ORIGINAL', {}).get('size_preserved_rate', 0)
        
        if success_rate >= 0.9 and size_preservation_rate >= 0.95:
            logger.info("🎉 HD Strategy测试通过！")
            return True
        else:
            logger.warning("⚠️  HD Strategy测试未完全通过，需要进一步优化")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)