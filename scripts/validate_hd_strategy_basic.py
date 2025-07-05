#!/usr/bin/env python3
"""
HD Strategy 基础验证脚本
验证IOPaint HD策略的基本功能
"""

import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_iopaint_schema():
    """测试IOPaint schema导入"""
    try:
        from iopaint.schema import HDStrategy, InpaintRequest, LDMSampler
        
        print("✅ IOPaint导入成功")
        print(f"HD策略选项: {[attr for attr in dir(HDStrategy) if not attr.startswith('_')]}")
        
        # 测试配置创建
        config = InpaintRequest(
            ldm_steps=20,
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.ORIGINAL,
            hd_strategy_crop_margin=64,
            hd_strategy_crop_trigger_size=1024,
            hd_strategy_resize_limit=2048
        )
        
        print("✅ IOPaint配置创建成功")
        print(f"配置详情: hd_strategy={config.hd_strategy}, crop_trigger={config.hd_strategy_crop_trigger_size}")
        
        return True
        
    except ImportError as e:
        print(f"❌ IOPaint导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ IOPaint配置失败: {e}")
        return False

def test_model_manager():
    """测试IOPaint ModelManager"""
    try:
        from iopaint.model_manager import ModelManager
        from iopaint.schema import HDStrategy, InpaintRequest, LDMSampler
        
        print("📦 测试ModelManager创建...")
        
        # 创建LaMA模型（最快的模型）
        device = "cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu"
        model_manager = ModelManager(name="lama", device=device)
        
        print(f"✅ ModelManager创建成功 (device: {device})")
        
        # 创建测试图像和mask
        test_size = (512, 512)
        test_image = np.random.randint(0, 255, (*test_size, 3), dtype=np.uint8)
        test_mask = np.zeros(test_size, dtype=np.uint8)
        test_mask[200:300, 200:300] = 255  # 中央区域
        
        print(f"📷 测试图像: {test_image.shape}, mask: {test_mask.shape}")
        
        # 测试不同HD策略
        strategies = ['ORIGINAL', 'CROP', 'RESIZE']
        results = {}
        
        for strategy in strategies:
            print(f"\n🧪 测试 {strategy} 策略...")
            
            if strategy == 'ORIGINAL':
                config = InpaintRequest(
                    ldm_steps=10,
                    hd_strategy=HDStrategy.ORIGINAL,
                    hd_strategy_crop_margin=64,
                    hd_strategy_crop_trigger_size=99999,  # 永不触发
                    hd_strategy_resize_limit=99999
                )
            elif strategy == 'CROP':
                config = InpaintRequest(
                    ldm_steps=10,
                    hd_strategy=HDStrategy.CROP,
                    hd_strategy_crop_margin=64,
                    hd_strategy_crop_trigger_size=400,  # 较低阈值
                    hd_strategy_resize_limit=2048
                )
            else:  # RESIZE
                config = InpaintRequest(
                    ldm_steps=10,
                    hd_strategy=HDStrategy.RESIZE,
                    hd_strategy_crop_margin=64,
                    hd_strategy_crop_trigger_size=99999,
                    hd_strategy_resize_limit=400  # 较低限制
                )
            
            start_time = time.time()
            
            try:
                result = model_manager(test_image, test_mask, config)
                processing_time = time.time() - start_time
                
                results[strategy] = {
                    'success': True,
                    'input_size': test_image.shape[:2],
                    'output_size': result.shape[:2],
                    'size_preserved': result.shape[:2] == test_image.shape[:2],
                    'processing_time': processing_time
                }
                
                print(f"   ✅ {strategy}: {test_image.shape[:2]} -> {result.shape[:2]} ({processing_time:.2f}s)")
                
            except Exception as e:
                results[strategy] = {
                    'success': False,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
                print(f"   ❌ {strategy}: {str(e)}")
        
        return results
        
    except ImportError as e:
        print(f"❌ ModelManager导入失败: {e}")
        return None
    except Exception as e:
        print(f"❌ ModelManager测试失败: {e}")
        return None

def test_our_processor():
    """测试我们的IOPaint处理器"""
    try:
        from core.models.iopaint_processor import IOPaintProcessor
        from config.config import ConfigManager
        
        print("🔧 测试我们的IOPaint处理器...")
        
        # 创建配置
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # 创建处理器
        processor = IOPaintProcessor(config)
        
        print("✅ IOPaint处理器创建成功")
        
        # 创建测试数据
        test_image = Image.new('RGB', (800, 600), color=(200, 200, 200))
        test_mask = Image.new('L', (800, 600), color=0)
        
        # 添加一些内容到测试图像
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([200, 200, 400, 400], fill=(100, 150, 200))
        
        # 添加mask区域
        mask_draw = ImageDraw.Draw(test_mask)
        mask_draw.rectangle([300, 300, 350, 350], fill=255)
        
        # 测试不同策略
        strategies = ['ORIGINAL', 'CROP', 'RESIZE']
        results = {}
        
        for strategy in strategies:
            print(f"\n🧪 测试处理器 {strategy} 策略...")
            
            if strategy == 'ORIGINAL':
                custom_config = {
                    'hd_strategy': 'ORIGINAL',
                    'hd_strategy_crop_trigger_size': 99999,
                    'hd_strategy_resize_limit': 99999,
                    'ldm_steps': 5
                }
            elif strategy == 'CROP':
                custom_config = {
                    'hd_strategy': 'CROP',
                    'hd_strategy_crop_trigger_size': 500,  # 较低阈值
                    'hd_strategy_resize_limit': 2048,
                    'ldm_steps': 5
                }
            else:  # RESIZE
                custom_config = {
                    'hd_strategy': 'RESIZE',
                    'hd_strategy_crop_trigger_size': 99999,
                    'hd_strategy_resize_limit': 600,  # 较低限制
                    'ldm_steps': 5
                }
            
            start_time = time.time()
            
            try:
                result = processor.predict(test_image, test_mask, custom_config)
                processing_time = time.time() - start_time
                
                result_image = Image.fromarray(result)
                
                results[strategy] = {
                    'success': True,
                    'input_size': test_image.size,
                    'output_size': result_image.size,
                    'size_preserved': result_image.size == test_image.size,
                    'processing_time': processing_time
                }
                
                print(f"   ✅ {strategy}: {test_image.size} -> {result_image.size} ({processing_time:.2f}s)")
                
                # 保存结果用于检查
                output_dir = Path("scripts/hd_validation_output")
                output_dir.mkdir(exist_ok=True)
                
                test_image.save(output_dir / f"input_{strategy.lower()}.png")
                test_mask.save(output_dir / f"mask_{strategy.lower()}.png")
                result_image.save(output_dir / f"result_{strategy.lower()}.png")
                
            except Exception as e:
                results[strategy] = {
                    'success': False,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
                print(f"   ❌ {strategy}: {str(e)}")
        
        return results
        
    except ImportError as e:
        print(f"❌ 处理器导入失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 处理器测试失败: {e}")
        return None

def analyze_results(schema_ok, model_results, processor_results):
    """分析测试结果"""
    print("\n" + "="*60)
    print("HD Strategy 基础验证结果")
    print("="*60)
    
    # Schema测试结果
    print(f"📋 IOPaint Schema: {'✅ 正常' if schema_ok else '❌ 失败'}")
    
    # ModelManager测试结果
    if model_results:
        print("\n🔧 ModelManager测试:")
        for strategy, result in model_results.items():
            if result['success']:
                status = "✅"
                size_status = "✅" if result['size_preserved'] else "⚠️"
                print(f"  {strategy:>8}: {status} {result['input_size']} -> {result['output_size']} {size_status}")
            else:
                print(f"  {strategy:>8}: ❌ {result['error']}")
    else:
        print("\n🔧 ModelManager测试: ❌ 失败")
    
    # 处理器测试结果
    if processor_results:
        print("\n⚙️  处理器测试:")
        for strategy, result in processor_results.items():
            if result['success']:
                status = "✅"
                size_status = "✅" if result['size_preserved'] else "⚠️"
                print(f"  {strategy:>8}: {status} {result['input_size']} -> {result['output_size']} {size_status}")
            else:
                print(f"  {strategy:>8}: ❌ {result['error']}")
    else:
        print("\n⚙️  处理器测试: ❌ 失败")
    
    # 策略行为分析
    print("\n💡 策略行为分析:")
    print("-" * 30)
    
    if processor_results:
        # ORIGINAL策略
        original = processor_results.get('ORIGINAL', {})
        if original.get('success'):
            if original['size_preserved']:
                print("✅ ORIGINAL策略正确：保持了原始尺寸")
            else:
                print("❌ ORIGINAL策略异常：未保持原始尺寸")
        else:
            print("❌ ORIGINAL策略失败")
        
        # CROP策略
        crop = processor_results.get('CROP', {})
        if crop.get('success'):
            if crop['size_preserved']:
                print("✅ CROP策略正确：分块处理后保持了原始尺寸")
            else:
                print("❌ CROP策略异常：未保持原始尺寸")
        else:
            print("❌ CROP策略失败")
        
        # RESIZE策略
        resize = processor_results.get('RESIZE', {})
        if resize.get('success'):
            if not resize['size_preserved']:
                print("✅ RESIZE策略正确：改变了图像尺寸")
            else:
                print("⚠️  RESIZE策略可能异常：未改变图像尺寸")
        else:
            print("❌ RESIZE策略失败")
    
    # 总体评估
    print("\n🎯 总体评估:")
    print("-" * 30)
    
    total_checks = 3
    passed_checks = 0
    
    if schema_ok:
        passed_checks += 1
    
    if model_results and any(r.get('success', False) for r in model_results.values()):
        passed_checks += 1
    
    if processor_results and any(r.get('success', False) for r in processor_results.values()):
        passed_checks += 1
    
    success_rate = passed_checks / total_checks
    
    if success_rate >= 0.8:
        print("🎉 HD策略基础功能正常")
        return True
    elif success_rate >= 0.5:
        print("⚠️  HD策略部分功能正常，需要进一步检查")
        return False
    else:
        print("❌ HD策略功能异常，需要修复")
        return False

def main():
    """主函数"""
    print("🔍 HD Strategy 基础验证开始")
    print("=" * 50)
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 1. 测试IOPaint schema
    print("\n1️⃣  测试IOPaint Schema...")
    schema_ok = test_iopaint_schema()
    
    # 2. 测试ModelManager
    print("\n2️⃣  测试IOPaint ModelManager...")
    model_results = test_model_manager()
    
    # 3. 测试我们的处理器
    print("\n3️⃣  测试我们的处理器...")
    processor_results = test_our_processor()
    
    # 4. 分析结果
    success = analyze_results(schema_ok, model_results, processor_results)
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        sys.exit(1)