#!/usr/bin/env python3
"""
简化的CUDA内存管理测试
直接测试内存监控和基本功能
"""

import sys
import os
import time
import torch
from pathlib import Path
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_memory_monitor():
    """测试内存监控功能"""
    try:
        from core.utils.memory_monitor import MemoryMonitor
        
        logger.info("🧪 测试内存监控功能...")
        
        monitor = MemoryMonitor()
        
        # 测试基本功能
        memory_info = monitor.get_memory_info()
        logger.info(f"📊 当前内存状态: {memory_info}")
        
        # 测试CUDA内存监控
        if torch.cuda.is_available():
            logger.info("✅ CUDA可用，测试GPU内存监控")
            
            # 分配一些内存
            test_tensor = torch.randn(100, 100, device='cuda')
            memory_after = monitor.get_memory_info()
            logger.info(f"📊 分配后内存: {memory_after}")
            
            # 清理内存
            del test_tensor
            torch.cuda.empty_cache()
            memory_cleaned = monitor.get_memory_info()
            logger.info(f"📊 清理后内存: {memory_cleaned}")
            
        else:
            logger.warning("⚠️ CUDA不可用，跳过GPU内存测试")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 内存监控测试失败: {e}")
        return False

def test_base_inpainter():
    """测试基础inpainter类"""
    try:
        from core.models.base_inpainter import BaseInpainter, ModelRegistry
        
        logger.info("🧪 测试BaseInpainter和ModelRegistry...")
        
        # 测试模型注册表
        available_models = ModelRegistry.get_available_models()
        logger.info(f"📋 可用模型: {available_models}")
        
        # 导入模型以触发注册
        from core.models.mat_processor import MatProcessor
        from core.models.zits_processor import ZitsProcessor
        from core.models.fcf_processor import FcfProcessor
        
        # 再次检查注册结果
        available_models = ModelRegistry.get_available_models()
        logger.info(f"📋 注册后可用模型: {available_models}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BaseInpainter测试失败: {e}")
        return False

def test_simplified_processor():
    """测试简化处理器的核心功能"""
    try:
        # 直接导入，不通过__init__.py
        sys.path.insert(0, str(Path(__file__).parent / "core" / "processors"))
        from simplified_watermark_processor import SimplifiedWatermarkProcessor
        
        logger.info("🧪 测试SimplifiedWatermarkProcessor...")
        
        # 创建测试配置
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'mask_generator': {},
            'iopaint_config': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'hd_strategy': 'Original',
                'hd_strategy_crop_margin': 128,
                'hd_strategy_crop_trigger_size': 512,
                'hd_strategy_resize_limit': 2048
            }
        }
        
        # 测试初始化
        processor = SimplifiedWatermarkProcessor(config)
        logger.info("✅ 处理器初始化成功")
        
        # 测试状态获取
        status = processor.get_model_status()
        logger.info(f"📊 处理器状态: {status}")
        
        # 测试清理
        processor.cleanup()
        logger.info("✅ 处理器清理成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SimplifiedWatermarkProcessor测试失败: {e}")
        return False

def test_memory_pressure():
    """测试内存压力"""
    try:
        from core.utils.memory_monitor import MemoryMonitor
        
        logger.info("🔥 开始内存压力测试...")
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA不可用，跳过GPU压力测试")
            return True
        
        monitor = MemoryMonitor()
        
        # 分配张量测试
        test_tensors = []
        for i in range(5):
            try:
                # 分配50MB张量
                tensor = torch.randn(50, 1024, 1024, device='cuda')
                test_tensors.append(tensor)
                
                memory_info = monitor.get_memory_info()
                logger.info(f"📊 张量{i+1}分配后: GPU使用{memory_info['gpu_info']['usage_percent']:.1f}%")
                
                # 如果超过70%就停止
                if memory_info['gpu_info']['usage_percent'] > 70:
                    logger.warning("⚠️ GPU内存使用超过70%，停止分配")
                    break
                    
            except RuntimeError as e:
                logger.warning(f"⚠️ 张量分配失败: {e}")
                break
        
        # 清理
        logger.info("🧹 清理测试张量...")
        del test_tensors
        torch.cuda.empty_cache()
        
        final_memory = monitor.get_memory_info()
        logger.info(f"📊 最终内存状态: GPU使用{final_memory['gpu_info']['usage_percent']:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 内存压力测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始简化内存管理测试套件...")
    
    tests = [
        ("内存监控", test_memory_monitor),
        ("基础Inpainter", test_base_inpainter),
        ("简化处理器", test_simplified_processor),
        ("内存压力", test_memory_pressure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- 测试: {test_name} ---")
        try:
            if test_func():
                logger.info(f"✅ {test_name} 测试通过")
                passed += 1
            else:
                logger.error(f"❌ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
    
    logger.info(f"\n🎯 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！")
        return 0
    else:
        logger.error("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())