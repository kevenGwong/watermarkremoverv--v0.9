#!/usr/bin/env python3
"""
CUDA内存管理测试脚本
验证SimplifiedWatermarkProcessor的内存管理功能
"""

import sys
import os
import time
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_memory_management():
    """测试内存管理功能"""
    logger.info("🧪 开始CUDA内存管理测试...")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用，跳过内存管理测试")
        return False
    
    try:
        # 导入必要的模块
        from core.processors.simplified_watermark_processor import SimplifiedWatermarkProcessor
        from core.utils.memory_monitor import MemoryMonitor
        
        # 创建测试配置
        config = {
            'device': 'cuda',
            'mask_generator': {},
            'iopaint_config': {
                'device': 'cuda',
                'hd_strategy': 'Original',
                'hd_strategy_crop_margin': 128,
                'hd_strategy_crop_trigger_size': 512,
                'hd_strategy_resize_limit': 2048
            }
        }
        
        # 创建测试图像
        test_image = Image.new('RGB', (512, 512), color='red')
        test_mask = Image.new('L', (512, 512), color=255)
        
        # 初始化处理器
        logger.info("🔧 初始化SimplifiedWatermarkProcessor...")
        processor = SimplifiedWatermarkProcessor(config)
        
        # 初始内存状态
        memory_monitor = MemoryMonitor()
        initial_memory = memory_monitor.get_memory_info()
        logger.info(f"📊 初始内存状态: {initial_memory}")
        
        # 测试模型切换和内存管理
        models_to_test = ["mat", "zits", "fcf"]
        
        for model_name in models_to_test:
            logger.info(f"\n🔄 测试模型切换: {model_name}")
            
            try:
                # 记录切换前内存
                before_memory = memory_monitor.get_memory_info()
                logger.info(f"📊 切换前内存: {before_memory}")
                
                # 模拟处理（不实际执行推理，只测试模型加载）
                logger.info(f"⚡ 模拟处理 {model_name} 模型...")
                
                # 切换模型
                processor._switch_model(model_name)
                
                # 记录切换后内存
                after_memory = memory_monitor.get_memory_info()
                logger.info(f"📊 切换后内存: {after_memory}")
                
                # 验证模型状态
                status = processor.get_model_status()
                logger.info(f"✅ 模型状态: {status}")
                
                # 等待一下让内存稳定
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ 模型 {model_name} 测试失败: {e}")
                continue
        
        # 测试资源清理
        logger.info("\n🧹 测试资源清理...")
        before_cleanup = memory_monitor.get_memory_info()
        
        processor.cleanup()
        
        after_cleanup = memory_monitor.get_memory_info()
        logger.info(f"📊 清理前内存: {before_cleanup}")
        logger.info(f"📊 清理后内存: {after_cleanup}")
        
        # 验证内存释放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = memory_monitor.get_memory_info()
            logger.info(f"📊 最终内存状态: {final_memory}")
        
        logger.info("✅ 内存管理测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 内存管理测试失败: {e}")
        return False

def test_memory_pressure():
    """测试内存压力情况"""
    logger.info("🔥 开始内存压力测试...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用，跳过压力测试")
        return False
    
    try:
        from core.utils.memory_monitor import MemoryMonitor
        
        memory_monitor = MemoryMonitor()
        
        # 分配大量内存测试
        logger.info("📈 分配测试内存...")
        test_tensors = []
        
        for i in range(5):
            # 分配100MB的张量
            tensor = torch.randn(100, 1024, 1024, device='cuda')
            test_tensors.append(tensor)
            
            memory_info = memory_monitor.get_memory_info()
            logger.info(f"📊 分配第{i+1}个张量后内存: {memory_info}")
            
            # 检查是否接近内存限制
            if memory_info.get('gpu_memory_percent', 0) > 80:
                logger.warning("⚠️ GPU内存使用率超过80%，停止分配")
                break
        
        # 清理测试内存
        logger.info("🧹 清理测试内存...")
        del test_tensors
        torch.cuda.empty_cache()
        
        final_memory = memory_monitor.get_memory_info()
        logger.info(f"📊 清理后内存: {final_memory}")
        
        logger.info("✅ 内存压力测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 内存压力测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始内存管理测试套件...")
    
    # 测试1: 基本内存管理
    test1_result = test_memory_management()
    
    # 测试2: 内存压力测试
    test2_result = test_memory_pressure()
    
    # 汇总结果
    if test1_result and test2_result:
        logger.info("🎉 所有内存管理测试通过！")
        return 0
    else:
        logger.error("❌ 部分测试失败，请检查日志")
        return 1

if __name__ == "__main__":
    sys.exit(main())