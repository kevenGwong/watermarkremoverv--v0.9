#!/usr/bin/env python3
"""
测试IOPaint集成
验证新的IOPaint处理器是否能正常工作
"""

import sys
import os
import logging
from PIL import Image
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.inference import InferenceManager
from config.config import ConfigManager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image():
    """创建测试图像"""
    # 创建一个简单的测试图像
    img = Image.new('RGB', (512, 512), color='white')
    
    # 添加一些简单的图形作为"水印"
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 400], fill='red', outline='black', width=3)
    draw.text((200, 250), "TEST", fill='black')
    
    return img

def create_test_mask():
    """创建测试mask"""
    # 创建一个简单的mask，覆盖红色矩形区域
    mask = Image.new('L', (512, 512), color=0)
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([100, 100, 400, 400], fill=255)
    
    return mask

def test_iopaint_integration():
    """测试IOPaint集成"""
    logger.info("🧪 开始测试IOPaint集成...")
    
    try:
        # 1. 测试配置管理器
        logger.info("📋 测试配置管理器...")
        config_manager = ConfigManager()
        logger.info("✅ 配置管理器加载成功")
        
        # 2. 测试推理管理器
        logger.info("🔧 测试推理管理器...")
        inference_manager = InferenceManager(config_manager)
        success = inference_manager.load_processor()
        
        if not success:
            logger.error("❌ 推理管理器加载失败")
            return False
            
        logger.info("✅ 推理管理器加载成功")
        
        # 3. 创建测试图像和mask
        logger.info("🖼️ 创建测试图像...")
        test_image = create_test_image()
        test_mask = create_test_mask()
        
        logger.info(f"📏 测试图像尺寸: {test_image.size}")
        logger.info(f"📏 测试mask尺寸: {test_mask.size}")
        
        # 4. 测试IOPaint处理
        logger.info("🎨 测试IOPaint处理...")
        
        # 准备参数
        mask_params = {
            'uploaded_mask': test_mask,  # 模拟上传的mask
            'mask_dilate_kernel_size': 0  # 不进行膨胀处理
        }
        
        inpaint_params = {
            'inpaint_model': 'iopaint',
            'auto_model_selection': True,
            'ldm_steps': 20,  # 减少步数以加快测试
            'hd_strategy': 'CROP'
        }
        
        performance_params = {}
        
        # 执行处理
        result = inference_manager.process_image(
            image=test_image,
            mask_model='upload',
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params=performance_params,
            transparent=False
        )
        
        if result.success:
            logger.info("✅ IOPaint处理成功!")
            logger.info(f"⏱️ 处理时间: {result.processing_time:.2f}秒")
            
            # 保存结果
            if result.result_image:
                result.result_image.save('test_iopaint_result.png')
                logger.info("💾 结果已保存为 test_iopaint_result.png")
            
            if result.mask_image:
                result.mask_image.save('test_iopaint_mask.png')
                logger.info("💾 Mask已保存为 test_iopaint_mask.png")
                
        else:
            logger.error(f"❌ IOPaint处理失败: {result.error_message}")
            return False
            
        # 5. 测试LaMA备选处理
        logger.info("🎨 测试LaMA备选处理...")
        
        inpaint_params['inpaint_model'] = 'lama'
        
        result = inference_manager.process_image(
            image=test_image,
            mask_model='upload',
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params=performance_params,
            transparent=False
        )
        
        if result.success:
            logger.info("✅ LaMA处理成功!")
            logger.info(f"⏱️ 处理时间: {result.processing_time:.2f}秒")
            
            if result.result_image:
                result.result_image.save('test_lama_result.png')
                logger.info("💾 LaMA结果已保存为 test_lama_result.png")
        else:
            logger.error(f"❌ LaMA处理失败: {result.error_message}")
            return False
            
        logger.info("🎉 所有测试通过!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_iopaint_integration()
    sys.exit(0 if success else 1) 