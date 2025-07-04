#!/usr/bin/env python3
"""
测试完整处理流程的调试脚本
"""

import time
import numpy as np
from PIL import Image
from core.inference import EnhancedWatermarkProcessor, WatermarkProcessor

def test_full_pipeline():
    """测试完整的处理流程"""
    print("🧪 测试完整处理流程...")
    
    # 初始化处理器
    print("🔧 初始化处理器...")
    base = WatermarkProcessor('web_config.yaml')
    enhanced = EnhancedWatermarkProcessor(base)
    
    # 创建测试图像和 mask
    image_size = (2000, 1500)
    mask_size = (2000, 1500)
    
    # 创建测试图像
    test_image = Image.new('RGB', image_size, color='white')
    print(f"📏 测试图像尺寸: {test_image.size}")
    
    # 创建测试 mask - 模拟水印区域
    test_mask = Image.new('L', mask_size, color=0)  # 黑色背景
    
    # 在 mask 中心添加一个白色区域（模拟水印）
    mask_array = np.array(test_mask)
    center_x, center_y = mask_size[0] // 2, mask_size[1] // 2
    mask_array[center_y-100:center_y+100, center_x-200:center_x+200] = 255  # 白色水印区域
    test_mask = Image.fromarray(mask_array, mode='L')
    
    print(f"📏 测试 mask 尺寸: {test_mask.size}")
    print(f"🔍 原始 mask 覆盖率: {np.sum(mask_array > 128) / mask_array.size * 100:.2f}%")
    
    # 模拟上传的 mask 文件
    class MockUploadedFile:
        def __init__(self, image):
            self.image = image
            self._buffer = None
        
        def seek(self, pos):
            pass
        
        def read(self, size=None):
            if self._buffer is None:
                import io
                self._buffer = io.BytesIO()
                self.image.save(self._buffer, format='PNG')
                self._buffer.seek(0)
            return self._buffer.getvalue()
    
    # 模拟 Web UI 参数
    mask_params = {
        'uploaded_mask': MockUploadedFile(test_mask),
        'mask_dilate_kernel_size': 5,
        'mask_dilate_iterations': 2
    }
    
    inpaint_params = {
        'inpaint_model': 'powerpaint',
        'task': 'object-removal',
        'prompt': 'empty scene blur, clean background, natural environment',
        'negative_prompt': 'object, person, animal, vehicle, building, text, watermark, logo, worst quality, low quality, normal quality, bad quality, blurry, artifacts',
        'num_inference_steps': 20,
        'guidance_scale': 7.5,
        'strength': 1.0,
        'crop_trigger_size': 512,
        'crop_margin': 64,
        'resize_to_512': True,
        'blend_edges': True,
        'edge_feather': 5,
        'seed': -1
    }
    
    performance_params = {}
    
    print("\n🚀 开始处理...")
    start_time = time.time()
    
    # 执行处理
    result = enhanced.process_image_with_params(
        image=test_image,
        mask_model='upload',
        mask_params=mask_params,
        inpaint_params=inpaint_params,
        performance_params=performance_params,
        transparent=False
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n✅ 处理完成，耗时: {processing_time:.2f}秒")
    print(f"处理成功: {result.success}")
    
    if result.success:
        print(f"结果图像尺寸: {result.result_image.size}")
        print(f"Mask 图像尺寸: {result.mask_image.size}")
        
        # 检查最终 mask 内容
        final_mask_array = np.array(result.mask_image)
        final_white_pixels = np.sum(final_mask_array > 128)
        final_total_pixels = final_mask_array.size
        final_coverage = final_white_pixels / final_total_pixels * 100
        print(f"最终 mask 覆盖率: {final_coverage:.2f}%")
        print(f"最终 mask 像素值范围: {final_mask_array.min()} - {final_mask_array.max()}")
        
        # 检查结果图像
        result_array = np.array(result.result_image)
        print(f"结果图像像素值范围: {result_array.min()} - {result_array.max()}")
        
        # 计算处理区域的变化
        if final_white_pixels > 0:
            # 只检查 mask 区域的变化
            mask_region = result_array[final_mask_array > 128]
            if len(mask_region) > 0:
                original_region = np.array(test_image)[final_mask_array > 128]
                change = np.mean(np.abs(mask_region.astype(float) - original_region.astype(float)))
                print(f"Mask 区域平均变化: {change:.2f}")
            else:
                print("⚠️ Mask 区域为空")
        else:
            print("⚠️ 没有检测到 mask 区域")
    else:
        print(f"处理失败: {result.error_message}")

if __name__ == "__main__":
    test_full_pipeline() 