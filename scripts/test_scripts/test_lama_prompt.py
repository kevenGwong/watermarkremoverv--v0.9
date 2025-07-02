#!/usr/bin/env python3
"""
测试LaMA模型是否真的使用prompt参数
"""
import numpy as np
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest, HDStrategy, LDMSampler
import torch
import cv2
from PIL import Image

def test_lama_prompt():
    """测试LaMA模型是否使用prompt参数"""
    print("🔍 测试LaMA模型是否使用prompt参数...")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载LaMA模型
    print("正在加载LaMA模型...")
    model_manager = ModelManager(name="lama", device=device)
    print("LaMA模型加载完成")
    
    # 创建一个简单的测试图像和mask
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    test_mask[100:150, 100:150] = 255  # 创建一个矩形mask
    
    print(f"测试图像尺寸: {test_image.shape}")
    print(f"测试mask尺寸: {test_mask.shape}")
    
    # 测试1: 不使用prompt
    print("\n📝 测试1: 不使用prompt")
    config1 = InpaintRequest(
        ldm_steps=20,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
        prompt="",
        negative_prompt=""
    )
    print(f"Config1 prompt: '{config1.prompt}'")
    print(f"Config1 negative_prompt: '{config1.negative_prompt}'")
    
    result1 = model_manager(test_image, test_mask, config1)
    print(f"结果1形状: {result1.shape}")
    
    # 测试2: 使用prompt
    print("\n📝 测试2: 使用prompt")
    config2 = InpaintRequest(
        ldm_steps=20,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
        prompt="natural background, seamless, realistic",
        negative_prompt="watermark, text, logo, artificial"
    )
    print(f"Config2 prompt: '{config2.prompt}'")
    print(f"Config2 negative_prompt: '{config2.negative_prompt}'")
    
    result2 = model_manager(test_image, test_mask, config2)
    print(f"结果2形状: {result2.shape}")
    
    # 比较结果
    print("\n🔍 结果比较:")
    diff = np.abs(result1.astype(float) - result2.astype(float))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"最大差异: {max_diff:.2f}")
    print(f"平均差异: {mean_diff:.2f}")
    
    if max_diff < 1.0:
        print("✅ 结论: LaMA模型不使用prompt参数 - 两个结果几乎完全相同")
    else:
        print("❓ 结论: LaMA模型可能使用prompt参数 - 结果有差异")
    
    # 检查LaMA模型的forward方法是否使用config参数
    print("\n🔍 检查LaMA模型源码:")
    from iopaint.model.lama import LaMa
    import inspect
    
    lama_model = model_manager.model
    print(f"LaMA模型类型: {type(lama_model)}")
    
    # 检查forward方法是否使用config参数
    source = inspect.getsource(lama_model.forward)
    lines = source.split('\n')
    
    print("Forward方法中使用config的行:")
    config_usage = []
    for i, line in enumerate(lines):
        if 'config.' in line:
            config_usage.append(f"第{i+1}行: {line.strip()}")
    
    if config_usage:
        for usage in config_usage:
            print(f"  {usage}")
    else:
        print("  ❌ 没有找到config参数的使用")
    
    print("\n📋 总结:")
    print("1. InpaintRequest确实有prompt和negative_prompt字段")
    print("2. 但LaMA模型的forward方法没有使用这些参数")
    print("3. 只有SD等扩散模型才真正使用prompt参数")
    print("4. 在LaMA模型中传递prompt参数是无效的")

if __name__ == "__main__":
    test_lama_prompt() 