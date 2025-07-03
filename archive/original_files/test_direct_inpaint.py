#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch

def dilate_mask(mask, kernel_size=4):
    """对mask进行膨胀操作"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def process_image_with_lama_dilate(image, mask, model_manager, dilate_kernel_size=4):
    """使用LaMa进行图像修复，支持mask膨胀"""
    # 对mask进行膨胀
    if dilate_kernel_size > 0:
        mask = dilate_mask(mask, dilate_kernel_size)
    
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def main():
    # 设置路径
    input_image_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/input/IMG_0308-3.jpg"
    mask_path = "/home/duolaameng/SAM_Remove/Watermark_sam/mask/watermark_2000x1500.png"
    output_dir = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/output"
    dilate_kernel_size = 4
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否存在
    if not Path(input_image_path).exists():
        print(f"错误：输入图片不存在: {input_image_path}")
        return
    
    if not Path(mask_path).exists():
        print(f"错误：mask文件不存在: {mask_path}")
        return
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载LaMa模型
    print("正在加载LaMa模型...")
    model_manager = ModelManager(name="lama", device=device)
    print("LaMa模型加载完成")
    
    # 读取图片和mask
    print("正在读取图片和mask...")
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"错误：无法读取图片: {input_image_path}")
        return
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"错误：无法读取mask: {mask_path}")
        return
    
    print(f"图片尺寸: {image.shape}")
    print(f"Mask尺寸: {mask.shape}")
    
    # 处理图片
    print("正在使用LaMa进行图像修复...")
    result = process_image_with_lama_dilate(image, mask, model_manager, dilate_kernel_size)
    
    # 保存结果
    output_path = Path(output_dir) / "result.jpg"
    cv2.imwrite(str(output_path), result)
    print(f"处理完成，结果保存到: {output_path}")
    
    # 同时保存膨胀后的mask用于调试
    dilated_mask = dilate_mask(mask, dilate_kernel_size)
    mask_output_path = Path(output_dir) / "dilated_mask.png"
    cv2.imwrite(str(mask_output_path), dilated_mask)
    print(f"膨胀后的mask保存到: {mask_output_path}")

if __name__ == "__main__":
    main() 