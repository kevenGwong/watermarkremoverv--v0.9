#!/usr/bin/env python3
"""
Debugging script to compare mask formats and processing between:
1. test_iopaint.py (working well)
2. UI app processing (not working as well)
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config

def analyze_mask_format(mask_path: str):
    """分析mask文件的格式"""
    print(f"\n=== Analyzing mask: {mask_path} ===")
    
    # 使用cv2读取（test_iopaint.py的方式）
    mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"CV2 reading:")
    print(f"  Shape: {mask_cv2.shape}")
    print(f"  Dtype: {mask_cv2.dtype}")
    print(f"  Min/Max: {mask_cv2.min()}/{mask_cv2.max()}")
    print(f"  Unique values: {np.unique(mask_cv2)}")
    
    # 使用PIL读取（UI app的方式）
    mask_pil = Image.open(mask_path)
    print(f"\nPIL reading:")
    print(f"  Mode: {mask_pil.mode}")
    print(f"  Size: {mask_pil.size}")
    mask_pil_array = np.array(mask_pil)
    print(f"  Shape after np.array: {mask_pil_array.shape}")
    print(f"  Dtype: {mask_pil_array.dtype}")
    print(f"  Min/Max: {mask_pil_array.min()}/{mask_pil_array.max()}")
    print(f"  Unique values: {np.unique(mask_pil_array)}")
    
    # 转换为L模式（UI app可能的处理）
    mask_pil_l = mask_pil.convert('L')
    mask_pil_l_array = np.array(mask_pil_l)
    print(f"\nPIL L mode:")
    print(f"  Shape: {mask_pil_l_array.shape}")
    print(f"  Dtype: {mask_pil_l_array.dtype}")
    print(f"  Min/Max: {mask_pil_l_array.min()}/{mask_pil_l_array.max()}")
    print(f"  Unique values: {np.unique(mask_pil_l_array)}")
    
    # 检查是否相同
    if np.array_equal(mask_cv2, mask_pil_l_array):
        print("\n✅ CV2 and PIL L mode arrays are identical")
    else:
        print("\n❌ CV2 and PIL L mode arrays are different")
        diff = np.abs(mask_cv2.astype(np.int16) - mask_pil_l_array.astype(np.int16))
        print(f"  Max difference: {diff.max()}")
        print(f"  Different pixels: {np.sum(diff > 0)}")

def analyze_image_format(image_path: str):
    """分析输入图像的格式"""
    print(f"\n=== Analyzing image: {image_path} ===")
    
    # 使用cv2读取（test_iopaint.py的方式）
    image_cv2 = cv2.imread(image_path)
    print(f"CV2 reading:")
    print(f"  Shape: {image_cv2.shape}")
    print(f"  Dtype: {image_cv2.dtype}")
    print(f"  Color order: BGR")
    
    # 使用PIL读取（UI app的方式）
    image_pil = Image.open(image_path)
    print(f"\nPIL reading:")
    print(f"  Mode: {image_pil.mode}")
    print(f"  Size: {image_pil.size}")
    image_pil_array = np.array(image_pil)
    print(f"  Shape after np.array: {image_pil_array.shape}")
    print(f"  Dtype: {image_pil_array.dtype}")
    print(f"  Color order: RGB")

def test_iopaint_processing(image_path: str, mask_path: str, output_path: str):
    """测试test_iopaint.py的处理方式"""
    print(f"\n=== Test iopaint processing (Original Method) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_manager = ModelManager(name="lama", device=device)
    
    # test_iopaint.py的方式 - CV2读取BGR格式
    image = cv2.imread(image_path)  # BGR format
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"✅ CV2 读取:")
    print(f"  Image shape: {image.shape}, dtype: {image.dtype}, format: BGR")
    print(f"  Image sample pixel [0,0]: {image[0,0]} (BGR)")
    print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"  Mask unique values: {np.unique(mask)}")
    
    # 膨胀处理（使用kernel_size=5）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # LaMA处理 - 注意：这里image是BGR格式！
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    
    print(f"🔄 传入LaMA的数据:")
    print(f"  Image format: BGR (CV2原生)")
    print(f"  Image sample pixel [0,0]: {image[0,0]}")
    
    result = model_manager(image, dilated_mask, config)
    
    print(f"📤 LaMA输出:")
    print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
    print(f"  Result sample pixel [0,0]: {result[0,0]}")
    print(f"  Result min/max: {result.min()}/{result.max()}")
    
    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    # CV2保存 - 保持BGR格式
    cv2.imwrite(output_path, result)
    print(f"💾 CV2保存 (BGR格式): {output_path}")
    
    return result

def test_ui_processing(image_path: str, mask_path: str, output_path: str):
    """测试UI app的处理方式"""
    print(f"\n=== Test UI processing (Current App Method) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_manager = ModelManager(name="lama", device=device)
    
    # UI app的方式 - PIL读取RGB格式
    image_pil = Image.open(image_path).convert("RGB")  # RGB format
    mask_pil = Image.open(mask_path).convert("L")
    
    print(f"✅ PIL 读取:")
    print(f"  Image mode: {image_pil.mode}, size: {image_pil.size}, format: RGB")
    
    # 转换为numpy
    image_np = np.array(image_pil)  # RGB format
    mask_np = np.array(mask_pil)
    
    print(f"  Image sample pixel [0,0]: {image_np[0,0]} (RGB)")
    print(f"  Mask unique values: {np.unique(mask_np)}")
    
    # 添加相同的膨胀处理（kernel_size=5）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    
    # LaMA处理 - 注意：这里image是RGB格式！
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    
    print(f"🔄 传入LaMA的数据:")
    print(f"  Image format: RGB (PIL转numpy)")
    print(f"  Image sample pixel [0,0]: {image_np[0,0]}")
    
    result = model_manager(image_np, mask_np, config)
    
    print(f"📤 LaMA输出:")
    print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
    print(f"  Result sample pixel [0,0]: {result[0,0]}")
    print(f"  Result min/max: {result.min()}/{result.max()}")
    
    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 这是问题所在！UI app做了BGR→RGB转换
    print(f"🔄 UI app的颜色转换:")
    print(f"  转换前 result[0,0]: {result[0,0]} (LaMA输出)")
    
    # UI app的处理：认为result是BGR，转换为RGB
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    result_converted = np.array(result_pil)
    print(f"  转换后 result[0,0]: {result_converted[0,0]} (BGR→RGB转换后)")
    
    result_pil.save(output_path)
    print(f"💾 PIL保存 (经过BGR→RGB转换): {output_path}")
    
    return result

def test_ui_processing_fixed(image_path: str, mask_path: str, output_path: str):
    """测试修复后的UI app处理方式"""
    print(f"\n=== Test UI processing (FIXED Method) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_manager = ModelManager(name="lama", device=device)
    
    # UI app的方式 - PIL读取RGB格式
    image_pil = Image.open(image_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")
    
    # 转换为numpy
    image_np = np.array(image_pil)  # RGB format
    mask_np = np.array(mask_pil)
    
    print(f"✅ 修复方案 - 直接使用RGB格式:")
    print(f"  Input sample pixel [0,0]: {image_np[0,0]} (RGB)")
    
    # 膨胀处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    
    # LaMA处理配置
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    
    result = model_manager(image_np, mask_np, config)
    
    print(f"📤 LaMA输出:")
    print(f"  Result sample pixel [0,0]: {result[0,0]} (LaMA输出格式)")
    
    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 修复：直接保存，不做颜色转换
    print(f"🔧 修复方案 - 不做颜色转换:")
    result_pil = Image.fromarray(result)  # 直接使用result，不转换
    print(f"  Final sample pixel [0,0]: {np.array(result_pil)[0,0]} (无转换)")
    
    result_pil.save(output_path)
    print(f"💾 修复后保存: {output_path}")
    
    return result

def main():
    # 文件路径
    image_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/input/IMG_0001-3.jpg"
    mask_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/mask/IMG_0095-4_dilated_mask.png"
    output_dir = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/output"
    
    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)
    
    print("🔍 颜色格式诊断分析")
    print("=" * 60)
    
    # 测试三种处理方式
    test_iopaint_output = str(Path(output_dir) / "debug_test_iopaint_method.png")
    test_ui_output = str(Path(output_dir) / "debug_ui_app_method.png") 
    test_ui_fixed_output = str(Path(output_dir) / "debug_ui_app_FIXED.png")
    
    test_iopaint_processing(image_path, mask_path, test_iopaint_output)
    test_ui_processing(image_path, mask_path, test_ui_output)
    test_ui_processing_fixed(image_path, mask_path, test_ui_fixed_output)
    
    print(f"\n🎯 颜色格式诊断结果:")
    print(f"=" * 60)
    print(f"1. ✅ 原test_iopaint方法: {test_iopaint_output}")
    print(f"2. ❌ 当前UI app方法 (颜色问题): {test_ui_output}")
    print(f"3. 🔧 修复后UI方法: {test_ui_fixed_output}")
    print(f"\n💡 如果方法3的结果和方法1类似，说明问题就是颜色转换!")

if __name__ == "__main__":
    main()