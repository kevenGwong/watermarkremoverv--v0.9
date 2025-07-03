#!/usr/bin/env python3
import sys
import click
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from torch.nn import Module
import tqdm
from loguru import logger
from enum import Enum

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""

def identify(task_prompt: TaskType, image: MatLike, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )

def get_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float):
    text_input = "watermark"
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    parsed_answer = identify(task_prompt, image, text_input, model, processor, device)

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        image_area = image.width * image.height
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            if (bbox_area / image_area) * 100 <= max_bbox_percent:
                draw.rectangle([x1, y1, x2, y2], fill=255)
            else:
                logger.warning(f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image")

    return mask

def dilate_mask(mask, kernel_size=4):
    """对mask进行膨胀操作"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def process_image_with_lama(image: MatLike, mask: MatLike, model_manager: ModelManager, dilate_kernel_size=4):
    """使用LaMa进行图像修复，支持mask膨胀，使用最佳参数配置"""
    # 对mask进行膨胀
    if dilate_kernel_size > 0:
        mask = dilate_mask(mask, dilate_kernel_size)
    
    # 使用最佳参数配置（来自test_direct_inpaint.py）
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

def make_region_transparent(image: Image.Image, mask: Image.Image):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

def load_custom_mask(mask_path: str, target_size: tuple) -> Image.Image:
    """加载自定义mask文件"""
    try:
        mask = Image.open(mask_path).convert("L")
        if mask.size != target_size:
            logger.info(f"Resizing mask from {mask.size} to {target_size}")
            mask = mask.resize(target_size, Image.NEAREST)
        return mask
    except Exception as e:
        logger.error(f"Failed to load mask from {mask_path}: {e}")
        return None

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--mask-path", type=click.Path(exists=True), help="Path to custom mask file (PNG)")
@click.option("--dilate-kernel", default=4, help="Kernel size for mask dilation (0 to disable)")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG"], case_sensitive=False), default=None, help="Force output format. Defaults to input format.")
def main(input_path: str, output_path: str, mask_path: str, dilate_kernel: int, overwrite: bool, transparent: bool, max_bbox_percent: float, force_format: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 只有在没有指定mask文件时才加载Florence-2模型
    florence_model = None
    florence_processor = None
    if not mask_path:
        florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(device).eval()
        florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        logger.info("Florence-2 Model loaded")
    else:
        logger.info(f"Using custom mask: {mask_path}")

    if not transparent:
        model_manager = ModelManager(name="lama", device=device)
        logger.info("LaMa model loaded")

    def handle_one(image_path: Path, output_path: Path):
        if output_path.exists() and not overwrite:
            logger.info(f"Skipping existing file: {output_path}")
            return

        # 读取图片为BGR格式（cv2格式）
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # 使用自定义mask或自动生成mask
        if mask_path:
            # 加载mask并转换为numpy数组
            mask_pil = load_custom_mask(mask_path, (image_bgr.shape[1], image_bgr.shape[0]))
            if mask_pil is None:
                logger.error("Failed to load custom mask, skipping this image")
                return
            mask_array = np.array(mask_pil)
        else:
            # 自动生成mask
            image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            mask_pil = get_watermark_mask(image_pil, florence_model, florence_processor, device, max_bbox_percent)
            mask_array = np.array(mask_pil)

        if transparent:
            # 透明模式需要PIL格式
            image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            result_image = make_region_transparent(image_pil, mask_pil)
        else:
            # 使用LaMa进行图像修复（BGR格式输入）
            lama_result = process_image_with_lama(image_bgr, mask_array, model_manager, dilate_kernel)
            
            # 直接使用cv2保存BGR格式（和test_direct_inpaint.py一致）
            cv2.imwrite(str(output_path), lama_result)
            logger.info(f"input_path:{image_path}, output_path:{output_path}")
            return  # 直接返回，跳过PIL保存逻辑

        # Determine output format
        if force_format:
            output_format = force_format.upper()
        elif transparent:
            output_format = "PNG"
        else:
            output_format = image_path.suffix[1:].upper()
            if output_format not in ["PNG", "WEBP", "JPG"]:
                output_format = "PNG"
        
        # Map JPG to JPEG for PIL compatibility
        if output_format == "JPG":
            output_format = "JPEG"

        if transparent and output_format == "JPG":
            logger.warning("Transparency detected. Defaulting to PNG for transparency support.")
            output_format = "PNG"

        new_output_path = output_path.with_suffix(f".{output_format.lower()}")
        result_image.save(new_output_path, format=output_format)
        logger.info(f"input_path:{image_path}, output_path:{new_output_path}")

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)

        images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
        total_images = len(images)

        for idx, image_path in enumerate(tqdm.tqdm(images, desc="Processing images")):
            output_file = output_path / image_path.name
            handle_one(image_path, output_file)
            progress = int((idx + 1) / total_images * 100)
            print(f"input_path:{image_path}, output_path:{output_file}, overall_progress:{progress}")
    else:
        output_file = output_path.with_suffix(".webp" if transparent else output_path.suffix)
        handle_one(input_path, output_file)
        print(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")

if __name__ == "__main__":
    main() 