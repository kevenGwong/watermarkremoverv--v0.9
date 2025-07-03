"""
Florence-2 Utilities
Florence-2模型专用的工具函数
"""

from enum import Enum
import random
import matplotlib.patches as patches
import numpy as np
from PIL import ImageDraw
import logging

logger = logging.getLogger(__name__)

# Constants
colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']


class TaskType(str, Enum):
    """支持的任务类型"""
    CAPTION = '<CAPTION>'
    DETAILED_CAPTION = '<DETAILED_CAPTION>'
    MORE_DETAILED_CAPTION = '<MORE_DETAILED_CAPTION>'
    OPEN_VOCAB_DETECTION = '<OPEN_VOCABULARY_DETECTION>'


def run_example(task_prompt: TaskType, image, text_input=None, model=None, processor=None):
    """
    使用模型运行推理任务
    
    Args:
        task_prompt: 任务类型
        image: 输入图像
        text_input: 文本输入
        model: Florence-2模型
        processor: Florence-2处理器
        
    Returns:
        解析后的答案
    """
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt.value,
        image_size=(image.width, image.height)
    )
    return parsed_answer


def identify(task_prompt: TaskType, image, text_input, model, processor, device):
    """
    使用Florence-2进行识别任务
    
    Args:
        task_prompt: 任务类型
        image: 输入图像
        text_input: 文本输入
        model: Florence-2模型
        processor: Florence-2处理器
        device: 设备
        
    Returns:
        识别结果
    """
    try:
        prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # 移动到指定设备
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device)
        
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt.value,
            image_size=(image.width, image.height)
        )
        return parsed_answer
    except Exception as e:
        logger.error(f"Florence-2识别失败: {e}")
        raise


def draw_polygons(image, prediction, fill_mask=False):
    """
    在图像上绘制分割掩码的多边形
    
    Args:
        image: 输入图像
        prediction: 预测结果
        fill_mask: 是否填充掩码
        
    Returns:
        绘制了多边形的图像
    """
    draw = ImageDraw.Draw(image)
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for polygon in polygons:
            polygon = np.array(polygon).reshape(-1, 2)
            if len(polygon) < 3:
                logger.warning('Invalid polygon: %s', polygon)
                continue

            polygon = (polygon * 1).reshape(-1).tolist()  # No scaling
            draw.polygon(polygon, outline=color, fill=fill_color)
            draw.text((polygon[0] + 8, polygon[1] + 2), label, fill=color)

    return image


def draw_ocr_bboxes(image, prediction):
    """
    在图像上绘制OCR边界框
    
    Args:
        image: 输入图像
        prediction: 预测结果
        
    Returns:
        绘制了边界框的图像
    """
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * 1).tolist()  # No scaling
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0] + 8, new_box[1] + 2), "{}".format(label), align="right", fill=color)
    return image


def convert_bbox_to_relative(box, image):
    """
    将边界框像素坐标转换为相对坐标（范围0-999）
    
    Args:
        box: 边界框坐标
        image: 图像
        
    Returns:
        相对坐标列表
    """
    return [
        (box[0] / image.width) * 999,
        (box[1] / image.height) * 999,
        (box[2] / image.width) * 999,
        (box[3] / image.height) * 999,
    ]


def convert_relative_to_bbox(relative, image):
    """
    将相对坐标列表转换为像素坐标
    
    Args:
        relative: 相对坐标列表
        image: 图像
        
    Returns:
        像素坐标列表
    """
    return [
        (relative[0] / 999) * image.width,
        (relative[1] / 999) * image.height,
        (relative[2] / 999) * image.width,
        (relative[3] / 999) * image.height,
    ]


def convert_bbox_to_loc(box, image):
    """
    将边界框像素坐标转换为位置标记
    
    Args:
        box: 边界框坐标
        image: 图像
        
    Returns:
        位置标记字符串
    """
    relative_coordinates = convert_bbox_to_relative(box, image)
    return ''.join([f'<loc_{i}>' for i in relative_coordinates]) 