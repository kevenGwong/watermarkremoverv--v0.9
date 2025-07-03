"""
Image Processing Utilities
图像处理相关的工具函数
"""

from typing import Union, Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import io
import logging

logger = logging.getLogger(__name__)


def load_image_opencv(image_source: Union[str, io.BytesIO, bytes]) -> np.ndarray:
    """
    使用OpenCV直接读取图像，避免PIL的色彩校正
    
    Args:
        image_source: 图像源（文件路径、BytesIO或字节数据）
        
    Returns:
        OpenCV格式的图像数组（BGR）
    """
    logger.info(f"🔍 load_image_opencv: 输入类型 {type(image_source)}")
    
    if isinstance(image_source, str):
        # 文件路径
        logger.info(f"📁 从文件路径读取: {image_source}")
        image = cv2.imread(image_source)
        logger.info(f"   cv2.imread结果: shape={image.shape if image is not None else 'None'}, dtype={image.dtype if image is not None else 'None'}")
        return image
    elif isinstance(image_source, (io.BytesIO, bytes)):
        # BytesIO或字节数据
        logger.info(f"💾 从BytesIO/bytes读取")
        if isinstance(image_source, io.BytesIO):
            bytes_data = image_source.getvalue()
            logger.info(f"   BytesIO.getvalue() 数据长度: {len(bytes_data)}")
        else:
            bytes_data = image_source
            logger.info(f"   直接bytes数据长度: {len(bytes_data)}")
        
        bytes_array = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        logger.info(f"   转换为numpy数组: shape={bytes_array.shape}, dtype={bytes_array.dtype}")
        
        image = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
        logger.info(f"   cv2.imdecode结果: shape={image.shape if image is not None else 'None'}, dtype={image.dtype if image is not None else 'None'}")
        if image is not None:
            logger.info(f"   解码后第一个像素 (BGR): {image[0,0]}")
        return image
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")


def convert_bgr_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """
    将BGR图像转换为RGB
    
    Args:
        bgr_image: BGR格式的图像数组
        
    Returns:
        RGB格式的图像数组
    """
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_bgr(rgb_image: np.ndarray) -> np.ndarray:
    """
    将RGB图像转换为BGR
    
    Args:
        rgb_image: RGB格式的图像数组
        
    Returns:
        BGR格式的图像数组
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """
    将PIL图像转换为numpy数组
    
    Args:
        pil_image: PIL图像
        
    Returns:
        numpy数组
    """
    return np.array(pil_image.convert("RGB"))


def numpy_to_pil(numpy_array: np.ndarray) -> Image.Image:
    """
    将numpy数组转换为PIL图像
    
    Args:
        numpy_array: numpy数组
        
    Returns:
        PIL图像
    """
    return Image.fromarray(numpy_array)


def resize_image(image: Union[Image.Image, np.ndarray], 
                target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> Union[Image.Image, np.ndarray]:
    """
    调整图像大小
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
        keep_aspect_ratio: 是否保持宽高比
        
    Returns:
        调整后的图像
    """
    if isinstance(image, Image.Image):
        if keep_aspect_ratio:
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
        else:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image
    else:
        if keep_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        return image


def add_background(rgba_image: Image.Image, bg_type: str = "white") -> Image.Image:
    """
    为透明图像添加背景
    
    Args:
        rgba_image: RGBA图像
        bg_type: 背景类型 ("white", "black", "checkerboard")
        
    Returns:
        添加背景后的图像
    """
    if rgba_image.mode != "RGBA":
        return rgba_image
    
    # 创建背景
    if bg_type == "white":
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
    elif bg_type == "black":
        background = Image.new("RGB", rgba_image.size, (0, 0, 0))
    elif bg_type == "checkerboard":
        # 创建棋盘格背景
        background = create_checkerboard_background(rgba_image.size)
    else:
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
    
    # 合成图像
    background.paste(rgba_image, mask=rgba_image.split()[-1])  # 使用alpha通道作为mask
    return background


def create_checkerboard_background(size: Tuple[int, int], 
                                 square_size: int = 20) -> Image.Image:
    """
    创建棋盘格背景
    
    Args:
        size: 图像尺寸
        square_size: 方格大小
        
    Returns:
        棋盘格背景图像
    """
    width, height = size
    background = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(background)
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                draw.rectangle([x, y, x + square_size, y + square_size], 
                             fill=(200, 200, 200))
    
    return background


def make_region_transparent(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    使指定区域透明
    
    Args:
        image: 输入图像
        mask: 掩码图像
        
    Returns:
        透明处理后的图像
    """
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


def validate_image_format(image: Union[Image.Image, np.ndarray], 
                         allowed_formats: list = None) -> bool:
    """
    验证图像格式
    
    Args:
        image: 输入图像
        allowed_formats: 允许的格式列表
        
    Returns:
        是否有效
    """
    if allowed_formats is None:
        allowed_formats = ["RGB", "RGBA", "L"]
    
    if isinstance(image, Image.Image):
        return image.mode in allowed_formats
    else:
        # numpy数组
        if len(image.shape) == 3:
            return image.shape[2] in [1, 3, 4]  # 灰度、RGB、RGBA
        elif len(image.shape) == 2:
            return True  # 灰度图像
        else:
            return False


def get_image_info(image: Union[Image.Image, np.ndarray]) -> dict:
    """
    获取图像信息
    
    Args:
        image: 输入图像
        
    Returns:
        图像信息字典
    """
    if isinstance(image, Image.Image):
        return {
            "type": "PIL",
            "size": image.size,
            "mode": image.mode,
            "format": getattr(image, 'format', None)
        }
    else:
        return {
            "type": "numpy",
            "shape": image.shape,
            "dtype": str(image.dtype),
            "min_value": float(image.min()),
            "max_value": float(image.max())
        } 