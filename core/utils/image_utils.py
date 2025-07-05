"""
图像处理工具模块
负责图像转换、背景处理、下载等功能
"""

import io
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """图像处理器"""
    
    @staticmethod
    def resize_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """调整图像尺寸"""
        return image.resize(target_size, Image.LANCZOS)
    
    @staticmethod
    def ensure_rgb(image: Image.Image) -> Image.Image:
        """确保图像为RGB格式"""
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image
    
    @staticmethod
    def ensure_rgba(image: Image.Image) -> Image.Image:
        """确保图像为RGBA格式"""
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        return image
    
    @staticmethod
    def ensure_grayscale(image: Image.Image) -> Image.Image:
        """确保图像为灰度格式"""
        if image.mode != 'L':
            return image.convert('L')
        return image
    
    @staticmethod
    def apply_transparency(image: Image.Image, mask: Image.Image) -> Image.Image:
        """应用透明效果"""
        image_rgba = ImageProcessor.ensure_rgba(image)
        mask_gray = ImageProcessor.ensure_grayscale(mask)
        
        img_array = np.array(image_rgba)
        mask_array = np.array(mask_gray)
        
        # 应用透明效果
        transparent_mask = mask_array > 128
        img_array[transparent_mask, 3] = 0
        
        return Image.fromarray(img_array, 'RGBA')
    
    @staticmethod
    def add_background(rgba_image: Image.Image, bg_type: str) -> Image.Image:
        """为RGBA图像添加背景"""
        if bg_type == "black":
            bg = Image.new('RGB', rgba_image.size, (0, 0, 0))
        elif bg_type == "checkered":
            bg = Image.new('RGB', rgba_image.size, (255, 255, 255))
            # 创建棋盘背景
            for y in range(0, rgba_image.size[1], 20):
                for x in range(0, rgba_image.size[0], 20):
                    if (x//20 + y//20) % 2:
                        for dy in range(min(20, rgba_image.size[1] - y)):
                            for dx in range(min(20, rgba_image.size[0] - x)):
                                bg.putpixel((x + dx, y + dy), (200, 200, 200))
        else:  # white
            bg = Image.new('RGB', rgba_image.size, (255, 255, 255))
        
        # 合并图像
        bg.paste(rgba_image, mask=rgba_image.split()[-1])
        return bg
    
    @staticmethod
    def apply_mask_dilation(mask: Image.Image, kernel_size: int, iterations: int) -> Image.Image:
        """应用mask膨胀"""
        if kernel_size <= 0 or iterations <= 0:
            return mask
        
        try:
            import cv2
            mask_array = np.array(mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_array = cv2.dilate(mask_array, kernel, iterations=iterations)
            return Image.fromarray(dilated_array, mode='L')
        except ImportError:
            logger.warning("OpenCV not available, skipping mask dilation")
            return mask
    
    @staticmethod
    def calculate_mask_coverage(mask: Image.Image) -> float:
        """计算mask覆盖率"""
        mask_array = np.array(mask)
        white_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        return (white_pixels / total_pixels) * 100

class ImageDownloader:
    """图像下载器"""
    
    @staticmethod
    def create_download_data(image: Image.Image, format: str, quality: int = 95) -> Tuple[bytes, str]:
        """创建下载数据"""
        img_buffer = io.BytesIO()
        
        if format.upper() == "PNG":
            image.save(img_buffer, format="PNG")
            mime_type = "image/png"
        elif format.upper() == "WEBP":
            image.save(img_buffer, format="WEBP", quality=quality)
            mime_type = "image/webp"
        elif format.upper() == "JPG" or format.upper() == "JPEG":
            # 处理RGBA图像
            if image.mode == "RGBA":
                rgb_img = Image.new("RGB", image.size, (255, 255, 255))
                rgb_img.paste(image, mask=image.split()[-1])
                image = rgb_img
            image.save(img_buffer, format="JPEG", quality=quality)
            mime_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        img_buffer.seek(0)
        return img_buffer.getvalue(), mime_type
    
    @staticmethod
    def get_filename_base(original_filename: str) -> str:
        """获取文件名基础部分"""
        return Path(original_filename).stem
    
    @staticmethod
    def generate_filename(base_name: str, format: str, suffix: str = "") -> str:
        """生成文件名"""
        return f"{base_name}{suffix}.{format.lower()}"
    
    @staticmethod
    def create_download_info(image: Image.Image, filename_base: str) -> list:
        """创建下载信息列表"""
        download_info = []
        formats = [("PNG", "image/png"), ("WEBP", "image/webp"), ("JPG", "image/jpeg")]
        
        for fmt, mime in formats:
            try:
                data, mime_type = ImageDownloader.create_download_data(image, fmt)
                filename = ImageDownloader.generate_filename(filename_base, fmt, "_debug")
                
                download_info.append({
                    'format': fmt,
                    'data': data,
                    'filename': filename,
                    'mime_type': mime_type
                })
            except Exception as e:
                logger.error(f"Failed to create {fmt} download: {e}")
        
        return download_info

class ImageUtils:
    """统一图像处理工具类 - 支持简化处理器"""
    
    @staticmethod
    def preprocess_for_model(image: Image.Image, mask: Image.Image, model_name: str) -> Tuple[Image.Image, Image.Image]:
        """为特定模型预处理图像和mask"""
        # 标准预处理 - 所有模型都使用RGB格式的PIL图像
        # 具体的颜色空间转换在各模型内部处理
        processed_image = image.convert("RGB")
        processed_mask = mask.convert("L")
        
        # 确保尺寸匹配
        if processed_image.size != processed_mask.size:
            processed_mask = processed_mask.resize(processed_image.size, Image.NEAREST)
        
        return processed_image, processed_mask
    
    @staticmethod
    def prepare_arrays_for_iopaint(image: Image.Image, mask: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """为所有IOPaint模型（MAT/ZITS/FCF/LaMA）准备numpy数组
        
        IOPaint实际行为：
        - IOPaint内部模型期望BGR格式输入（通过测试验证）
        - 输入BGR → 输出RGB（IOPaint内部进行转换）
        - Mask：单通道灰度数组 (H, W)，值范围0-255，白色=需要修复的区域
        """
        import cv2
        
        # 先转为RGB，再转为BGR给IOPaint
        image_rgb = np.array(image.convert("RGB"))
        image_array = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        mask_array = np.array(mask.convert("L"))
        
        # 确保mask是二值的，白色区域=需要修复的区域
        mask_array = (mask_array > 128).astype(np.uint8) * 255
        
        logger.debug(f"IOPaint准备: 图像{image_array.shape} BGR格式(转换自RGB), Mask{mask_array.shape} 灰度格式")
        
        return image_array, mask_array
    
    
    @staticmethod
    def postprocess_result(result_array: np.ndarray, model_name: str, target_size: Tuple[int, int]) -> Image.Image:
        """后处理模型输出结果"""
        # 确保结果是uint8格式
        if result_array.dtype != np.uint8:
            result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
        
        # 转换为PIL图像
        if len(result_array.shape) == 3 and result_array.shape[2] == 3:
            result_image = Image.fromarray(result_array, 'RGB')
        else:
            result_image = Image.fromarray(result_array)
        
        # 调整到目标尺寸
        if result_image.size != target_size:
            result_image = result_image.resize(target_size, Image.LANCZOS)
        
        return result_image
    
    @staticmethod
    def process_uploaded_mask(uploaded_file, target_size: Tuple[int, int]) -> Image.Image:
        """处理上传的mask文件"""
        try:
            # 读取上传的文件
            if hasattr(uploaded_file, 'read'):
                uploaded_file.seek(0)
                mask_image = Image.open(uploaded_file)
            elif isinstance(uploaded_file, Image.Image):
                mask_image = uploaded_file
            else:
                mask_image = Image.open(uploaded_file)
            
            # 转换为灰度图像
            mask_image = mask_image.convert('L')
            
            # 调整尺寸
            if mask_image.size != target_size:
                mask_image = mask_image.resize(target_size, Image.NEAREST)
            
            return mask_image
            
        except Exception as e:
            logger.error(f"Failed to process uploaded mask: {e}")
            raise ValueError(f"Invalid mask file: {e}")
    
    @staticmethod
    def calculate_mask_coverage(mask: Image.Image) -> float:
        """计算mask覆盖率"""
        mask_array = np.array(mask.convert('L'))
        white_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        return (white_pixels / total_pixels) * 100
    
    @staticmethod
    def validate_image_for_processing(image: Image.Image) -> bool:
        """验证图像是否适合处理"""
        try:
            # 检查尺寸
            if image.size[0] < 64 or image.size[1] < 64:
                logger.warning("Image too small for processing")
                return False
            
            if image.size[0] > 4096 or image.size[1] > 4096:
                logger.warning("Image too large, may cause memory issues")
                return False
            
            # 检查格式
            if image.mode not in ['RGB', 'RGBA', 'L']:
                logger.warning(f"Unsupported image mode: {image.mode}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False

class ImageValidator:
    """图像验证器"""
    
    @staticmethod
    def validate_image(image: Image.Image) -> bool:
        """验证图像是否有效"""
        try:
            # 检查图像尺寸
            if image.size[0] <= 0 or image.size[1] <= 0:
                return False
            
            # 检查图像模式
            if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
                return False
            
            # 尝试访问像素数据
            image.getpixel((0, 0))
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_mask(mask: Image.Image) -> bool:
        """验证mask是否有效"""
        if not ImageValidator.validate_image(mask):
            return False
        
        # 确保mask是灰度图像
        if mask.mode != 'L':
            return False
        
        return True
    
    @staticmethod
    def get_image_info(image: Image.Image) -> dict:
        """获取图像信息"""
        return {
            'size': image.size,
            'mode': image.mode,
            'format': getattr(image, 'format', 'Unknown'),
            'width': image.size[0],
            'height': image.size[1]
        } 