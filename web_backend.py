"""
Web Backend for Watermark Remover
Modular backend that processes images with different mask generation methods
"""
import os
import sys
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Dict, Any, Optional, Tuple, Union
import io
import logging
from dataclasses import dataclass

# Import existing modules
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config

# Import mask generators
sys.path.append(str(Path(__file__).parent.parent / "Watermark_Remover_KK"))
sys.path.append(str(Path(__file__).parent.parent / "Watermark_sam" / "watermark-segmentation"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """结果数据类"""
    success: bool
    result_image: Optional[Image.Image] = None
    mask_image: Optional[Image.Image] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

class CustomMaskGenerator:
    """自定义mask生成器，基于Watermark_sam模型"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        self._setup_preprocessing()
    
    def _load_model(self):
        """加载自定义分割模型"""
        try:
            import segmentation_models_pytorch as smp
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            # 定义模型架构（与Produce_mask.py一致）
            class WMModel(torch.nn.Module):
                def __init__(self, freeze_encoder=True):
                    super().__init__()
                    self.net = smp.create_model("FPN", encoder_name="mit_b5", in_channels=3, classes=1)
                    if freeze_encoder:
                        for p in self.net.encoder.parameters():
                            p.requires_grad = False

                def forward(self, x):
                    return self.net(x)
            
            # 加载模型
            self.model = WMModel(freeze_encoder=False).to(self.device)
            
            # 加载checkpoint
            mask_config = self.config['mask_generator']
            ckpt_path = mask_config['mask_model_path']
            
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
                
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
            self.model.net.load_state_dict(state_dict)
            self.model.eval()
            
            logger.info(f"Custom mask model loaded from: {ckpt_path}")
            
        except Exception as e:
            logger.error(f"Failed to load custom mask model: {e}")
            raise
    
    def _setup_preprocessing(self):
        """设置预处理管道"""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        mask_config = self.config['mask_generator']
        self.image_size = mask_config['image_size']
        self.imagenet_mean = mask_config['imagenet_mean']
        self.imagenet_std = mask_config['imagenet_std']
        self.mask_threshold = mask_config['mask_threshold']
        
        self.aug_val = A.Compose([
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    
    def generate_mask(self, image: Image.Image) -> Image.Image:
        """生成水印mask"""
        try:
            # 转换为numpy数组
            image_rgb = np.array(image.convert("RGB"))
            orig_h, orig_w = image_rgb.shape[:2]
            
            # 预处理
            sample = self.aug_val(image=image_rgb, mask=None)
            img_tensor = sample["image"].unsqueeze(0).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                pred_mask = self.model(img_tensor)
                pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
            
            # 后处理：正确计算缩放和padding信息
            scale = min(self.image_size / orig_h, self.image_size / orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            
            # 计算实际使用的padding（与aug_val和Produce_mask.py保持一致）
            pad_top = (self.image_size - new_h) // 2
            pad_left = (self.image_size - new_w) // 2
            
            # 裁剪掉padding，恢复到缩放后的尺寸（与原版实现完全一致）
            pred_crop = pred_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
            
            # 直接缩放回原尺寸，避免形变
            pred_mask = cv2.resize(pred_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # 二值化
            binary_mask = (pred_mask > self.mask_threshold).astype(np.uint8) * 255
            
            # 膨胀处理
            dilate_size = self.config['mask_generator'].get('mask_dilate_kernel_size', 3)
            if dilate_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            
            # 转换为PIL Image
            return Image.fromarray(binary_mask, mode='L')
            
        except Exception as e:
            logger.error(f"Custom mask generation failed: {e}")
            raise

class FlorenceMaskGenerator:
    """Florence-2 mask生成器（原版）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """加载Florence-2模型"""
        try:
            model_name = self.config['models']['florence_model']
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device).eval()
            self.florence_processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.info(f"Florence-2 model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}")
            raise
    
    def generate_mask(self, image: Image.Image, max_bbox_percent: float = 10.0, detection_prompt: str = "watermark") -> Image.Image:
        """使用Florence-2生成mask"""
        try:
            from utils import TaskType, identify
            
            # 检测水印（使用自定义prompt）
            text_input = detection_prompt
            task_prompt = TaskType.OPEN_VOCAB_DETECTION
            logger.info(f"🤖 Florence-2检测: 使用prompt '{text_input}', max_bbox_percent={max_bbox_percent}")
            parsed_answer = identify(task_prompt, image, text_input, 
                                   self.florence_model, self.florence_processor, 
                                   str(self.device))
            
            # 创建mask
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
                        logger.warning(f"Skipping large bounding box: {bbox}")
            
            return mask
            
        except Exception as e:
            logger.error(f"Florence mask generation failed: {e}")
            raise

class WatermarkProcessor:
    """水印处理主类"""
    
    def __init__(self, config_path: str = "web_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化mask生成器
        mask_type = self.config['mask_generator']['model_type']
        if mask_type == "custom":
            self.mask_generator = CustomMaskGenerator(self.config)
        else:
            self.mask_generator = FlorenceMaskGenerator(self.config)
        
        # 初始化inpainting模型
        self.model_manager = None
        self._load_lama_model()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _load_lama_model(self):
        """加载LaMA inpainting模型"""
        try:
            model_name = self.config['models']['lama_model']
            self.model_manager = ModelManager(name=model_name, device=str(self.device))
            logger.info(f"LaMA model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load LaMA model: {e}")
            raise
    
    @staticmethod
    def load_image_opencv(image_source: Union[str, io.BytesIO, bytes]) -> np.ndarray:
        """使用OpenCV直接读取图像，避免PIL的色彩校正"""
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

    def process_image(self, 
                     image: Union[Image.Image, str, io.BytesIO, bytes],
                     transparent: bool = False,
                     max_bbox_percent: float = 10.0,
                     detection_prompt: str = "watermark",
                     force_format: Optional[str] = None,
                     custom_inpaint_config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """处理单张图片"""
        import time
        start_time = time.time()
        
        logger.info("🚀 开始处理图像...")
        logger.info(f"📸 输入图像类型: {type(image)}")
        logger.info(f"🎯 处理模式: {'透明' if transparent else '修复'}")
        logger.info(f"📋 自定义配置: {custom_inpaint_config}")
        
        try:
            # 修正图像预处理 - 遵循iopaint要求（RGB输入）
            logger.info("🔄 开始修正图像预处理...")
            
            if isinstance(image, (str, io.BytesIO, bytes)):
                logger.info("📁 检测到文件路径/BytesIO/bytes输入，使用OpenCV读取然后转RGB")
                # 使用OpenCV读取BGR格式，但要转换为RGB给iopaint
                image_bgr = self.load_image_opencv(image)
                logger.info(f"   OpenCV读取BGR: shape={image_bgr.shape}, 第一个像素={image_bgr[0,0]}")
                
                # 转换为RGB用于LaMA处理（iopaint要求）
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                logger.info(f"   BGR→RGB转换: shape={image_rgb.shape}, 第一个像素={image_rgb[0,0]}")
                
                # 为mask生成创建PIL版本
                image_pil = Image.fromarray(image_rgb)
                logger.info(f"   PIL图像（用于mask生成）: size={image_pil.size}")
                
                # 主要处理流程使用RGB数据（iopaint要求）
                image_for_processing = image_rgb
            else:
                logger.info("🖼️ 检测到PIL图像输入")
                # PIL图像输入 - 直接使用RGB格式（iopaint要求）
                image_pil = image
                image_rgb = np.array(image_pil.convert("RGB"))
                logger.info(f"   PIL→RGB数组: shape={image_rgb.shape}, 第一个像素={image_rgb[0,0]}")
                
                # 主要处理流程使用RGB数据（iopaint要求）
                image_for_processing = image_rgb
            
            # 生成mask
            logger.info("🎭 开始生成mask...")
            if isinstance(self.mask_generator, CustomMaskGenerator):
                logger.info("🔧 使用Custom Mask Generator")
                mask_image = self.mask_generator.generate_mask(image_pil)
            else:
                logger.info("🤖 使用Florence-2 Mask Generator")
                mask_image = self.mask_generator.generate_mask(image_pil, max_bbox_percent, detection_prompt)
            
            logger.info(f"✅ Mask生成完成, size: {mask_image.size}, mode: {mask_image.mode}")
            
            if transparent:
                logger.info("🫥 开始透明处理...")
                # 透明处理
                result_image = self._make_region_transparent(image_pil, mask_image)
                logger.info(f"✅ 透明处理完成, size: {result_image.size}, mode: {result_image.mode}")
            else:
                logger.info("🤖 开始LaMA修复处理...")
                # LaMA inpainting（使用BGR格式 - 最佳实践）
                logger.info("🔄 使用BGR格式数据进行LaMA处理（最佳实践）")
                result_image = self._process_with_lama(image_for_processing, mask_image, custom_inpaint_config)
                
                logger.info(f"✅ LaMA修复完成, size: {result_image.size}, mode: {result_image.mode}")
            
            processing_time = time.time() - start_time
            logger.info(f"⏱️ 总处理时间: {processing_time:.2f}秒")
            
            return ProcessingResult(
                success=True,
                result_image=result_image,
                mask_image=mask_image,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 图像处理失败: {e}")
            logger.error(f"⏱️ 处理时间: {processing_time:.2f}秒")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _make_region_transparent(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """使区域透明（与remwm.py中的函数一致）"""
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
    
    def _process_with_lama(self, image: Union[Image.Image, np.ndarray, str], mask: Union[Image.Image, np.ndarray], custom_config: Optional[Dict[str, Any]] = None) -> Image.Image:
        """使用LaMA进行inpainting（支持自定义配置）- 采用test_direct_inpaint最佳实践"""
        logger.info("🔍 开始LaMA处理流程...")
        
        # 默认配置（与test_direct_inpaint.py保持一致）
        default_config = {
            'ldm_steps': 50,
            'ldm_sampler': 'ddim',
            'hd_strategy': 'CROP',
            'hd_strategy_crop_margin': 64,
            'hd_strategy_crop_trigger_size': 800,
            'hd_strategy_resize_limit': 1600,
        }
        
        # 合并自定义配置
        if custom_config:
            default_config.update(custom_config)
            logger.info(f"📋 使用自定义配置: {custom_config}")
        
        # 转换策略枚举
        strategy_map = {
            'CROP': HDStrategy.CROP,
            'RESIZE': HDStrategy.RESIZE,
            'ORIGINAL': HDStrategy.ORIGINAL
        }
        
        sampler_map = {
            'ddim': LDMSampler.ddim,
            'plms': LDMSampler.plms
        }
        
        config = Config(
            ldm_steps=default_config['ldm_steps'],
            ldm_sampler=sampler_map.get(default_config['ldm_sampler'], LDMSampler.ddim),
            hd_strategy=strategy_map.get(default_config['hd_strategy'], HDStrategy.CROP),
            hd_strategy_crop_margin=default_config['hd_strategy_crop_margin'],
            hd_strategy_crop_trigger_size=default_config['hd_strategy_crop_trigger_size'],
            hd_strategy_resize_limit=default_config['hd_strategy_resize_limit'],
        )
        
        logger.info(f"⚙️ LaMA配置: ldm_steps={config.ldm_steps}, sampler={config.ldm_sampler}, strategy={config.hd_strategy}")
        
        # 简化的颜色处理流程 - 遵循iopaint要求
        logger.info("🎨 开始LaMA处理流程...")
        
        # ① 处理图像输入 - 确保传入RGB格式给iopaint
        logger.info(f"📸 输入图像类型: {type(image)}")
        
        if isinstance(image, np.ndarray):
            # numpy数组输入，应该已经是RGB格式（从process_image传入）
            logger.info(f"🔢 RGB数组输入: shape={image.shape}, 第一个像素={image[0,0]}")
            image_rgb_for_lama = image
        elif isinstance(image, (bytes, io.BytesIO)):
            # 字节输入 - 需要解码为RGB
            logger.info(f"💾 字节输入，类型: {type(image)}")
            if isinstance(image, io.BytesIO):
                bytes_data = image.getvalue()
            else:
                bytes_data = image
            
            bytes_array = np.asarray(bytearray(bytes_data), dtype=np.uint8)
            image_bgr = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
            image_rgb_for_lama = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            logger.info(f"   解码为RGB: shape={image_rgb_for_lama.shape}, 第一个像素={image_rgb_for_lama[0,0]}")
        elif hasattr(image, 'size'):
            # PIL图像
            logger.info(f"🖼️ PIL图像输入: size={image.size}, mode={image.mode}")
            image_rgb_for_lama = np.array(image.convert("RGB"))
            logger.info(f"   转换为RGB数组: shape={image_rgb_for_lama.shape}, 第一个像素={image_rgb_for_lama[0,0]}")
        else:
            # 未知类型
            logger.error(f"❌ 不支持的图像类型: {type(image)}")
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # ② 处理mask输入
        logger.info("🎭 处理mask...")
        logger.info(f"   Mask输入类型: {type(mask)}")
        
        if isinstance(mask, np.ndarray):
            logger.info(f"   numpy mask: shape={mask.shape}, dtype={mask.dtype}")
            mask_np = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.IMREAD_GRAYSCALE)
        else:
            logger.info(f"   PIL mask: size={mask.size}, mode={mask.mode}")
            mask_np = np.array(mask.convert("L"))
        
        logger.info(f"✅ mask处理完成: shape={mask_np.shape}, dtype={mask_np.dtype}")
        logger.info(f"   白色像素数量: {np.sum(mask_np > 128)}")
        logger.info(f"   黑色像素数量: {np.sum(mask_np <= 128)}")
        logger.info(f"   mask覆盖率: {np.sum(mask_np > 128) / mask_np.size * 100:.2f}%")
        logger.info(f"   mask值范围: min={mask_np.min()}, max={mask_np.max()}")
        
        # ③ LaMA处理 - iopaint要求RGB输入，输出BGR
        logger.info("🤖 LaMA模型推理（RGB输入→BGR输出）...")
        logger.info(f"   输入RGB图像: shape={image_rgb_for_lama.shape}, 第一个像素={image_rgb_for_lama[0,0]}")
        logger.info(f"   输入mask: shape={mask_np.shape}, 覆盖率={np.sum(mask_np > 128) / mask_np.size * 100:.2f}%")
        
        # 验证输入数据有效性
        if np.sum(mask_np > 128) == 0:
            logger.warning("⚠️ WARNING: Mask中没有白色像素！LaMA将不会进行任何修复！")
        
        # 保存调试信息
        logger.info(f"   LaMA配置验证: {config}")
        logger.info(f"   图像数据范围: min={image_rgb_for_lama.min()}, max={image_rgb_for_lama.max()}")
        logger.info(f"   Mask数据范围: min={mask_np.min()}, max={mask_np.max()}")
        
        result_bgr = self.model_manager(image_rgb_for_lama, mask_np, config)
        
        logger.info(f"✅ LaMA推理完成!")
        logger.info(f"   输出BGR图像: shape={result_bgr.shape}, dtype={result_bgr.dtype}")
        logger.info(f"   输出第一个像素: {result_bgr[0,0]}")
        logger.info(f"   输出数据范围: min={result_bgr.min()}, max={result_bgr.max()}")
        
        if result_bgr.dtype in [np.float64, np.float32]:
            logger.info(f"🔄 数据类型转换: {result_bgr.dtype} → uint8")
            result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
            logger.info(f"   转换后数据范围: min={result_bgr.min()}, max={result_bgr.max()}")
        
        # ④ 转换BGR输出为RGB用于PIL显示（与remwm.py一致）
        logger.info("🔄 最终BGR→RGB转换（与remwm.py一致）...")
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        logger.info(f"   转换后RGB第一个像素: {result_rgb[0,0]}")
        
        result_image = Image.fromarray(result_rgb)
        logger.info(f"✅ 最终PIL图像: size={result_image.size}, mode={result_image.mode}")
        
        return result_image
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import psutil
        
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
            "ram_usage": f"{psutil.virtual_memory().percent:.1f}%",
            "cpu_usage": f"{psutil.cpu_percent():.1f}%"
        }
        
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            vram_total = gpu_info.total_memory // (1024 ** 2)
            vram_used = vram_total - (torch.cuda.memory_reserved(0) // (1024 ** 2))
            info["vram_usage"] = f"{vram_used}/{vram_total} MB"
        else:
            info["vram_usage"] = "N/A"
        
        return info