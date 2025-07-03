"""
Advanced Web Backend with full parameter customization
支持所有可自定义参数的高级后端
"""
import os
import sys
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from typing import Dict, Any, Optional, Tuple, Union, List
import logging
from dataclasses import dataclass
from enum import Enum

# Import existing modules
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config

logger = logging.getLogger(__name__)

@dataclass
class AdvancedProcessingResult:
    """高级处理结果数据类"""
    success: bool
    result_image: Optional[Image.Image] = None
    mask_image: Optional[Image.Image] = None
    intermediate_results: Optional[Dict[str, Image.Image]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    parameters_used: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, float]] = None

class HDStrategyType(Enum):
    """高分辨率处理策略"""
    CROP = "CROP"
    RESIZE = "RESIZE" 
    ORIGINAL = "ORIGINAL"

class SamplerType(Enum):
    """采样器类型"""
    DDIM = "ddim"
    PLMS = "plms"
    DPM_SOLVER_PP = "dpm_solver++"

class AdvancedCustomMaskGenerator:
    """高级自定义mask生成器"""
    
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
            
            class WMModel(torch.nn.Module):
                def __init__(self, freeze_encoder=True):
                    super().__init__()
                    self.net = smp.create_model("FPN", encoder_name="mit_b5", in_channels=3, classes=1)
                    if freeze_encoder:
                        for p in self.net.encoder.parameters():
                            p.requires_grad = False

                def forward(self, x):
                    return self.net(x)
            
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
            
            logger.info(f"Advanced custom mask model loaded from: {ckpt_path}")
            
        except Exception as e:
            logger.error(f"Failed to load custom mask model: {e}")
            raise
    
    def _setup_preprocessing(self):
        """设置预处理管道"""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        custom_config = self.config['mask_generator']['custom_model']
        self.image_size = custom_config['image_size']
        self.imagenet_mean = custom_config['imagenet_mean']
        self.imagenet_std = custom_config['imagenet_std']
        self.mask_threshold = custom_config['mask_threshold']
        
        self.aug_val = A.Compose([
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    
    def generate_mask(self, image: Image.Image, **kwargs) -> Image.Image:
        """生成水印mask - 支持自定义参数"""
        try:
            # 获取自定义参数
            custom_threshold = kwargs.get('mask_threshold', self.mask_threshold)
            dilate_kernel_size = kwargs.get('mask_dilate_kernel_size', 
                                          self.config['mask_generator']['custom_model']['mask_dilate_kernel_size'])
            dilate_iterations = kwargs.get('mask_dilate_iterations',
                                         self.config['mask_generator']['custom_model']['mask_dilate_iterations'])
            
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
            
            # 后处理：计算padding信息
            scale = min(self.image_size / orig_h, self.image_size / orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            pad_top = (self.image_size - new_h) // 2
            pad_left = (self.image_size - new_w) // 2
            
            # 裁剪padding
            pred_crop = pred_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
            
            # 缩放回原尺寸
            pred_mask = cv2.resize(pred_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # 使用自定义阈值二值化
            binary_mask = (pred_mask > custom_threshold).astype(np.uint8) * 255
            
            # 膨胀处理
            if dilate_kernel_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=dilate_iterations)
            
            # 转换为PIL Image
            return Image.fromarray(binary_mask, mode='L')
            
        except Exception as e:
            logger.error(f"Advanced custom mask generation failed: {e}")
            raise

class AdvancedFlorenceMaskGenerator:
    """高级Florence-2 mask生成器"""
    
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
            logger.info(f"Advanced Florence-2 model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}")
            raise
    
    def generate_mask(self, image: Image.Image, **kwargs) -> Image.Image:
        """使用Florence-2生成mask - 支持自定义参数"""
        try:
            # 获取自定义参数
            custom_prompt = kwargs.get('detection_prompt', 
                                     self.config['mask_generator']['florence_model']['detection_prompt'])
            max_bbox_percent = kwargs.get('max_bbox_percent',
                                        self.config['mask_generator']['florence_model']['max_bbox_percent'])
            confidence_threshold = kwargs.get('confidence_threshold',
                                             self.config['mask_generator']['florence_model']['confidence_threshold'])
            
            # 导入识别函数
            from utils import TaskType, identify
            
            # 检测水印
            task_prompt = TaskType.OPEN_VOCAB_DETECTION
            parsed_answer = identify(task_prompt, image, custom_prompt, 
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
                    bbox_percent = (bbox_area / image_area) * 100
                    
                    # 应用置信度和尺寸过滤
                    if bbox_percent <= max_bbox_percent:
                        draw.rectangle([x1, y1, x2, y2], fill=255)
                        logger.info(f"Added bbox: {bbox}, area: {bbox_percent:.1f}%")
                    else:
                        logger.warning(f"Skipping large bbox: {bbox}, area: {bbox_percent:.1f}%")
            
            return mask
            
        except Exception as e:
            logger.error(f"Advanced Florence mask generation failed: {e}")
            raise

class AdvancedImageProcessor:
    """高级图像预处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def preprocess_image(self, image: Image.Image, **kwargs) -> Image.Image:
        """高级图像预处理"""
        proc_config = self.config['image_preprocessing']
        
        # 尺寸调整
        max_size = kwargs.get('max_input_size', proc_config['max_input_size'])
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            
            resize_method = proc_config['resize_method']
            resample_map = {
                'nearest': Image.NEAREST,
                'bilinear': Image.BILINEAR,
                'bicubic': Image.BICUBIC,
                'lanczos': Image.LANCZOS
            }
            image = image.resize(new_size, resample_map.get(resize_method, Image.LANCZOS))
        
        # Gamma校正
        gamma = kwargs.get('gamma_correction', proc_config['gamma_correction'])
        if gamma != 1.0:
            image = Image.eval(image, lambda x: int(255 * (x / 255) ** (1 / gamma)))
        
        # 对比度增强
        contrast = kwargs.get('contrast_enhancement', proc_config['contrast_enhancement'])
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        return image
    
    def postprocess_mask(self, mask: Image.Image, **kwargs) -> Image.Image:
        """高级mask后处理"""
        post_config = self.config['post_processing']
        
        mask_array = np.array(mask)
        
        # 模糊处理
        blur_radius = kwargs.get('mask_blur_radius', post_config['mask_blur_radius'])
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask_array = np.array(mask)
        
        # 腐蚀/膨胀
        erosion_size = kwargs.get('mask_erosion_size', post_config['mask_erosion_size'])
        if erosion_size != 0:
            kernel_size = abs(erosion_size)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            if erosion_size > 0:  # 腐蚀
                mask_array = cv2.erode(mask_array, kernel, iterations=1)
            else:  # 膨胀
                mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        
        # 羽化
        feather_size = kwargs.get('mask_feather_size', post_config['mask_feather_size'])
        if feather_size > 0:
            mask_array = cv2.GaussianBlur(mask_array, (feather_size*2+1, feather_size*2+1), 0)
        
        return Image.fromarray(mask_array, mode='L')

class AdvancedWatermarkProcessor:
    """高级水印处理器 - 支持所有自定义参数"""
    
    def __init__(self, config_path: str = "web_config_advanced.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化组件
        self.image_processor = AdvancedImageProcessor(self.config)
        
        # 初始化mask生成器
        mask_type = self.config['mask_generator']['model_type']
        if mask_type == "custom":
            self.mask_generator = AdvancedCustomMaskGenerator(self.config)
        else:
            self.mask_generator = AdvancedFlorenceMaskGenerator(self.config)
        
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
            logger.info(f"Advanced LaMA model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load LaMA model: {e}")
            raise
    
    def process_image(self, 
                     image: Image.Image,
                     transparent: bool = False,
                     advanced_params: Optional[Dict[str, Any]] = None) -> AdvancedProcessingResult:
        """高级图像处理 - 支持所有自定义参数"""
        import time
        import psutil
        
        start_time = time.time()
        initial_memory = psutil.virtual_memory().percent
        
        # 合并参数
        params = advanced_params or {}
        
        try:
            intermediate_results = {}
            
            # 1. 图像预处理
            processed_image = self.image_processor.preprocess_image(image, **params)
            if self.config['debug']['save_intermediate_results']:
                intermediate_results['preprocessed_image'] = processed_image
            
            # 2. 生成mask
            if isinstance(self.mask_generator, AdvancedCustomMaskGenerator):
                mask_image = self.mask_generator.generate_mask(processed_image, **params)
            else:
                mask_image = self.mask_generator.generate_mask(processed_image, **params)
            
            if self.config['debug']['save_intermediate_results']:
                intermediate_results['raw_mask'] = mask_image
            
            # 3. Mask后处理
            processed_mask = self.image_processor.postprocess_mask(mask_image, **params)
            if self.config['debug']['save_intermediate_results']:
                intermediate_results['processed_mask'] = processed_mask
            
            # 4. 应用处理
            if transparent:
                result_image = self._apply_transparent_effect(processed_image, processed_mask, **params)
            else:
                result_image = self._process_with_advanced_lama(processed_image, processed_mask, **params)
            
            # 5. 结果后处理
            final_result = self._postprocess_result(result_image, **params)
            
            # 计算性能指标
            processing_time = time.time() - start_time
            final_memory = psutil.virtual_memory().percent
            memory_usage = {
                'initial_memory_percent': initial_memory,
                'final_memory_percent': final_memory,
                'memory_increase': final_memory - initial_memory
            }
            
            if torch.cuda.is_available():
                memory_usage['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_usage['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2  # MB
            
            return AdvancedProcessingResult(
                success=True,
                result_image=final_result,
                mask_image=processed_mask,
                intermediate_results=intermediate_results,
                processing_time=processing_time,
                parameters_used=params,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            logger.error(f"Advanced image processing failed: {e}")
            return AdvancedProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                parameters_used=params
            )
    
    def _apply_transparent_effect(self, image: Image.Image, mask: Image.Image, **kwargs) -> Image.Image:
        """应用透明效果"""
        image = image.convert("RGBA")
        mask = mask.convert("L")
        
        # 转换为numpy数组进行快速处理
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # 创建透明结果
        result_array = img_array.copy()
        
        # 水印区域设为透明
        transparent_mask = mask_array > 128
        result_array[transparent_mask, 3] = 0  # 设置alpha通道为0
        
        return Image.fromarray(result_array, 'RGBA')
    
    def _process_with_advanced_lama(self, image: Image.Image, mask: Image.Image, **kwargs) -> Image.Image:
        """使用高级LaMA参数进行inpainting"""
        lama_config = self.config['lama_inpainting']
        
        # 获取自定义参数
        ldm_steps = kwargs.get('ldm_steps', lama_config['ldm_steps'])
        sampler = kwargs.get('ldm_sampler', lama_config['ldm_sampler'])
        hd_strategy = kwargs.get('hd_strategy', lama_config['hd_strategy'])
        crop_margin = kwargs.get('hd_strategy_crop_margin', lama_config['hd_strategy_crop_margin'])
        trigger_size = kwargs.get('hd_strategy_crop_trigger_size', lama_config['hd_strategy_crop_trigger_size'])
        resize_limit = kwargs.get('hd_strategy_resize_limit', lama_config['hd_strategy_resize_limit'])
        
        # 映射参数 - 只使用确认可用的采样器
        sampler_map = {
            'ddim': LDMSampler.ddim,
            'plms': LDMSampler.plms,
            # 移除可能不存在的dpm_solver_pp，避免AttributeError
        }
        
        strategy_map = {
            'CROP': HDStrategy.CROP,
            'RESIZE': HDStrategy.RESIZE,
            'ORIGINAL': HDStrategy.ORIGINAL
        }
        
        config = Config(
            ldm_steps=ldm_steps,
            ldm_sampler=sampler_map.get(sampler, LDMSampler.ddim),
            hd_strategy=strategy_map.get(hd_strategy, HDStrategy.CROP),
            hd_strategy_crop_margin=crop_margin,
            hd_strategy_crop_trigger_size=trigger_size,
            hd_strategy_resize_limit=resize_limit,
        )
        
        # 转换为numpy
        image_np = np.array(image.convert("RGB"))
        mask_np = np.array(mask.convert("L"))
        
        # LaMA处理
        result = self.model_manager(image_np, mask_np, config)
        
        if result.dtype in [np.float64, np.float32]:
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 转换回PIL
        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return result_image
    
    def _postprocess_result(self, image: Image.Image, **kwargs) -> Image.Image:
        """结果后处理"""
        post_config = self.config['post_processing']
        
        # 锐化
        sharpening = kwargs.get('output_sharpening', post_config['output_sharpening'])
        if sharpening > 0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1 + sharpening)
        
        # 降噪（简单实现）
        denoising = kwargs.get('output_denoising', post_config['output_denoising'])
        if denoising > 0:
            # 使用轻微的高斯模糊作为降噪
            image = image.filter(ImageFilter.GaussianBlur(radius=denoising))
        
        return image
    
    def get_advanced_system_info(self) -> Dict[str, Any]:
        """获取高级系统信息"""
        import psutil
        
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
            "ram_usage": f"{psutil.virtual_memory().percent:.1f}%",
            "cpu_usage": f"{psutil.cpu_percent():.1f}%",
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
        }
        
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            info.update({
                "gpu_name": gpu_info.name,
                "gpu_memory_total": f"{gpu_info.total_memory / (1024**2):.0f} MB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / (1024**2):.0f} MB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / (1024**2):.0f} MB",
                "gpu_utilization": f"{torch.cuda.utilization():.1f}%" if hasattr(torch.cuda, 'utilization') else "N/A"
            })
        
        return info
    
    def get_parameter_presets(self) -> Dict[str, Dict[str, Any]]:
        """获取参数预设"""
        return self.config['ui']['parameter_presets']
    
    def get_available_prompts(self) -> List[str]:
        """获取可用的检测提示词"""
        return self.config['mask_generator']['florence_model']['custom_prompts']