"""
推理逻辑模块
负责AI模型推理、mask生成和inpainting处理
基于原始 web_backend.py 的完整 LaMA 集成
"""

import time
import logging
import yaml
import io
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from pathlib import Path

from core.utils.image_utils import ImageProcessor
from core.models.lama_processor import LamaProcessor
from core.models.powerpaint_processor import PowerPaintProcessor

logger = logging.getLogger(__name__)

class ProcessingResult:
    """处理结果数据类"""
    def __init__(self, success: bool, result_image: Optional[Image.Image] = None,
                 mask_image: Optional[Image.Image] = None, error_message: Optional[str] = None,
                 processing_time: float = 0.0):
        self.success = success
        self.result_image = result_image
        self.mask_image = mask_image
        self.error_message = error_message
        self.processing_time = processing_time

class CustomMaskGenerator:
    """自定义mask生成器 - 基于 Watermark_sam 模型"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """加载自定义分割模型"""
        try:
            import torch
            import segmentation_models_pytorch as smp
            import albumentations as A
            import cv2
            from albumentations.pytorch import ToTensorV2
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 定义模型架构
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
            
            if not Path(ckpt_path).exists():
                logger.error(f"Custom mask model not found: {ckpt_path}")
                logger.error("Custom mask generation will not be available")
                self.model = None
                raise FileNotFoundError(f"Custom mask model not found: {ckpt_path}")
                
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
            self.model.net.load_state_dict(state_dict)
            self.model.eval()
            
            # Setup preprocessing
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
            
            logger.info(f"✅ Custom mask model loaded from: {ckpt_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load custom mask model: {e}")
            # Clean up any partially loaded resources
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def generate_mask(self, image: Image.Image, mask_params: Dict[str, Any] = None) -> Image.Image:
        """生成水印mask"""
        if self.model is None:
            logger.warning("Custom mask model not available, returning empty mask")
            return Image.new('L', image.size, 0)
        
        try:
            import torch
            import cv2
            
            # 使用动态参数或默认值
            mask_threshold = mask_params.get('mask_threshold', self.mask_threshold) if mask_params else self.mask_threshold
            dilate_size = mask_params.get('mask_dilate_kernel_size', 3) if mask_params else 3
            dilate_iterations = mask_params.get('mask_dilate_iterations', 1) if mask_params else 1
            
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
            
            # 后处理
            scale = min(self.image_size / orig_h, self.image_size / orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            
            pad_top = (self.image_size - new_h) // 2
            pad_left = (self.image_size - new_w) // 2
            
            pred_crop = pred_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
            pred_mask = cv2.resize(pred_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # 二值化
            binary_mask = (pred_mask > mask_threshold).astype(np.uint8) * 255
            
            # 膨胀处理
            if dilate_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=dilate_iterations)
            
            return Image.fromarray(binary_mask, mode='L')
            
        except Exception as e:
            logger.error(f"Custom mask generation failed: {e}")
            return Image.new('L', image.size, 0)

class FlorenceMaskGenerator:
    """Florence-2 mask生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.florence_model = None
        self.florence_processor = None
        self._load_model()
    
    def _load_model(self):
        """加载Florence-2模型"""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = self.config['models']['florence_model']
            
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to(device).eval()
            self.florence_processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.info(f"✅ Florence-2 model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Florence-2 model: {e}")
            self.florence_model = None
            self.florence_processor = None
    
    def generate_mask(self, image: Image.Image, mask_params: Dict[str, Any] = None) -> Image.Image:
        """使用Florence-2生成mask"""
        if self.florence_model is None:
            logger.warning("Florence-2 model not available, returning empty mask")
            return Image.new('L', image.size, 0)
        
        try:
            # 获取参数
            max_bbox_percent = mask_params.get('max_bbox_percent', 10.0) if mask_params else 10.0
            detection_prompt = mask_params.get('detection_prompt', 'watermark') if mask_params else 'watermark'
            
            # TODO: 实现Florence-2检测逻辑
            # 由于缺少utils模块，暂时返回空mask
            logger.warning("Florence-2 detection logic not implemented yet")
            return Image.new('L', image.size, 0)
            
        except Exception as e:
            logger.error(f"Florence mask generation failed: {e}")
            return Image.new('L', image.size, 0)

class WatermarkProcessor:
    """水印处理主类 - 基于原始 web_backend.py"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "web_config.yaml"
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._resources = []  # Track resources for cleanup
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup"""
        self.cleanup_resources()
    
    def cleanup_resources(self):
        """Clean up all processor resources"""
        try:
            # Clean up LaMA processor
            if hasattr(self, 'lama_processor') and self.lama_processor:
                if hasattr(self.lama_processor, 'cleanup_resources'):
                    self.lama_processor.cleanup_resources()
            
            # Clean up mask generator if it has cleanup method
            if hasattr(self, 'mask_generator') and hasattr(self.mask_generator, 'cleanup_resources'):
                self.mask_generator.cleanup_resources()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ WatermarkProcessor resources cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error during WatermarkProcessor cleanup: {e}")
    
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize processor components"""
        # 初始化mask生成器
        mask_type = self.config['mask_generator']['model_type']
        try:
            if mask_type == "custom":
                self.mask_generator = CustomMaskGenerator(self.config)
            else:
                self.mask_generator = FlorenceMaskGenerator(self.config)
        except Exception as e:
            logger.error(f"Failed to initialize mask generator: {e}")
            # 提供降级方案 - 使用基础的空mask生成器
            self.mask_generator = self._create_fallback_mask_generator()
            logger.info("Using fallback mask generator")
        
        # 初始化LaMA处理器
        try:
            self.lama_processor = LamaProcessor(self.config)
        except Exception as e:
            logger.error(f"Failed to initialize LaMA processor: {e}")
            raise RuntimeError(f"Critical failure: LaMA processor initialization failed: {e}")
        
        logger.info(f"✅ WatermarkProcessor initialized with {mask_type} mask generator")
    
    def _create_fallback_mask_generator(self):
        """创建降级mask生成器"""
        class FallbackMaskGenerator:
            def __init__(self):
                pass
            
            def generate_mask(self, image: Image.Image, mask_params: Dict[str, Any] = None) -> Image.Image:
                logger.warning("Using fallback mask generator - returning empty mask")
                return Image.new('L', image.size, 0)
        
        return FallbackMaskGenerator()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ Config loaded from: {config_path}")
                return config
        except Exception as e:
            logger.warning(f"Config file {config_path} not found, using defaults")
            # 使用ConfigManager的默认配置
            from config.config import ConfigManager
            config_manager = ConfigManager()
            return self._build_default_config(config_manager)
    
    def _build_default_config(self, config_manager) -> Dict[str, Any]:
        """构建默认配置"""
        return {
            'mask_generator': {
                'model_type': 'custom',
                'mask_model_path': '/home/duolaameng/SAM_Remove/Watermark_sam/output/checkpoints/epoch=071-valid_iou=0.7267.ckpt',
                'image_size': 768,
                'imagenet_mean': [0.485, 0.456, 0.406],
                'imagenet_std': [0.229, 0.224, 0.225],
                'mask_threshold': config_manager.app_config.default_mask_threshold,
            },
            'models': {
                'florence_model': 'microsoft/Florence-2-large',
                'lama_model': config_manager.get_model_config().get('lama_model', 'lama')
            }
        }
    
    def process_image(self, 
                     image: Image.Image,
                     transparent: bool = False,
                     max_bbox_percent: float = 10.0,
                     force_format: Optional[str] = None,
                     custom_inpaint_config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """处理单张图片"""
        start_time = time.time()
        
        try:
            logger.info("🚀 开始处理图像...")
            logger.info(f"📸 输入图像: size={image.size}, mode={image.mode}")
            logger.info(f"🎯 处理模式: {'透明' if transparent else '修复'}")
            
            # 生成mask
            logger.info("🎭 开始生成mask...")
            mask_params = {'max_bbox_percent': max_bbox_percent}
            mask_image = self.mask_generator.generate_mask(image, mask_params)
            
            # 验证mask
            mask_array = np.array(mask_image)
            white_pixels = np.sum(mask_array > 128)
            total_pixels = mask_array.size
            mask_coverage = white_pixels / total_pixels * 100
            logger.info(f"🔍 Mask验证: 覆盖率={mask_coverage:.2f}%")
            
            if transparent:
                logger.info("🫥 应用透明处理...")
                result_image = self._make_region_transparent(image, mask_image)
            else:
                logger.info("🎨 应用LaMA修复处理...")
                if custom_inpaint_config is None:
                    custom_inpaint_config = {}
                result_image_array = self.lama_processor.predict(image, mask_image, custom_inpaint_config)
                result_image = Image.fromarray(result_image_array)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ 处理完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                success=True,
                result_image=result_image,
                mask_image=mask_image,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 图像处理失败: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _make_region_transparent(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """使区域透明"""
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
    
    def _process_with_lama(self, image: Image.Image, mask: Image.Image, lama_config: Dict[str, Any]) -> Image.Image:
        """使用LaMA进行inpainting - 兼容接口"""
        result_array = self.lama_processor.predict(image, mask, lama_config)
        return Image.fromarray(result_array)
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import torch
        import psutil
        
        info = {
            "cuda_available": torch.cuda.is_available(),
            "lama_loaded": self.lama_processor.model_loaded,
            "mask_generator": self.config['mask_generator']['model_type'],
            "ram_usage": f"{psutil.virtual_memory().percent:.1f}%",
            "cpu_usage": f"{psutil.cpu_percent():.1f}%"
        }
        
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            vram_total = gpu_info.total_memory // (1024 ** 2)
            vram_used = torch.cuda.memory_reserved(0) // (1024 ** 2)
            info["vram_usage"] = f"{vram_used}/{vram_total} MB"
            info["gpu_name"] = gpu_info.name
        
        return info

class EnhancedWatermarkProcessor:
    """增强的水印处理器 - 支持 PowerPaint"""
    
    def __init__(self, base_processor: WatermarkProcessor):
        self.base_processor = base_processor
        self.powerpaint_processor = None
        self._load_powerpaint_processor()
    
    def _load_powerpaint_processor(self):
        """Load PowerPaint processor"""
        try:
            # Create PowerPaint config
            config = {
                'models': {
                    'powerpaint_model_path': './models/powerpaint_v2_real/realisticVisionV60B1_v51VAE'
                },
                'powerpaint_config': {
                    'use_fp16': True,
                    'enable_attention_slicing': True,
                    'enable_memory_efficient_attention': True,
                    'enable_vae_slicing': False
                }
            }
            
            self.powerpaint_processor = PowerPaintProcessor(config)
            logger.info("PowerPaint processor loaded successfully")
            
        except Exception as e:
            logger.warning(f"PowerPaint processor loading failed: {e}")
            logger.info("PowerPaint functionality will not be available")
    
    def process_image_with_params(self, 
                                image: Image.Image,
                                mask_model: str,
                                mask_params: Dict[str, Any],
                                inpaint_params: Dict[str, Any],
                                performance_params: Dict[str, Any],
                                transparent: bool = False) -> ProcessingResult:
        """使用详细参数处理图像"""
        start_time = time.time()
        
        try:
            logger.info("🚀 开始增强处理流程...")
            logger.info(f"🎭 Mask模型: {mask_model}")
            logger.info(f"⚙️ Inpaint参数: {inpaint_params}")
            
            # 生成mask
            if mask_model == "upload":
                mask_image = self._generate_uploaded_mask(image, mask_params)
            elif mask_model == "florence2":
                mask_image = self.base_processor.mask_generator.generate_mask(image, mask_params)
            else:  # custom
                mask_image = self.base_processor.mask_generator.generate_mask(image, mask_params)
            
            # 验证mask
            mask_array = np.array(mask_image)
            white_pixels = np.sum(mask_array > 128)
            total_pixels = mask_array.size
            mask_coverage = white_pixels / total_pixels * 100
            logger.info(f"🔍 Mask验证: 覆盖率={mask_coverage:.2f}%")
            
            # 应用处理
            if transparent:
                logger.info("🫥 应用透明处理...")
                result_image = self.base_processor._make_region_transparent(image, mask_image)
            else:
                inpaint_model = inpaint_params.get('inpaint_model', 'lama')
                
                if inpaint_model == 'powerpaint' and self.powerpaint_processor and self.powerpaint_processor.model_loaded:
                    logger.info("🎨 应用PowerPaint处理...")
                    result_array = self.powerpaint_processor.predict(image, mask_image, inpaint_params)
                    result_image = Image.fromarray(result_array)
                else:
                    logger.info("🎨 应用LaMA处理...")
                    result_array = self.base_processor.lama_processor.predict(image, mask_image, inpaint_params)
                    result_image = Image.fromarray(result_array)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ 处理完成，耗时: {processing_time:.2f}秒")
            
            return ProcessingResult(
                success=True,
                result_image=result_image,
                mask_image=mask_image,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 增强处理失败: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _generate_uploaded_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """处理上传的mask"""
        uploaded_mask = params.get('uploaded_mask')
        if not uploaded_mask:
            raise ValueError("No mask file uploaded")
        
        # Validate uploaded mask
        if hasattr(uploaded_mask, 'size') and uploaded_mask.size == 0:
            raise ValueError("Uploaded mask file is empty")
        
        try:
            logger.info(f"📂 处理上传的mask文件")
            logger.info(f"📏 原图尺寸: {image.size}")
            
            # 读取上传的mask
            if hasattr(uploaded_mask, 'read'):
                # Streamlit UploadedFile对象
                uploaded_mask.seek(0)  # 重置文件指针
                mask = Image.open(uploaded_mask)
            else:
                # 文件路径
                mask = Image.open(uploaded_mask)
                
        except Exception as e:
            raise ValueError(f"Failed to open uploaded mask file: {e}")
        
        # Validate mask after loading
        if mask.size[0] <= 0 or mask.size[1] <= 0:
            raise ValueError("Invalid mask dimensions")
        
        logger.info(f"📏 原始mask尺寸: {mask.size}")
        logger.info(f"🎨 原始mask模式: {mask.mode}")
        
        # 确保mask是灰度图像
        if mask.mode != 'L':
            mask = mask.convert('L')
            logger.info(f"🔄 转换mask为灰度模式: {mask.mode}")
        
        # 调整mask尺寸以匹配图像
        if mask.size != image.size:
            logger.info(f"📐 调整mask尺寸: {mask.size} -> {image.size}")
            mask = mask.resize(image.size, Image.LANCZOS)
        else:
            logger.info(f"✅ Mask尺寸已匹配: {mask.size}")
        
        # 检查mask内容
        mask_array = np.array(mask)
        white_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        mask_coverage = white_pixels / total_pixels * 100
        logger.info(f"🔍 Mask内容分析: 白色像素={white_pixels}, 总像素={total_pixels}, 覆盖率={mask_coverage:.2f}%")
        logger.info(f"📊 Mask像素值范围: {mask_array.min()} - {mask_array.max()}")
        
        # 应用额外的膨胀处理（如果需要）
        dilate_size = params.get('mask_dilate_kernel_size', 0)
        if dilate_size > 0:
            import cv2
            logger.info(f"🔧 应用膨胀处理: kernel_size={dilate_size}")
            mask_array = np.array(mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
            mask = Image.fromarray(mask_array, mode='L')
            
            # 检查膨胀后的mask
            white_pixels_after = np.sum(mask_array > 128)
            coverage_after = white_pixels_after / total_pixels * 100
            logger.info(f"🔍 膨胀后Mask分析: 白色像素={white_pixels_after}, 覆盖率={coverage_after:.2f}%")
        
        logger.info(f"✅ 最终mask尺寸: {mask.size}, 模式: {mask.mode}")
        return mask

class InferenceManager:
    """推理管理器"""
    
    def __init__(self, config_manager, config_path: Optional[str] = None):
        self.config_manager = config_manager
        self.config_path = config_path
        self.processor = None
        self.enhanced_processor = None
    
    def load_processor(self) -> bool:
        """加载处理器"""
        try:
            self.processor = WatermarkProcessor(self.config_path)
            self.enhanced_processor = EnhancedWatermarkProcessor(self.processor)
            logger.info("✅ Inference manager loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load inference manager: {e}")
            return False
    
    def process_image(self, 
                     image: Image.Image,
                     mask_model: str,
                     mask_params: Dict[str, Any],
                     inpaint_params: Dict[str, Any],
                     performance_params: Dict[str, Any],
                     transparent: bool = False) -> ProcessingResult:
        """处理图像"""
        if self.enhanced_processor is None:
            return ProcessingResult(
                success=False,
                error_message="Processor not loaded"
            )
        
        return self.enhanced_processor.process_image_with_params(
            image=image,
            mask_model=mask_model,
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params=performance_params,
            transparent=transparent
        )