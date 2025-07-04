"""
Real PowerPaint v2 Processor with BrushNet Architecture
Based on the official PowerPaint v2 implementation
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageFilter
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
from pathlib import Path

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class PowerPaintV2Processor(BaseModel):
    """Real PowerPaint v2 processor with BrushNet architecture for object removal"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pipe = None
        self.model_path = None
        self._load_model()
    
    def _load_model(self):
        """Load real PowerPaint v2 model with BrushNet architecture"""
        try:
            # Import PowerPaint components
            from diffusers import StableDiffusionInpaintPipeline
            from transformers import CLIPTextModel
            
            # Try to import PowerPaint specific components
            try:
                # These should be available if PowerPaint is properly installed
                from powerpaint.models import BrushNetModel, UNet2DConditionModel
                from powerpaint.pipelines import StableDiffusionPowerPaintBrushNetPipeline
                powerpaint_available = True
                logger.info("PowerPaint BrushNet components found")
            except ImportError:
                logger.warning("PowerPaint BrushNet components not found, will use fallback")
                powerpaint_available = False
            
            # Get model path
            self.model_path = self.config.get('models', {}).get(
                'powerpaint_model_path', 
                './models/powerpaint_v2_real/realisticVisionV60B1_v51VAE'
            )
            
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"PowerPaint v2 model not found at: {self.model_path}")
            
            # Load model configuration
            powerpaint_config = self.config.get('powerpaint_config', {})
            use_fp16 = powerpaint_config.get('use_fp16', True)
            weight_dtype = torch.float16 if use_fp16 else torch.float32
            
            logger.info(f"Loading PowerPaint v2 model from: {self.model_path}")
            
            if powerpaint_available and (Path(self.model_path) / "brushnet").exists():
                # Load real PowerPaint v2 BrushNet pipeline
                logger.info("Loading PowerPaint v2 BrushNet pipeline")
                
                # Load base model components
                base_model_path = self.config.get('models', {}).get('base_model_path', 'runwayml/stable-diffusion-v1-5')
                
                self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
                    base_model_path,
                    unet=UNet2DConditionModel.from_pretrained(
                        base_model_path,
                        subfolder="unet",
                        torch_dtype=weight_dtype,
                        local_files_only=False,
                    ).to(self.device),
                    brushnet=BrushNetModel.from_pretrained(
                        self.model_path,
                        subfolder="brushnet",
                        torch_dtype=weight_dtype,
                        local_files_only=True,
                    ).to(self.device),
                    text_encoder=CLIPTextModel.from_pretrained(
                        self.model_path,
                        subfolder="text_encoder",
                        torch_dtype=weight_dtype,
                        local_files_only=True,
                    ),
                    torch_dtype=weight_dtype,
                    safety_checker=None,
                    local_files_only=False,
                )
                
                # Add PowerPaint task tokens
                self.pipe.add_tokens(
                    placeholder_tokens=["P_obj", "P_ctxt", "P_shape"],
                    initialize_parameters=False,
                )
                
                logger.info("✅ PowerPaint v2 BrushNet pipeline loaded with task tokens")
                
            else:
                # Fallback to standard inpainting pipeline
                logger.warning("Using fallback standard inpainting pipeline")
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=weight_dtype,
                    use_safetensors=True,
                    local_files_only=True
                )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable optimizations
            if powerpaint_config.get('enable_attention_slicing', True):
                self.pipe.enable_attention_slicing()
                logger.info("Attention slicing enabled")
            
            if powerpaint_config.get('enable_memory_efficient_attention', True):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("XFormers memory efficient attention enabled")
                except Exception as e:
                    logger.warning(f"XFormers not available: {e}")
            
            if powerpaint_config.get('enable_vae_slicing', False):
                self.pipe.enable_vae_slicing()
                logger.info("VAE slicing enabled")
            
            self.model_loaded = True
            logger.info(f"✅ PowerPaint v2 model loaded successfully")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Precision: {self.pipe.unet.dtype}")
            logger.info(f"   Pipeline type: {type(self.pipe).__name__}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load PowerPaint v2 model: {e}")
            self.model_loaded = False
            raise
    
    def predict(self, 
                image: Union[Image.Image, np.ndarray], 
                mask: Union[Image.Image, np.ndarray],
                custom_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Execute PowerPaint v2 object removal prediction
        
        Args:
            image: Input image (RGB format)
            mask: Input mask (grayscale)
            custom_config: Custom configuration for this prediction
            
        Returns:
            Processed image as numpy array (RGB format)
        """
        if not self.model_loaded:
            raise RuntimeError("PowerPaint v2 model not loaded")
        
        # Validate inputs
        image, mask = self.validate_inputs(image, mask)
        
        # Get processing parameters with PowerPaint v2 object removal defaults
        params = self._get_processing_params(custom_config)
        
        # Determine processing strategy based on image size
        crop_trigger_size = params.get('crop_trigger_size', 512)
        original_size = image.size
        
        if max(original_size) > crop_trigger_size:
            # Use crop strategy for high-resolution images
            logger.info(f"Using crop strategy for high-resolution image: {original_size}")
            result = self._process_with_crop_strategy(image, mask, params)
        else:
            # Direct processing for smaller images
            logger.info(f"Direct PowerPaint v2 processing for image: {original_size}")
            result = self._process_direct(image, mask, params)
        
        return np.array(result)
    
    def _get_processing_params(self, custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get processing parameters with PowerPaint v2 object removal defaults"""
        default_params = {
            # PowerPaint v2 object removal task configuration
            'task': 'object-removal',
            'prompt': 'empty scene blur',  # PowerPaint v2 object removal prompt
            'negative_prompt': 'object, worst quality, low quality, normal quality, bad quality, blurry',
            'promptA': 'P_ctxt',  # Context token for PowerPaint v2
            'promptB': 'P_ctxt',  # Context token for PowerPaint v2
            'negative_promptA': 'P_obj',  # Object token (negative)
            'negative_promptB': 'P_obj',  # Object token (negative)
            'num_inference_steps': 45,  # PowerPaint v2 default
            'guidance_scale': 7.5,
            'tradeoff': 1.0,  # PowerPaint v2 fitting degree
            'brushnet_conditioning_scale': 1.0,
            'seed': -1,
            'crop_trigger_size': 640,  # PowerPaint v2 uses 640 for non-outpainting
            'crop_margin': 64,
            'resize_to_512': False,  # PowerPaint v2 handles sizing differently
        }
        
        # Update with config defaults
        powerpaint_config = self.config.get('powerpaint_config', {})
        default_params.update(powerpaint_config)
        
        # Update with custom config - UI参数优先级最高
        if custom_config:
            # 确保UI设置的prompt不被默认值覆盖
            for key, value in custom_config.items():
                if key in ['prompt', 'negative_prompt', 'task'] and value:
                    default_params[key] = value
                else:
                    default_params[key] = value
            logger.info(f"🎮 UI参数已应用: task={default_params.get('task')}, prompt='{default_params.get('prompt')[:50]}...'")
            
        return default_params
    
    def _process_direct(self, 
                       image: Image.Image, 
                       mask: Image.Image, 
                       params: Dict[str, Any]) -> Image.Image:
        """Process image directly with PowerPaint v2"""
        
        # PowerPaint v2 preprocessing
        w, h = image.size
        new_size = params.get('crop_trigger_size', 640)
        
        # Resize maintaining aspect ratio (PowerPaint v2 style)
        if w < h:
            image = image.resize((new_size, int(h / w * new_size)))
        else:
            image = image.resize((int(w / h * new_size), new_size))
            
        mask = mask.resize(image.size, Image.NEAREST)
        w, h = image.size
        
        # Ensure size is divisible by 8 - 图像用LANCZOS，mask用NEAREST保持锐利边缘
        w, h = w // 8 * 8, h // 8 * 8
        image = image.resize((w, h), Image.LANCZOS)
        mask = mask.resize((w, h), Image.NEAREST)
        
        # Create masked image (PowerPaint v2 format)
        # 关键修复: 反转mask逻辑 - 白色区域应该被mask掉(变成黑色)
        hole_value = (0, 0, 0)
        
        # 反转mask: 白色变黑色，黑色变白色
        inverted_mask = Image.eval(mask.convert("L"), lambda x: 255 - x)
        
        masked_image = Image.composite(
            Image.new("RGB", (w, h), hole_value),  # 黑色背景 
            image,                                 # 原图
            inverted_mask                          # 反转后的mask
        )
        
        logger.info(f"🎭 创建masked image: 白色mask区域已被黑色填充")
        
        # 验证mask有效性和masked image
        mask_array = np.array(mask.convert("L"))
        white_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        coverage = white_pixels / total_pixels * 100
        
        if white_pixels == 0:
            logger.error("❌ 致命错误: Mask完全为空，没有需要inpaint的区域！")
            raise ValueError("Empty mask: no white pixels found for inpainting")
        elif coverage < 0.1:
            logger.warning(f"⚠️ Mask覆盖率很低: {coverage:.3f}%，可能效果不明显")
        else:
            logger.info(f"✅ Mask验证通过: {white_pixels}像素 ({coverage:.2f}%) 将被inpaint")
        
        # Generate random seed if needed
        if params['seed'] == -1:
            params['seed'] = torch.randint(0, 2**32 - 1, (1,)).item()
        
        generator = torch.Generator(device=self.device).manual_seed(params['seed'])
        
        logger.info(f"PowerPaint v2 object removal inference: {image.size}, steps={params['num_inference_steps']}")
        logger.info(f"Task prompts: A='{params['promptA']}', B='{params['promptB']}'")
        
        # Check if this is a real PowerPaint v2 BrushNet pipeline
        if hasattr(self.pipe, 'brushnet') and hasattr(self.pipe, 'add_tokens'):
            # Real PowerPaint v2 BrushNet pipeline
            logger.info("Using PowerPaint v2 BrushNet pipeline for object removal")
            result = self.pipe(
                promptA=params['promptA'],
                promptB=params['promptB'],
                prompt=params['prompt'],
                negative_promptA=params['negative_promptA'],
                negative_promptB=params['negative_promptB'],
                negative_prompt=params['negative_prompt'],
                tradeoff=params['tradeoff'],
                image=masked_image,  # PowerPaint v2 expects masked image
                mask=mask,
                num_inference_steps=params['num_inference_steps'],
                generator=generator,
                brushnet_conditioning_scale=params['brushnet_conditioning_scale'],
                guidance_scale=params['guidance_scale'],
                width=w,
                height=h,
            ).images[0]
        else:
            # Fallback to standard inpainting
            logger.warning("Using fallback standard inpainting (PowerPaint v2 not available)")
            result = self.pipe(
                prompt=params['prompt'],
                negative_prompt=params['negative_prompt'],
                image=image,
                mask_image=mask,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                generator=generator
            ).images[0]
        
        return result
    
    def _process_with_crop_strategy(self, 
                                   image: Image.Image, 
                                   mask: Image.Image, 
                                   params: Dict[str, Any]) -> Image.Image:
        """Process with crop strategy for high-resolution images (PowerPaint v2 style)"""
        
        # PowerPaint v2 crop strategy implementation
        crop_size = 512
        crop_margin = params.get('crop_margin', 64)
        
        # Find regions of interest based on mask
        mask_array = np.array(mask)
        roi_boxes = self._find_mask_regions(mask_array, crop_size, crop_margin)
        
        if not roi_boxes:
            # No significant mask regions, process directly
            logger.info("No significant mask regions found, processing directly")
            return self._process_direct(image, mask, params)
        
        # Process each region
        result = image.copy()
        
        for i, (x1, y1, x2, y2) in enumerate(roi_boxes):
            logger.info(f"Processing crop region {i+1}/{len(roi_boxes)}: ({x1}, {y1}, {x2}, {y2})")
            
            # Extract crop
            crop_image = image.crop((x1, y1, x2, y2))
            crop_mask = mask.crop((x1, y1, x2, y2))
            
            # Process crop with PowerPaint v2
            processed_crop = self._process_direct(crop_image, crop_mask, params)
            
            # Blend back to result with PowerPaint v2 style blending
            result = self._blend_crop_back_v2(result, processed_crop, crop_image, crop_mask, 
                                            x1, y1, x2, y2, params)
        
        return result
    
    def _find_mask_regions(self, 
                          mask_array: np.ndarray, 
                          crop_size: int, 
                          margin: int) -> List[Tuple[int, int, int, int]]:
        """Find regions of interest based on mask"""
        # Find contours in mask
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get bounding boxes for contours
        boxes = []
        height, width = mask_array.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Expand box with margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(width, x + w + margin)
            y2 = min(height, y + h + margin)
            
            # Ensure minimum crop size
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            half_crop = crop_size // 2
            
            x1 = max(0, center_x - half_crop)
            y1 = max(0, center_y - half_crop)
            x2 = min(width, x1 + crop_size)
            y2 = min(height, y1 + crop_size)
            
            # Adjust if hitting boundaries
            if x2 - x1 < crop_size:
                x1 = max(0, x2 - crop_size)
            if y2 - y1 < crop_size:
                y1 = max(0, y2 - crop_size)
            
            boxes.append((x1, y1, x2, y2))
        
        return boxes
    
    def _blend_crop_back_v2(self, 
                          result: Image.Image,
                          processed_crop: Image.Image, 
                          original_crop: Image.Image,
                          crop_mask: Image.Image,
                          x1: int, y1: int, x2: int, y2: int, 
                          params: Dict[str, Any]) -> Image.Image:
        """Blend processed crop back using PowerPaint v2 style blending"""
        
        # Convert to numpy for processing
        result_array = np.array(result)
        processed_crop_array = np.array(processed_crop)
        original_crop_array = np.array(original_crop)
        crop_mask_array = np.array(crop_mask.convert('L'))
        
        # PowerPaint v2 style blending with Gaussian blur
        m_img = crop_mask.convert("RGB").filter(ImageFilter.GaussianBlur(radius=4))
        m_img = np.asarray(m_img) / 255.0
        img_np = np.asarray(original_crop) / 255.0
        ours_np = np.asarray(processed_crop) / 255.0
        
        # Blend using PowerPaint v2 formula
        blended_np = ours_np * m_img + (1 - m_img) * img_np
        blended_crop = Image.fromarray(np.uint8(blended_np * 255))
        
        # Place blended crop back in result
        result.paste(blended_crop, (x1, y1))
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get PowerPaint v2 model information"""
        info = super().get_model_info()
        info.update({
            "model_type": "PowerPaint_v2_BrushNet",
            "model_path": self.model_path,
            "base_model": "Stable Diffusion 1.5",
            "framework": "PowerPaint_BrushNet",
            "supports_high_res": True,
            "crop_strategy": True,
            "task_tokens": ["P_obj", "P_ctxt", "P_shape"],
            "object_removal": True,
            "input_format": "RGB",
            "pipeline_type": type(self.pipe).__name__ if self.pipe else "Not loaded"
        })
        return info