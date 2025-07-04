"""
PowerPaint processor for AI watermark removal using Stable Diffusion 1.5 inpainting
"""

import logging
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import math
import cv2

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class PowerPaintProcessor(BaseModel):
    """PowerPaint processor for watermark removal using SD 1.5 inpainting"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PowerPaint processor
        
        Args:
            config: Configuration dictionary containing PowerPaint parameters
        """
        super().__init__(config)
        self.pipe = None
        self.model_path = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load PowerPaint model"""
        try:
            from diffusers import StableDiffusionInpaintPipeline
            
            # Get model path
            self.model_path = self.config.get('models', {}).get(
                'powerpaint_model_path', 
                './models/powerpaint_v2_real/realisticVisionV60B1_v51VAE'
            )
            
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"PowerPaint model not found at: {self.model_path}")
            
            # Load model configuration
            powerpaint_config = self.config.get('powerpaint_config', {})
            use_fp16 = powerpaint_config.get('use_fp16', True)
            
            # Load pipeline
            logger.info(f"Loading PowerPaint model from: {self.model_path}")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
                use_safetensors=True,
                local_files_only=True
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Note: 标准 SD1.5 模型不支持 add_tokens，但我们可以使用 PowerPaint 的参数配置
            logger.info("Using PowerPaint-style parameters with standard SD1.5 inpainting")
            
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
            logger.info(f"✅ PowerPaint-style SD1.5 model loaded successfully")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Precision: {self.pipe.unet.dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load PowerPaint model: {e}")
            raise
    
    def predict(self, 
                image: Union[Image.Image, np.ndarray], 
                mask: Union[Image.Image, np.ndarray],
                custom_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Execute PowerPaint prediction with crop strategy for high-resolution images
        
        Args:
            image: Input image (RGB format)
            mask: Input mask (grayscale)
            custom_config: Custom configuration for this prediction
            
        Returns:
            Processed image as numpy array (RGB format)
        """
        if not self.model_loaded:
            raise RuntimeError("PowerPaint model not loaded")
        
        # Validate inputs
        image, mask = self.validate_inputs(image, mask)
        
        # Get processing parameters
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
            logger.info(f"Direct processing for image: {original_size}")
            result = self._process_direct(image, mask, params)
        
        return np.array(result)
    
    def _get_processing_params(self, custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get processing parameters with defaults for PowerPaint object removal"""
        default_params = {
            # PowerPaint v2 object removal task configuration
            'task': 'object-removal',  # PowerPaint task type
            'prompt': 'empty scene blur',  # PowerPaint v2 object removal prompt
            'negative_prompt': 'object, worst quality, low quality, normal quality, bad quality, blurry',
            'promptA': 'P_ctxt',  # PowerPaint task prompt A for context
            'promptB': 'P_ctxt',  # PowerPaint task prompt B for context
            'negative_promptA': 'P_obj',  # Negative task prompt A for object
            'negative_promptB': 'P_obj',  # Negative task prompt B for object
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
            'strength': 1.0,
            'seed': -1,
            'crop_trigger_size': 512,
            'crop_margin': 64,
            'resize_to_512': True,
            'blend_edges': True,
            'edge_feather': 5,
            'brushnet_conditioning_scale': 1.0,  # PowerPaint specific parameter
            'tradeoff': 1.0  # PowerPaint fitting degree parameter
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
        """Process image directly without cropping"""
        # Resize to model input size if needed
        if params.get('resize_to_512', True):
            processed_image, processed_mask = self._resize_for_model(image, mask, 512)
        else:
            processed_image, processed_mask = image, mask
        
        # Ensure dimensions are divisible by 8
        processed_image, processed_mask = self._ensure_divisible_by_8(processed_image, processed_mask)
        
        # Apply PowerPaint inpainting
        result = self._apply_inpainting(processed_image, processed_mask, params)
        
        # Resize back to original size if needed
        if result.size != image.size:
            result = result.resize(image.size, Image.LANCZOS)
        
        return result
    
    def _process_with_crop_strategy(self, 
                                   image: Image.Image, 
                                   mask: Image.Image, 
                                   params: Dict[str, Any]) -> Image.Image:
        """Process high-resolution image using crop strategy"""
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
            
            # Process crop
            processed_crop = self._process_direct(crop_image, crop_mask, params)
            
            # Blend back to result
            result = self._blend_crop_back(result, processed_crop, crop_image, crop_mask, 
                                         x1, y1, x2, y2, params)
        
        return result
    
    def _find_mask_regions(self, 
                          mask_array: np.ndarray, 
                          crop_size: int, 
                          margin: int) -> List[Tuple[int, int, int, int]]:
        """Find regions of interest based on mask"""
        logger.info(f"🔍 查找mask区域: mask_shape={mask_array.shape}, crop_size={crop_size}, margin={margin}")
        logger.info(f"📊 Mask像素值范围: {mask_array.min()} - {mask_array.max()}")
        
        # 确保mask是二值化的
        if mask_array.max() > 1:
            # 转换为二值mask
            binary_mask = (mask_array > 128).astype(np.uint8) * 255
            logger.info(f"🔄 转换为二值mask: 白色像素={np.sum(binary_mask > 0)}")
        else:
            binary_mask = mask_array
        
        # Find contours in mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"🎯 找到轮廓数量: {len(contours)}")
        
        if not contours:
            logger.warning("⚠️ 未找到任何轮廓，返回空列表")
            return []
        
        # Get bounding boxes for contours
        boxes = []
        height, width = binary_mask.shape
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            logger.info(f"📦 轮廓 {i+1}: 边界框=({x}, {y}, {w}, {h})")
            
            # Expand box with margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(width, x + w + margin)
            y2 = min(height, y + h + margin)
            
            # Ensure minimum crop size
            if x2 - x1 < crop_size or y2 - y1 < crop_size:
                # Center the crop around the contour
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                half_size = crop_size // 2
                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(width, x1 + crop_size)
                y2 = min(height, y1 + crop_size)
                
                # Adjust if we hit boundaries
                if x2 - x1 < crop_size:
                    x1 = max(0, x2 - crop_size)
                if y2 - y1 < crop_size:
                    y1 = max(0, y2 - crop_size)
                
                logger.info(f"📐 调整后的边界框 {i+1}: ({x1}, {y1}, {x2}, {y2})")
            
            boxes.append((x1, y1, x2, y2))
        
        # Merge overlapping boxes
        merged_boxes = self._merge_overlapping_boxes(boxes)
        logger.info(f"🔗 合并后边界框数量: {len(merged_boxes)}")
        
        return merged_boxes
    
    def _merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes"""
        if not boxes:
            return []
        
        # Sort by x1 coordinate
        boxes = sorted(boxes)
        merged = [boxes[0]]
        
        for current in boxes[1:]:
            last = merged[-1]
            
            # Check if boxes overlap
            if (current[0] <= last[2] and current[1] <= last[3] and 
                current[2] >= last[0] and current[3] >= last[1]):
                # Merge boxes
                merged[-1] = (
                    min(last[0], current[0]),
                    min(last[1], current[1]),
                    max(last[2], current[2]),
                    max(last[3], current[3])
                )
            else:
                merged.append(current)
        
        return merged
    
    def _blend_crop_back(self, 
                        result: Image.Image, 
                        processed_crop: Image.Image, 
                        original_crop: Image.Image,
                        crop_mask: Image.Image,
                        x1: int, y1: int, x2: int, y2: int, 
                        params: Dict[str, Any]) -> Image.Image:
        """Blend processed crop back to the result image"""
        # Convert to numpy for processing
        result_array = np.array(result)
        processed_crop_array = np.array(processed_crop)
        original_crop_array = np.array(original_crop)
        crop_mask_array = np.array(crop_mask)
        
        # Create blending mask
        if params.get('blend_edges', True):
            # Apply edge feathering
            feather_size = params.get('edge_feather', 5)
            blending_mask = self._create_feathered_mask(crop_mask_array, feather_size)
        else:
            blending_mask = crop_mask_array
        
        # Normalize mask to 0-1 range
        blending_mask = blending_mask.astype(np.float32) / 255.0
        
        # Blend only the mask region
        for c in range(3):  # RGB channels
            # Use the mask to determine blending
            result_array[y1:y2, x1:x2, c] = (
                blending_mask * processed_crop_array[:, :, c] + 
                (1 - blending_mask) * original_crop_array[:, :, c]
            ).astype(np.uint8)
        
        return Image.fromarray(result_array)
    
    def _create_feathered_mask(self, mask: np.ndarray, feather_size: int) -> np.ndarray:
        """Create a feathered mask for smooth blending"""
        # Apply Gaussian blur to create feathered edges
        feathered = cv2.GaussianBlur(mask, (feather_size * 2 + 1, feather_size * 2 + 1), 0)
        return feathered
    
    def _resize_for_model(self, 
                         image: Image.Image, 
                         mask: Image.Image, 
                         target_size: int) -> Tuple[Image.Image, Image.Image]:
        """Resize image and mask for model input while maintaining aspect ratio"""
        width, height = image.size
        
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        
        # Resize - 图像用LANCZOS保持质量，mask用NEAREST保持锐利边缘
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        resized_mask = mask.resize((new_width, new_height), Image.NEAREST)
        
        return resized_image, resized_mask
    
    def _ensure_divisible_by_8(self, 
                              image: Image.Image, 
                              mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Ensure image dimensions are divisible by 8 for Stable Diffusion"""
        width, height = image.size
        
        # Calculate new dimensions
        new_width = (width + 7) // 8 * 8
        new_height = (height + 7) // 8 * 8
        
        if new_width != width or new_height != height:
            # Resize to make divisible by 8 - 图像用LANCZOS，mask用NEAREST
            image = image.resize((new_width, new_height), Image.LANCZOS)
            mask = mask.resize((new_width, new_height), Image.NEAREST)
        
        return image, mask
    
    def _apply_inpainting(self, 
                         image: Image.Image, 
                         mask: Image.Image, 
                         params: Dict[str, Any]) -> Image.Image:
        """Apply PowerPaint-style inpainting with object removal parameters"""
        # Set random seed if specified
        if params.get('seed', -1) >= 0:
            generator = torch.Generator(device=self.device).manual_seed(params['seed'])
        else:
            generator = None
        
        # Ensure mask is binary and validate
        mask_array = np.array(mask)
        mask_array = (mask_array > 128).astype(np.uint8) * 255
        
        # 验证mask有效性
        white_pixels = np.sum(mask_array > 0)
        total_pixels = mask_array.size
        coverage = white_pixels / total_pixels * 100
        
        if white_pixels == 0:
            logger.error("❌ 致命错误: Mask完全为空，没有需要inpaint的区域！")
            raise ValueError("Empty mask: no white pixels found for inpainting")
        elif coverage < 0.1:
            logger.warning(f"⚠️ Mask覆盖率很低: {coverage:.3f}%，可能效果不明显")
        else:
            logger.info(f"✅ Mask验证通过: {white_pixels}像素 ({coverage:.2f}%)")
        
        mask = Image.fromarray(mask_array)
        
        logger.info(f"PowerPaint-style object removal inference: {image.size}, steps={params['num_inference_steps']}")
        
        # Get PowerPaint object removal parameters
        task = params.get('task', 'object-removal')
        
        if task == 'object-removal':
            # PowerPaint v2 object removal configuration
            # 注意：由于没有真正的 PowerPaint tokens，我们使用等效的文本提示
            prompt = params.get('prompt', 'empty scene blur, clean background, natural environment')
            negative_prompt = params.get('negative_prompt', 'object, person, animal, vehicle, building, text, watermark, logo, worst quality, low quality, normal quality, bad quality, blurry, artifacts')
        else:
            # Fallback to standard parameters
            prompt = params.get('prompt', 'high quality, detailed, clean, professional photo')
            negative_prompt = params.get('negative_prompt', 'watermark, logo, text, signature, blurry, low quality, artifacts')
        
        logger.info(f"PowerPaint-style {task}: prompt='{prompt}'")
        logger.info(f"  negative_prompt='{negative_prompt}'")
        
        # Run standard SD1.5 inpainting with PowerPaint-style parameters
        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                strength=params.get('strength', 1.0),
                generator=generator
            ).images[0]
            
            logger.info("✅ PowerPaint-style object removal completed successfully")
            
        except Exception as e:
            logger.error(f"PowerPaint-style inference failed: {e}")
            raise
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get PowerPaint model information"""
        info = super().get_model_info()
        info.update({
            "model_type": "PowerPaint_SD15_Inpainting",
            "model_path": self.model_path,
            "base_model": "Stable Diffusion 1.5",
            "framework": "diffusers",
            "supports_high_res": True,
            "crop_strategy": True,
            "input_format": "RGB",
            "max_resolution": "unlimited (with crop strategy)"
        })
        return info