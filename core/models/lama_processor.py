"""
LaMA Processor for Watermark Removal
Real LaMA model integration based on the original working implementation
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path
import io

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class LamaProcessor(BaseModel):
    """LaMA inpainting processor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_manager = None
        self._load_model()
    
    def _load_model(self):
        """Load LaMA model using iopaint"""
        try:
            from iopaint.model_manager import ModelManager
            from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
            
            model_name = self.config.get('models', {}).get('lama_model', 'lama')
            self.model_manager = ModelManager(name=model_name, device=str(self.device))
            self.register_model(self.model_manager)  # Register for cleanup
            
            # Store enums for later use
            self.HDStrategy = HDStrategy
            self.LDMSampler = LDMSampler
            self.Config = Config
            
            self.model_loaded = True
            logger.info(f"✅ LaMA model loaded successfully: {model_name}")
            logger.info(f"   Device: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load LaMA model: {e}")
            self.model_loaded = False
            raise
    
    def predict(self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """
        Perform LaMA inpainting
        
        Args:
            image: Input image (PIL or numpy array)
            mask: Mask image (PIL or numpy array)
            params: Processing parameters
            
        Returns:
            Processed image as numpy array (RGB)
        """
        try:
            # Default LaMA configuration
            default_config = {
                'ldm_steps': 50,
                'ldm_sampler': 'ddim',
                'hd_strategy': 'CROP',
                'hd_strategy_crop_margin': 64,
                'hd_strategy_crop_trigger_size': 800,
                'hd_strategy_resize_limit': 1600,
            }
            
            # Update with user parameters
            default_config.update(params)
            
            logger.info(f"🎨 LaMA processing with params: {default_config}")
            
            # Process image input - ensure RGB format for iopaint
            if isinstance(image, Image.Image):
                image_rgb = np.array(image.convert("RGB"))
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Assume RGB input
                    image_rgb = image
                else:
                    raise ValueError(f"Invalid image shape: {image.shape}")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Process mask input - ensure grayscale
            if isinstance(mask, Image.Image):
                mask_gray = np.array(mask.convert("L"))
            elif isinstance(mask, np.ndarray):
                if len(mask.shape) == 2:
                    mask_gray = mask
                elif len(mask.shape) == 3:
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                else:
                    raise ValueError(f"Invalid mask shape: {mask.shape}")
            else:
                raise ValueError(f"Unsupported mask type: {type(mask)}")
            
            logger.info(f"📸 Input image: shape={image_rgb.shape}, dtype={image_rgb.dtype}")
            logger.info(f"🎭 Input mask: shape={mask_gray.shape}, dtype={mask_gray.dtype}")
            logger.info(f"   Mask coverage: {np.sum(mask_gray > 128) / mask_gray.size * 100:.2f}%")
            
            # Validate mask
            if np.sum(mask_gray > 128) == 0:
                logger.warning("⚠️ WARNING: No white pixels in mask! LaMA will not perform any inpainting!")
                return image_rgb
            
            # Build LaMA config
            strategy_map = {
                'CROP': self.HDStrategy.CROP,
                'RESIZE': self.HDStrategy.RESIZE,
                'ORIGINAL': self.HDStrategy.ORIGINAL
            }
            
            sampler_map = {
                'ddim': self.LDMSampler.ddim,
                'plms': self.LDMSampler.plms
            }
            
            config = self.Config(
                ldm_steps=default_config['ldm_steps'],
                ldm_sampler=sampler_map.get(default_config['ldm_sampler'], self.LDMSampler.ddim),
                hd_strategy=strategy_map.get(default_config['hd_strategy'], self.HDStrategy.CROP),
                hd_strategy_crop_margin=default_config['hd_strategy_crop_margin'],
                hd_strategy_crop_trigger_size=default_config['hd_strategy_crop_trigger_size'],
                hd_strategy_resize_limit=default_config['hd_strategy_resize_limit'],
            )
            
            logger.info(f"⚙️ LaMA config: steps={config.ldm_steps}, sampler={config.ldm_sampler}, strategy={config.hd_strategy}")
            
            # Perform LaMA inpainting
            # iopaint expects RGB input and returns BGR output
            result_bgr = self.model_manager(image_rgb, mask_gray, config)
            
            logger.info(f"🤖 LaMA inference completed")
            logger.info(f"   Output shape: {result_bgr.shape}, dtype: {result_bgr.dtype}")
            
            # Handle data type conversion
            if result_bgr.dtype in [np.float64, np.float32]:
                result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
            
            # Convert BGR output to RGB for consistency
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            
            logger.info(f"✅ LaMA processing completed successfully")
            return result_rgb
            
        except Exception as e:
            logger.error(f"❌ LaMA processing failed: {e}")
            raise
    
    def unload_model(self):
        """Unload model to free memory"""
        self.cleanup_resources()
        if self.model_manager:
            self.model_manager = None
        logger.info("LaMA model unloaded")