"""
Base model class for AI watermark removal processors
"""

import logging
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for AI model processors"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.device = self._get_device()
        self.model_loaded = False
        self._models = []  # Track loaded models for cleanup
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup"""
        self.cleanup_resources()
    
    def cleanup_resources(self):
        """Clean up model resources"""
        try:
            # Clear model list
            for model in self._models:
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
            self._models.clear()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ CUDA cache cleared")
            
            self.model_loaded = False
            logger.info("✅ Model resources cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during resource cleanup: {e}")
    
    def register_model(self, model):
        """Register a model for automatic cleanup"""
        self._models.append(model)
        
    def _get_device(self) -> torch.device:
        """Get optimal device for model execution"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        return device
    
    @abstractmethod
    def predict(self, 
                image: Union[Image.Image, np.ndarray], 
                mask: Union[Image.Image, np.ndarray],
                custom_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Execute model prediction
        
        Args:
            image: Input image
            mask: Input mask
            custom_config: Custom configuration for this prediction
            
        Returns:
            Processed image as numpy array
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "device": str(self.device),
            "model_loaded": self.model_loaded,
            "config": self.config
        }
    
    def validate_inputs(self, 
                       image: Union[Image.Image, np.ndarray], 
                       mask: Union[Image.Image, np.ndarray]) -> tuple:
        """
        Validate and prepare inputs
        
        Args:
            image: Input image
            mask: Input mask
            
        Returns:
            Tuple of (validated_image, validated_mask)
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
            
        # Ensure RGB mode for image
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Ensure L mode for mask
        if mask.mode != 'L':
            mask = mask.convert('L')
            
        # Ensure same size
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.LANCZOS)
            
        return image, mask