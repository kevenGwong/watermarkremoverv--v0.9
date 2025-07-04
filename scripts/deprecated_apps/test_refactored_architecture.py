#!/usr/bin/env python3
"""
Test script to verify the refactored architecture works correctly
"""

import sys
from pathlib import Path
import logging

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all core modules can be imported"""
    logger.info("Testing module imports...")
    
    try:
        # Test core imports
        from watermark_remover_ai.core.utils.config_utils import get_default_config
        from watermark_remover_ai.core.utils.image_utils import ensure_pil_image
        from watermark_remover_ai.core.utils.mask_utils import create_binary_mask
        
        from watermark_remover_ai.core.models.base_model import BaseModel
        from watermark_remover_ai.core.models.florence_detector import FlorenceDetector
        from watermark_remover_ai.core.models.custom_segmenter import CustomSegmenter
        from watermark_remover_ai.core.models.lama_inpainter import LamaInpainter
        
        from watermark_remover_ai.core.processors.image_processor import ImageProcessor
        from watermark_remover_ai.core.processors.mask_generator import MaskGenerator
        from watermark_remover_ai.core.processors.watermark_remover import WatermarkRemover, ProcessingResult
        
        # Test interface imports
        from watermark_remover_ai.interfaces.cli.watermark_cli import main as cli_main
        
        logger.info("✅ All core module imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during imports: {e}")
        return False


def test_configuration():
    """Test configuration system"""
    logger.info("Testing configuration system...")
    
    try:
        from watermark_remover_ai.core.utils.config_utils import (
            get_default_config, validate_config, get_config_schema
        )
        
        # Test default config
        config = get_default_config()
        assert isinstance(config, dict)
        assert "models" in config
        assert "processing" in config
        assert "inpainting" in config
        
        # Test config validation
        schema = get_config_schema()
        validation = validate_config(config, schema)
        assert validation["is_valid"], f"Config validation failed: {validation['errors']}"
        
        logger.info("✅ Configuration system works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_image_utilities():
    """Test image processing utilities"""
    logger.info("Testing image utilities...")
    
    try:
        from watermark_remover_ai.core.utils.image_utils import ensure_pil_image, get_image_info
        from watermark_remover_ai.core.utils.mask_utils import create_binary_mask, validate_mask
        from PIL import Image
        import numpy as np
        
        # Test image creation and conversion
        test_image = Image.new("RGB", (100, 100), color="red")
        
        # Test image utilities
        converted_image = ensure_pil_image(test_image)
        assert isinstance(converted_image, Image.Image)
        
        image_info = get_image_info(test_image)
        assert image_info["width"] == 100
        assert image_info["height"] == 100
        
        # Test mask utilities
        test_mask_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        binary_mask = create_binary_mask(test_mask_array, threshold=0.5)
        assert binary_mask.shape == (100, 100)
        assert binary_mask.dtype == np.uint8
        
        # Test mask validation
        mask_image = Image.fromarray(binary_mask, mode='L')
        validation = validate_mask(mask_image, (100, 100))
        assert validation["is_valid"]
        
        logger.info("✅ Image utilities work correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Image utilities test failed: {e}")
        return False


def test_processors():
    """Test processor initialization"""
    logger.info("Testing processor initialization...")
    
    try:
        from watermark_remover_ai.core.processors.image_processor import ImageProcessor
        from watermark_remover_ai.core.processors.mask_generator import MaskGenerator
        from watermark_remover_ai.core.utils.config_utils import get_default_config
        
        config = get_default_config()
        
        # Test ImageProcessor
        image_processor = ImageProcessor(config)
        assert image_processor is not None
        
        # Test MaskGenerator
        mask_generator = MaskGenerator(config)
        assert mask_generator is not None
        
        # Test available methods
        methods = mask_generator.get_available_methods()
        assert "florence" in methods
        assert "custom" in methods
        assert "upload" in methods
        
        logger.info("✅ Processors initialize correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Processor test failed: {e}")
        return False


def test_cli_interface():
    """Test CLI interface structure"""
    logger.info("Testing CLI interface...")
    
    try:
        from watermark_remover_ai.interfaces.cli.watermark_cli import main as cli_main
        
        # Test that CLI main function exists and is callable
        assert callable(cli_main)
        
        logger.info("✅ CLI interface structure correct")
        return True
        
    except Exception as e:
        logger.error(f"❌ CLI interface test failed: {e}")
        return False


def test_app_entry_point():
    """Test main app entry point"""
    logger.info("Testing app entry point...")
    
    try:
        # Test that app.py exists and can be imported
        from pathlib import Path
        app_path = Path(__file__).parent / "app.py"
        assert app_path.exists(), "app.py not found"
        
        # Test config file exists
        config_path = Path(__file__).parent / "watermark_remover_ai" / "config" / "default_config.yaml"
        assert config_path.exists(), "default_config.yaml not found"
        
        logger.info("✅ App entry point structure correct")
        return True
        
    except Exception as e:
        logger.error(f"❌ App entry point test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("🧪 Starting architecture validation tests...")
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration System", test_configuration),
        ("Image Utilities", test_image_utilities),
        ("Processors", test_processors),
        ("CLI Interface", test_cli_interface),
        ("App Entry Point", test_app_entry_point),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' crashed: {e}")
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("🏁 TEST SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Refactored architecture is working correctly.")
        return 0
    else:
        logger.error(f"💥 {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())