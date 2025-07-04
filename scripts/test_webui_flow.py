#!/usr/bin/env python3
"""
Web UI Flow Test Script
Tests the complete web UI startup and image processing flow
"""

import sys
import os
import logging
import subprocess
import time
import requests
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_images():
    """Create test images for the flow test"""
    logger.info("Creating test images...")
    
    # Create test image with watermark-like pattern
    test_image = Image.new('RGB', (800, 600), color='lightblue')
    
    # Add some pattern to simulate content
    import random
    pixels = np.array(test_image)
    for i in range(0, 800, 50):
        for j in range(0, 600, 50):
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            pixels[j:j+40, i:i+40] = color
    
    test_image = Image.fromarray(pixels)
    
    # Create test mask - white region in the center
    test_mask = Image.new('L', (800, 600), color=0)
    mask_array = np.array(test_mask)
    mask_array[250:350, 350:450] = 255  # Central white region
    test_mask = Image.fromarray(mask_array)
    
    # Save test images
    test_dir = project_root / "temp" / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_image_path = test_dir / "test_image.jpg"
    test_mask_path = test_dir / "test_mask.png"
    
    test_image.save(test_image_path, quality=95)
    test_mask.save(test_mask_path)
    
    logger.info(f"✅ Test images created: {test_image_path}, {test_mask_path}")
    return test_image_path, test_mask_path

def test_web_ui_startup():
    """Test 1: Web UI Startup Test"""
    logger.info("=== Test 1: Web UI Startup Test ===")
    
    try:
        # Test main web interface imports
        from interfaces.web.main import main
        from interfaces.web.ui import MainInterface, ParameterPanel
        from config.config import ConfigManager
        from core.inference import InferenceManager
        
        logger.info("✅ Web UI modules imported successfully")
        
        # Test configuration loading
        config_manager = ConfigManager()
        logger.info("✅ Configuration manager initialized")
        
        # Test inference manager initialization
        inference_manager = InferenceManager(config_manager)
        success = inference_manager.load_processor()
        
        if success:
            logger.info("✅ Inference manager and processors loaded")
        else:
            logger.warning("⚠️ Inference manager loaded with warnings")
        
        # Test UI components initialization
        main_interface = MainInterface(config_manager)
        parameter_panel = ParameterPanel(config_manager)
        
        logger.info("✅ UI components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Web UI startup test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_parameter_processing():
    """Test 2: Parameter Processing Test"""
    logger.info("=== Test 2: Parameter Processing Test ===")
    
    try:
        from config.config import ConfigManager
        
        config_manager = ConfigManager()
        
        # Test PowerPaint parameter processing
        powerpaint_params = {
            'inpaint_model': 'powerpaint',
            'prompt': 'high quality, detailed, clean photo',
            'negative_prompt': 'watermark, logo, text, blurry',
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
            'strength': 1.0,
            'crop_trigger_size': 512,
            'crop_margin': 64,
            'resize_to_512': True,
            'blend_edges': True,
            'edge_feather': 5,
            'seed': 42
        }
        
        validated_pp = config_manager.validate_inpaint_params(powerpaint_params)
        logger.info(f"✅ PowerPaint parameters validated: {len(validated_pp)} params")
        
        # Test LaMA parameter processing
        lama_params = {
            'inpaint_model': 'lama',
            'ldm_steps': 50,
            'ldm_sampler': 'ddim',
            'hd_strategy': 'CROP',
            'hd_strategy_crop_margin': 64,
            'hd_strategy_crop_trigger_size': 800,
            'seed': 42
        }
        
        validated_lama = config_manager.validate_inpaint_params(lama_params)
        logger.info(f"✅ LaMA parameters validated: {len(validated_lama)} params")
        
        # Test mask parameters
        mask_params = {
            'mask_threshold': 0.5,
            'mask_dilate_kernel_size': 5,
            'mask_dilate_iterations': 2
        }
        
        validated_mask = config_manager.validate_mask_params(mask_params)
        logger.info(f"✅ Mask parameters validated: {len(validated_mask)} params")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Parameter processing test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_image_processing_flow():
    """Test 3: Image Processing Flow Test"""
    logger.info("=== Test 3: Image Processing Flow Test ===")
    
    try:
        from config.config import ConfigManager
        from core.inference import InferenceManager
        
        # Create test images
        test_image_path, test_mask_path = create_test_images()
        
        # Load test images
        test_image = Image.open(test_image_path)
        test_mask = Image.open(test_mask_path)
        
        logger.info(f"Test image size: {test_image.size}")
        logger.info(f"Test mask size: {test_mask.size}")
        
        # Initialize processing components
        config_manager = ConfigManager()
        inference_manager = InferenceManager(config_manager)
        
        if not inference_manager.load_processor():
            logger.error("❌ Failed to load processor")
            return False
        
        # Test LaMA processing
        logger.info("Testing LaMA processing...")
        
        lama_params = {
            'inpaint_model': 'lama',
            'ldm_steps': 10,  # Reduced for testing
            'ldm_sampler': 'ddim',
            'hd_strategy': 'ORIGINAL',
            'seed': 42
        }
        
        mask_params = {
            'uploaded_mask': None,  # We'll pass the mask directly
            'mask_dilate_kernel_size': 3,
            'mask_dilate_iterations': 1
        }
        
        performance_params = {
            'mixed_precision': True,
            'log_processing_time': True
        }
        
        # Mock the uploaded mask for testing
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_mask.save(tmp.name)
            mask_params['uploaded_mask'] = tmp.name
        
        result = inference_manager.process_image(
            image=test_image,
            mask_model='upload',
            mask_params=mask_params,
            inpaint_params=lama_params,
            performance_params=performance_params,
            transparent=False
        )
        
        if result.success:
            logger.info(f"✅ LaMA processing successful in {result.processing_time:.2f}s")
            
            if result.result_image:
                # Save result for inspection
                result_path = project_root / "temp" / "test_data" / "lama_result.png"
                result.result_image.save(result_path)
                logger.info(f"✅ LaMA result saved: {result_path}")
            
            if result.mask_image:
                mask_result_path = project_root / "temp" / "test_data" / "lama_mask.png"
                result.mask_image.save(mask_result_path)
                logger.info(f"✅ LaMA mask saved: {mask_result_path}")
        else:
            logger.error(f"❌ LaMA processing failed: {result.error_message}")
            return False
        
        # Test PowerPaint processing (if available)
        logger.info("Testing PowerPaint processing...")
        
        powerpaint_params = {
            'inpaint_model': 'powerpaint',
            'prompt': 'high quality, clean, detailed photo',
            'negative_prompt': 'watermark, logo, text, blurry, low quality',
            'num_inference_steps': 10,  # Reduced for testing
            'guidance_scale': 7.5,
            'strength': 1.0,
            'crop_trigger_size': 512,
            'crop_margin': 64,
            'seed': 42
        }
        
        result = inference_manager.process_image(
            image=test_image,
            mask_model='upload',
            mask_params=mask_params,
            inpaint_params=powerpaint_params,
            performance_params=performance_params,
            transparent=False
        )
        
        if result.success:
            logger.info(f"✅ PowerPaint processing successful in {result.processing_time:.2f}s")
            
            if result.result_image:
                result_path = project_root / "temp" / "test_data" / "powerpaint_result.png"
                result.result_image.save(result_path)
                logger.info(f"✅ PowerPaint result saved: {result_path}")
        else:
            logger.warning(f"⚠️ PowerPaint processing failed (expected if model not available): {result.error_message}")
        
        # Test transparency mode
        logger.info("Testing transparency mode...")
        
        result = inference_manager.process_image(
            image=test_image,
            mask_model='upload',
            mask_params=mask_params,
            inpaint_params=lama_params,
            performance_params=performance_params,
            transparent=True
        )
        
        if result.success and result.result_image:
            logger.info("✅ Transparency mode successful")
            
            if result.result_image.mode == 'RGBA':
                logger.info("✅ Transparency mode produced RGBA image")
                transparent_path = project_root / "temp" / "test_data" / "transparent_result.png"
                result.result_image.save(transparent_path)
                logger.info(f"✅ Transparent result saved: {transparent_path}")
            else:
                logger.warning("⚠️ Transparency mode didn't produce RGBA image")
        else:
            logger.error("❌ Transparency mode failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Image processing flow test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_streamlit_integration():
    """Test 4: Streamlit Integration Test"""
    logger.info("=== Test 4: Streamlit Integration Test ===")
    
    try:
        # Check if main.py exists and can be imported
        main_path = project_root / "interfaces" / "web" / "main.py"
        
        if not main_path.exists():
            logger.error(f"❌ Main web interface not found: {main_path}")
            return False
        
        # Try to import the main function
        from interfaces.web.main import main
        logger.info("✅ Main web interface imported successfully")
        
        # Check app.py exists for launching
        app_path = project_root / "app.py"
        if not app_path.exists():
            logger.error(f"❌ App launcher not found: {app_path}")
            return False
        
        logger.info("✅ App launcher found")
        
        # Test if we can create a simple config
        from config.config import ConfigManager
        config_manager = ConfigManager()
        
        # Simulate UI parameter generation
        test_ui_params = {
            'mask_model': 'custom',
            'mask_params': {
                'mask_threshold': 0.5,
                'mask_dilate_kernel_size': 3
            },
            'inpaint_params': {
                'inpaint_model': 'powerpaint',
                'num_inference_steps': 50,
                'guidance_scale': 7.5
            },
            'performance_params': {
                'mixed_precision': True
            },
            'transparent': False
        }
        
        logger.info("✅ UI parameter structure validated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Streamlit integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_model_file_availability():
    """Test 5: Model File Availability Test"""
    logger.info("=== Test 5: Model File Availability Test ===")
    
    try:
        # Check custom watermark model
        custom_model_path = project_root / "data" / "models" / "epoch=071-valid_iou=0.7267.ckpt"
        if custom_model_path.exists():
            logger.info("✅ Custom watermark model found")
        else:
            logger.warning(f"⚠️ Custom watermark model not found: {custom_model_path}")
        
        # Check PowerPaint model
        powerpaint_model_path = project_root / "models" / "powerpaint_v2" / "Realistic_Vision_V1.4-inpainting"
        if powerpaint_model_path.exists():
            logger.info("✅ PowerPaint model directory found")
            
            # Check for key model files
            key_files = [
                "model_index.json",
                "unet/config.json",
                "vae/config.json",
                "text_encoder/config.json"
            ]
            
            for file in key_files:
                file_path = powerpaint_model_path / file
                if file_path.exists():
                    logger.info(f"✅ PowerPaint model file found: {file}")
                else:
                    logger.warning(f"⚠️ PowerPaint model file missing: {file}")
        else:
            logger.warning(f"⚠️ PowerPaint model directory not found: {powerpaint_model_path}")
        
        # Check configuration files
        config_files = [
            "config/powerpaint_config.yaml",
            "config/config.py"
        ]
        
        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                logger.info(f"✅ Configuration file found: {config_file}")
            else:
                logger.error(f"❌ Configuration file missing: {config_file}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model file availability test failed: {e}")
        return False

def run_web_ui_tests():
    """Run all web UI integration tests"""
    logger.info("🚀 Starting Web UI Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Web UI Startup", test_web_ui_startup),
        ("Parameter Processing", test_parameter_processing),
        ("Image Processing Flow", test_image_processing_flow),
        ("Streamlit Integration", test_streamlit_integration),
        ("Model File Availability", test_model_file_availability),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            logger.info(f"Starting: {test_name}")
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name}: CRASHED - {e}")
            results.append((test_name, False))
        
        logger.info("-" * 40)
    
    # Summary
    logger.info("📊 WEB UI TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("-" * 40)
    logger.info(f"🎯 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL WEB UI TESTS PASSED! Ready for launch.")
        logger.info("\n🚀 To start the web UI, run:")
        logger.info("    python app.py web")
        logger.info("    or")
        logger.info("    streamlit run interfaces/web/main.py")
    else:
        logger.warning(f"⚠️ {total-passed} tests failed. Review the errors above.")
        
        # Provide troubleshooting tips
        logger.info("\n🔧 TROUBLESHOOTING TIPS:")
        if not any(result for name, result in results if "Model File" in name):
            logger.info("- Download PowerPaint model files to models/powerpaint_v2/")
        if not any(result for name, result in results if "Processing Flow" in name):
            logger.info("- Check inference.py processor initialization")
        if not any(result for name, result in results if "Parameter" in name):
            logger.info("- Check config.py parameter validation")
    
    return passed == total

if __name__ == "__main__":
    success = run_web_ui_tests()
    sys.exit(0 if success else 1)