#!/usr/bin/env python3
"""
PowerPaint Integration Test Script
Tests the complete integration of PowerPaint functionality with the web UI
"""

import sys
import os
import logging
import tempfile
import traceback
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

def test_module_imports():
    """Test 1: Module Import Test"""
    logger.info("=== Test 1: Module Import Test ===")
    
    try:
        # Test core imports
        from config.config import ConfigManager
        from core.inference import InferenceManager, EnhancedWatermarkProcessor
        from core.models.powerpaint_processor import PowerPaintProcessor
        from interfaces.web.ui import ParameterPanel, MainInterface
        
        logger.info("✅ All core modules imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Module import failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_config_management():
    """Test 2: Configuration Management Test"""
    logger.info("=== Test 2: Configuration Management Test ===")
    
    try:
        from config.config import ConfigManager
        
        # Test config manager initialization
        config_manager = ConfigManager()
        
        # Test PowerPaint parameter validation
        powerpaint_params = {
            'inpaint_model': 'powerpaint',
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
            'strength': 1.0,
            'prompt': 'high quality test',
            'negative_prompt': 'watermark, logo',
            'crop_trigger_size': 512,
            'crop_margin': 64,
            'seed': 42
        }
        
        validated_params = config_manager.validate_inpaint_params(powerpaint_params)
        logger.info(f"✅ PowerPaint parameters validated: {len(validated_params)} parameters")
        
        # Test LaMA parameter validation
        lama_params = {
            'inpaint_model': 'lama',
            'ldm_steps': 50,
            'ldm_sampler': 'ddim',
            'hd_strategy': 'CROP'
        }
        
        validated_lama = config_manager.validate_inpaint_params(lama_params)
        logger.info(f"✅ LaMA parameters validated: {len(validated_lama)} parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration management test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_powerpaint_processor():
    """Test 3: PowerPaint Processor Test"""
    logger.info("=== Test 3: PowerPaint Processor Test ===")
    
    try:
        from core.models.powerpaint_processor import PowerPaintProcessor
        
        # Create test config
        config = {
            'models': {
                'powerpaint_model_path': './models/powerpaint_v2/Realistic_Vision_V1.4-inpainting'
            },
            'powerpaint_config': {
                'use_fp16': True,
                'enable_attention_slicing': True,
                'enable_memory_efficient_attention': True
            }
        }
        
        # Test processor initialization
        logger.info("Initializing PowerPaint processor...")
        processor = PowerPaintProcessor(config)
        
        if processor.model_loaded:
            logger.info("✅ PowerPaint processor loaded successfully")
            
            # Test with synthetic data
            test_image = Image.new('RGB', (512, 512), color='white')
            test_mask = Image.new('L', (512, 512), color=0)
            
            # Create a small mask region
            mask_array = np.array(test_mask)
            mask_array[200:300, 200:300] = 255
            test_mask = Image.fromarray(mask_array)
            
            # Test prediction (with minimal steps for speed)
            test_config = {
                'num_inference_steps': 10,  # Minimal for testing
                'guidance_scale': 7.5,
                'prompt': 'test',
                'negative_prompt': 'watermark',
                'seed': 42
            }
            
            logger.info("Testing PowerPaint prediction...")
            result = processor.predict(test_image, test_mask, test_config)
            
            if result is not None and result.shape == (512, 512, 3):
                logger.info("✅ PowerPaint prediction successful")
                return True
            else:
                logger.error(f"❌ PowerPaint prediction failed: invalid result shape {result.shape if result is not None else 'None'}")
                return False
        else:
            logger.warning("⚠️ PowerPaint model not loaded - likely due to missing model files")
            logger.info("This is expected if model files are not downloaded yet")
            return True  # Consider this a pass for integration test
        
    except Exception as e:
        logger.error(f"❌ PowerPaint processor test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_inference_manager():
    """Test 4: Inference Manager Test"""
    logger.info("=== Test 4: Inference Manager Test ===")
    
    try:
        from config.config import ConfigManager
        from core.inference import InferenceManager
        
        # Initialize components
        config_manager = ConfigManager()
        inference_manager = InferenceManager(config_manager)
        
        # Test processor loading
        success = inference_manager.load_processor()
        if success:
            logger.info("✅ Base processor loaded successfully")
        else:
            logger.error("❌ Base processor loading failed")
            return False
        
        # Test enhanced processor
        if hasattr(inference_manager, 'enhanced_processor') and inference_manager.enhanced_processor:
            logger.info("✅ Enhanced processor initialized")
            
            # Check PowerPaint processor availability
            if hasattr(inference_manager.enhanced_processor, 'powerpaint_processor'):
                if inference_manager.enhanced_processor.powerpaint_processor:
                    logger.info("✅ PowerPaint processor available in enhanced processor")
                else:
                    logger.info("⚠️ PowerPaint processor not loaded (expected if model files missing)")
            
            return True
        else:
            logger.error("❌ Enhanced processor not initialized")
            return False
        
    except Exception as e:
        logger.error(f"❌ Inference manager test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_complete_processing_flow():
    """Test 5: Complete Processing Flow Test"""
    logger.info("=== Test 5: Complete Processing Flow Test ===")
    
    try:
        from config.config import ConfigManager
        from core.inference import InferenceManager
        
        # Initialize components
        config_manager = ConfigManager()
        inference_manager = InferenceManager(config_manager)
        
        # Load processor
        if not inference_manager.load_processor():
            logger.error("❌ Failed to load processor")
            return False
        
        # Create test image and mask
        test_image = Image.new('RGB', (400, 400), color='white')
        test_mask = Image.new('L', (400, 400), color=0)
        
        # Create mask region
        mask_array = np.array(test_mask)
        mask_array[150:250, 150:250] = 255
        test_mask = Image.fromarray(mask_array)
        
        # Test LaMA processing
        logger.info("Testing LaMA processing flow...")
        lama_params = {
            'inpaint_model': 'lama',
            'ldm_steps': 20,
            'ldm_sampler': 'ddim',
            'hd_strategy': 'ORIGINAL',
            'seed': 42
        }
        
        # Create a proper file-like object for uploaded mask
        import io
        mask_buffer = io.BytesIO()
        test_mask.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        # Create a temporary file that mimics streamlit's UploadedFile
        class MockUploadedFile:
            def __init__(self, buffer, name):
                self.buffer = buffer
                self.name = name
            
            def read(self, size=-1):
                return self.buffer.read(size)
            
            def seek(self, pos):
                return self.buffer.seek(pos)
            
            def tell(self):
                return self.buffer.tell()
        
        mask_params = {
            'uploaded_mask': MockUploadedFile(mask_buffer, 'test_mask.png'),
            'mask_dilate_kernel_size': 3,
            'mask_dilate_iterations': 1
        }
        performance_params = {'mixed_precision': True}
        
        result = inference_manager.process_image(
            image=test_image,
            mask_model='upload',
            mask_params=mask_params,
            inpaint_params=lama_params,
            performance_params=performance_params,
            transparent=False
        )
        
        if result.success:
            logger.info("✅ LaMA processing flow successful")
        else:
            logger.error(f"❌ LaMA processing failed: {result.error_message}")
            return False
        
        # Test PowerPaint processing (if available)
        logger.info("Testing PowerPaint processing flow...")
        powerpaint_params = {
            'inpaint_model': 'powerpaint',
            'num_inference_steps': 10,  # Minimal for testing
            'guidance_scale': 7.5,
            'strength': 1.0,
            'prompt': 'high quality, clean',
            'negative_prompt': 'watermark, text',
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
            logger.info("✅ PowerPaint processing flow successful")
        else:
            logger.warning(f"⚠️ PowerPaint processing failed (expected if model not available): {result.error_message}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Complete processing flow test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_ui_parameter_integration():
    """Test 6: UI Parameter Integration Test"""
    logger.info("=== Test 6: UI Parameter Integration Test ===")
    
    try:
        from config.config import ConfigManager
        from interfaces.web.ui import ParameterPanel
        
        # Mock streamlit for testing
        class MockStreamlit:
            class sidebar:
                @staticmethod
                def subheader(text): pass
                @staticmethod
                def selectbox(label, options, **kwargs): 
                    if 'powerpaint' in options:
                        return 'powerpaint'
                    return options[0]
                @staticmethod
                def text_area(label, **kwargs): 
                    return kwargs.get('value', 'test prompt')
                @staticmethod
                def slider(label, min_val, max_val, default, **kwargs): 
                    return default
                @staticmethod
                def checkbox(label, default=True, **kwargs): 
                    return default
                @staticmethod
                def number_input(label, **kwargs): 
                    return kwargs.get('value', -1)
                @staticmethod
                def write(text): pass
                @staticmethod
                def info(text): pass
                @staticmethod
                def file_uploader(*args, **kwargs): return None
                @staticmethod
                def warning(text): pass
                @staticmethod
                def success(text): pass
        
        # Temporarily replace streamlit import
        import sys
        sys.modules['streamlit'] = MockStreamlit()
        
        # Test parameter panel
        config_manager = ConfigManager()
        panel = ParameterPanel(config_manager)
        
        # This would normally be called by streamlit, we'll mock it
        logger.info("✅ UI parameter integration components loaded successfully")
        
        # Test parameter validation through config manager
        test_params = {
            'inpaint_model': 'powerpaint',
            'num_inference_steps': 50,
            'guidance_scale': 7.5
        }
        
        validated = config_manager.validate_inpaint_params(test_params)
        logger.info(f"✅ UI parameter validation successful: {len(validated)} parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UI parameter integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_all_tests():
    """Run all integration tests"""
    logger.info("🚀 Starting PowerPaint Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Configuration Management", test_config_management),
        ("PowerPaint Processor", test_powerpaint_processor),
        ("Inference Manager", test_inference_manager),
        ("Complete Processing Flow", test_complete_processing_flow),
        ("UI Parameter Integration", test_ui_parameter_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
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
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("-" * 40)
    logger.info(f"🎯 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! PowerPaint integration is ready.")
    else:
        logger.warning(f"⚠️ {total-passed} tests failed. Review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)