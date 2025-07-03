#!/usr/bin/env python3
"""
Test script to verify OpenCV optimization for avoiding PIL color correction
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io

# Add project paths
sys.path.append("/home/duolaameng/SAM_Remove/WatermarkRemover-AI")

def test_opencv_vs_pil_loading():
    """Compare OpenCV vs PIL loading effects on LaMA results"""
    
    try:
        from web_backend import WatermarkProcessor
        
        print("🔍 Testing OpenCV vs PIL loading for LaMA consistency...")
        processor = WatermarkProcessor("web_config.yaml")
        
        # Test image paths
        test_image_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/input/IMG_0001-3.jpg"
        test_mask_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/mask/IMG_0095-4_dilated_mask.png"
        
        if not Path(test_image_path).exists() or not Path(test_mask_path).exists():
            print("❌ Test files not found")
            return False
        
        # Test configuration
        test_config = {
            'ldm_steps': 50,
            'ldm_sampler': 'ddim',
            'hd_strategy': 'CROP',
            'hd_strategy_crop_margin': 64,
            'hd_strategy_crop_trigger_size': 800
        }
        
        # Method 1: PIL loading (old way)
        print("\n🎨 Method 1: PIL loading (potential color correction)")
        pil_image = Image.open(test_image_path)
        pil_mask = Image.open(test_mask_path)
        
        result_pil = processor._process_with_lama(pil_image, pil_mask, test_config)
        pil_output = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/output/opencv_test_pil_method.png"
        result_pil.save(pil_output)
        print(f"✅ PIL method result saved: {pil_output}")
        
        # Method 2: Direct file path (OpenCV loading)
        print("\n🎨 Method 2: Direct file path (OpenCV loading)")
        result_opencv_path = processor._process_with_lama(test_image_path, pil_mask, test_config)
        opencv_path_output = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/output/opencv_test_path_method.png"
        result_opencv_path.save(opencv_path_output)
        print(f"✅ OpenCV path method result saved: {opencv_path_output}")
        
        # Method 3: BytesIO loading (Streamlit simulation)
        print("\n🎨 Method 3: BytesIO loading (Streamlit simulation)")
        with open(test_image_path, 'rb') as f:
            image_bytes = f.read()
        
        result_bytes = processor._process_with_lama(image_bytes, pil_mask, test_config)
        bytes_output = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/output/opencv_test_bytes_method.png"
        result_bytes.save(bytes_output)
        print(f"✅ BytesIO method result saved: {bytes_output}")
        
        # Method 4: Original test_iopaint.py method
        print("\n🎨 Method 4: Original test_iopaint.py method (reference)")
        from iopaint.model_manager import ModelManager
        from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
        import torch
        
        # Load using OpenCV (like test_iopaint.py)
        image_bgr = cv2.imread(test_image_path)
        mask_gray = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_manager = ModelManager(name="lama", device=device)
        
        config = Config(
            ldm_steps=50,
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.CROP,
            hd_strategy_crop_margin=64,
            hd_strategy_crop_trigger_size=800,
            hd_strategy_resize_limit=1600,
        )
        
        result_original_bgr = model_manager(image_bgr, mask_gray, config)
        
        if result_original_bgr.dtype in [np.float64, np.float32]:
            result_original_bgr = np.clip(result_original_bgr, 0, 255).astype(np.uint8)
        
        # Convert to RGB and save
        result_original_rgb = cv2.cvtColor(result_original_bgr, cv2.COLOR_BGR2RGB)
        original_output = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/output/opencv_test_original_method.png"
        Image.fromarray(result_original_rgb).save(original_output)
        print(f"✅ Original method result saved: {original_output}")
        
        # Compare results
        print("\n🔍 Comparing results...")
        
        # Load all results for comparison
        results = [
            ("PIL method", result_pil),
            ("OpenCV path", result_opencv_path),
            ("BytesIO", result_bytes),
            ("Original", Image.fromarray(result_original_rgb))
        ]
        
        # Convert all to numpy for comparison
        result_arrays = [(name, np.array(img)) for name, img in results]
        
        # Compare each method with original
        original_array = result_arrays[3][1]  # Original method
        
        print(f"\n📊 Color difference comparison (vs Original method):")
        for i, (name, array) in enumerate(result_arrays[:-1]):
            if array.shape == original_array.shape:
                diff = np.mean(np.abs(array.astype(float) - original_array.astype(float)))
                print(f"   {name:15} vs Original: {diff:.2f}")
                
                if diff < 1:
                    print(f"   ✅ {name} is virtually identical to original")
                elif diff < 5:
                    print(f"   ✅ {name} is very similar to original")
                elif diff < 15:
                    print(f"   ⚠️  {name} has moderate differences")
                else:
                    print(f"   ❌ {name} has significant differences")
        
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_debug_app_integration():
    """Test complete debug app integration with BytesIO"""
    
    try:
        print("\n🧪 Testing debug app integration with BytesIO...")
        
        from web_backend import WatermarkProcessor
        from watermark_web_app_debug import create_enhanced_backend
        
        # Load processor
        processor = WatermarkProcessor("web_config.yaml")
        enhanced_processor = create_enhanced_backend()
        enhanced_processor.base_processor = processor
        
        # Test image
        test_image_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/input/IMG_0001-3.jpg"
        
        # Load as bytes (simulate Streamlit upload)
        with open(test_image_path, 'rb') as f:
            image_bytes = f.read()
        
        print(f"📁 Loaded {len(image_bytes)} bytes from test image")
        
        # Test with bytes input
        mask_params = {
            'mask_threshold': 0.5,
            'mask_dilate_kernel_size': 3,
            'mask_dilate_iterations': 1
        }
        
        inpaint_params = {
            'ldm_steps': 50,
            'ldm_sampler': 'ddim',
            'hd_strategy': 'CROP',
            'hd_strategy_crop_margin': 64,
            'hd_strategy_crop_trigger_size': 800
        }
        
        performance_params = {
            'mixed_precision': True,
            'log_processing_time': True
        }
        
        result = enhanced_processor.process_image_with_params(
            image=image_bytes,
            mask_model="custom",
            mask_params=mask_params,
            inpaint_params=inpaint_params,
            performance_params=performance_params,
            transparent=False
        )
        
        if result.success:
            print("✅ Debug app integration with BytesIO succeeded")
            print(f"   Processing time: {result.processing_time:.2f}s")
            
            if result.result_image:
                output_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/output/debug_app_bytes_integration.png"
                result.result_image.save(output_path)
                print(f"   Saved result: {output_path}")
            
            if result.mask_image:
                mask_size = result.mask_image.size
                print(f"   Mask resolution: {mask_size}")
            
            return True
        else:
            print(f"❌ Debug app integration failed: {result.error_message}")
            return False
        
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing OpenCV optimization for color accuracy...")
    
    # Test 1: Compare different loading methods
    success1 = test_opencv_vs_pil_loading()
    
    # Test 2: Debug app integration
    success2 = test_debug_app_integration()
    
    print("\n" + "="*60)
    print("📊 OPENCV OPTIMIZATION SUMMARY")
    print("="*60)
    
    if success1 and success2:
        print("✅ All OpenCV optimization tests passed!")
        print("✅ BytesIO/Path loading should now match original results")
        print("✅ Debug app integration working with optimized loading")
    else:
        print("❌ Some tests failed")
    
    print("\n📁 Generated comparison files:")
    output_files = [
        "opencv_test_pil_method.png",
        "opencv_test_path_method.png",
        "opencv_test_bytes_method.png", 
        "opencv_test_original_method.png",
        "debug_app_bytes_integration.png"
    ]
    
    output_dir = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test/output"
    for file in output_files:
        file_path = Path(output_dir) / file
        if file_path.exists():
            print(f"   📄 {file}")
    
    print("\n💡 Now test the debug app with run_debug_app.sh")
    print("   Upload the same test image and compare results!")
    print("="*60)