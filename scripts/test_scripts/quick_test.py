"""
Quick test script to verify Web UI setup and basic functionality
"""
import sys
import os
import traceback
from PIL import Image
import numpy as np

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        'streamlit',
        'yaml', 
        'torch',
        'PIL',
        'cv2',
        'numpy',
        'transformers',
        'iopaint'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing modules: {missing_modules}")
        print("Run: pip install -r requirements_web.txt")
        return False
    
    print("✅ All imports successful")
    return True

def test_config():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        import yaml
        with open('web_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['app', 'processing', 'mask_generator', 'models']
        for section in required_sections:
            if section not in config:
                print(f"  ❌ Missing config section: {section}")
                return False
            print(f"  ✅ {section}")
        
        print("✅ Configuration valid")
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_model_files():
    """Test if model files exist"""
    print("\n🔍 Testing model files...")
    
    try:
        import yaml
        with open('web_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        model_path = config['mask_generator']['mask_model_path']
        
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  ✅ Custom model found: {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ Custom model not found: {model_path}")
            return False
        
        print("✅ Model files verified")
        return True
        
    except Exception as e:
        print(f"❌ Model check error: {e}")
        return False

def test_backend_import():
    """Test backend module import"""
    print("\n🔍 Testing backend import...")
    
    try:
        # Add current directory to path for import
        sys.path.insert(0, os.getcwd())
        
        from web_backend import WatermarkProcessor, CustomMaskGenerator, ProcessingResult
        print("  ✅ Backend modules imported")
        
        # Try to load config and create minimal processor (without models)
        print("  ✅ Backend import successful")
        return True
        
    except Exception as e:
        print(f"  ❌ Backend import failed: {e}")
        traceback.print_exc()
        return False

def test_gpu_availability():
    """Test GPU/CUDA availability"""
    print("\n🔍 Testing GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✅ CUDA available: {gpu_count} GPU(s)")
            print(f"  ✅ GPU 0: {gpu_name} ({memory_gb:.1f} GB)")
        else:
            print("  ⚠️  CUDA not available, will use CPU")
        
        print("✅ Device check complete")
        return True
        
    except Exception as e:
        print(f"❌ GPU check error: {e}")
        return False

def test_create_sample_image():
    """Create a sample test image"""
    print("\n🔍 Creating sample test image...")
    
    try:
        # Create a simple test image with watermark-like pattern
        img = Image.new('RGB', (400, 300), 'lightblue')
        
        # Add a "watermark" - black text-like pattern
        pixels = img.load()
        
        # Simple rectangular "watermark" in bottom right
        for x in range(320, 390):
            for y in range(250, 280):
                pixels[x, y] = (0, 0, 0)
        
        # Save test image
        os.makedirs('temp', exist_ok=True)
        test_path = 'temp/test_image.png'
        img.save(test_path)
        
        print(f"  ✅ Test image created: {test_path}")
        print(f"  ✅ Image size: {img.size}")
        
        return test_path
        
    except Exception as e:
        print(f"❌ Sample image creation failed: {e}")
        return None

def main():
    """Run all tests"""
    print("🚀 Quick Test for AI Watermark Remover Web UI")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_imports,
        test_config,
        test_model_files,
        test_backend_import,
        test_gpu_availability,
    ]
    
    for test in tests:
        if not test():
            all_passed = False
    
    # Create sample image for manual testing
    sample_image = test_create_sample_image()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready to start Web UI!")
        print("\nNext steps:")
        print("1. Start the web app: ./run_web_app.sh")
        print("2. Open browser: http://localhost:8501")
        if sample_image:
            print(f"3. Test with sample image: {sample_image}")
        
        print("\nAlternatively, run manual test:")
        print("python validate_consistency.py")
        
    else:
        print("❌ SOME TESTS FAILED - Please fix issues before starting")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements_web.txt")
        print("2. Check model files are in ./models/")
        print("3. Verify conda environment: conda activate py310aiwatermark")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)