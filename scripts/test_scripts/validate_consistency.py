"""
Validation script to ensure Web UI produces identical results to remwm.py
"""
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_images(img1: Image.Image, img2: Image.Image, tolerance: float = 0.01) -> dict:
    """
    Compare two PIL images and return similarity metrics
    
    Args:
        img1, img2: Images to compare
        tolerance: Acceptable difference threshold (0-1)
    
    Returns:
        dict with comparison results
    """
    # Ensure same size
    if img1.size != img2.size:
        return {
            "identical": False,
            "error": f"Size mismatch: {img1.size} vs {img2.size}"
        }
    
    # Convert to same format for comparison
    if img1.mode != img2.mode:
        if img1.mode == 'RGBA' and img2.mode == 'RGB':
            # Convert RGBA to RGB with white background
            rgb_img1 = Image.new('RGB', img1.size, (255, 255, 255))
            rgb_img1.paste(img1, mask=img1.split()[-1] if len(img1.split()) == 4 else None)
            img1 = rgb_img1
        elif img2.mode == 'RGBA' and img1.mode == 'RGB':
            rgb_img2 = Image.new('RGB', img2.size, (255, 255, 255))
            rgb_img2.paste(img2, mask=img2.split()[-1] if len(img2.split()) == 4 else None)
            img2 = rgb_img2
    
    # Convert to numpy arrays
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    
    # Calculate metrics
    mse = np.mean((arr1 - arr2) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # Normalized difference
    max_diff = np.max(np.abs(arr1 - arr2)) / 255.0
    mean_diff = np.mean(np.abs(arr1 - arr2)) / 255.0
    
    identical = max_diff <= tolerance
    
    return {
        "identical": identical,
        "mse": mse,
        "psnr": psnr,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "tolerance": tolerance
    }

def test_remwm_cli(input_image: str, output_dir: str, **kwargs) -> str:
    """
    Run remwm.py CLI and return output path
    """
    output_file = os.path.join(output_dir, "remwm_result.png")
    
    # Build command
    cmd = [
        "python", "remwm.py",
        input_image,
        output_file
    ]
    
    # Add options
    if kwargs.get('transparent', False):
        cmd.append("--transparent")
    if kwargs.get('overwrite', True):
        cmd.append("--overwrite")
    if 'max_bbox_percent' in kwargs:
        cmd.append(f"--max-bbox-percent={kwargs['max_bbox_percent']}")
    if 'force_format' in kwargs and kwargs['force_format']:
        cmd.append(f"--force-format={kwargs['force_format']}")
    
    logger.info(f"Running CLI: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("CLI execution successful")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"CLI execution failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise

def test_web_backend(input_image: str, output_dir: str, **kwargs) -> str:
    """
    Run web backend and return output path
    """
    from web_backend import WatermarkProcessor
    
    output_file = os.path.join(output_dir, "web_result.png")
    
    try:
        # Initialize processor
        processor = WatermarkProcessor("web_config.yaml")
        
        # Load and process image
        image = Image.open(input_image)
        result = processor.process_image(
            image=image,
            transparent=kwargs.get('transparent', False),
            max_bbox_percent=kwargs.get('max_bbox_percent', 10.0),
            force_format=kwargs.get('force_format', None)
        )
        
        if not result.success:
            raise Exception(f"Web processing failed: {result.error_message}")
        
        # Save result
        result.result_image.save(output_file)
        logger.info(f"Web backend result saved to: {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Web backend execution failed: {e}")
        raise

def validate_single_case(test_image: str, test_params: dict) -> dict:
    """
    Validate a single test case comparing CLI vs Web backend
    """
    logger.info(f"Validating: {test_image} with params {test_params}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Test CLI
            cli_output = test_remwm_cli(test_image, temp_dir, **test_params)
            
            # Test Web backend
            web_output = test_web_backend(test_image, temp_dir, **test_params)
            
            # Compare results
            if os.path.exists(cli_output) and os.path.exists(web_output):
                cli_img = Image.open(cli_output)
                web_img = Image.open(web_output)
                
                comparison = compare_images(cli_img, web_img, tolerance=0.02)
                
                return {
                    "success": True,
                    "comparison": comparison,
                    "test_params": test_params
                }
            else:
                return {
                    "success": False,
                    "error": f"Output files missing: CLI={os.path.exists(cli_output)}, Web={os.path.exists(web_output)}"
                }
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "test_params": test_params
            }

def create_test_images(output_dir: str) -> list:
    """
    Create synthetic test images with different characteristics
    """
    test_images = []
    
    # Simple image with corner watermark
    img1 = Image.new('RGB', (400, 300), 'white')
    # Add black rectangle as watermark
    for x in range(320, 380):
        for y in range(20, 60):
            img1.putpixel((x, y), (0, 0, 0))
    
    img1_path = os.path.join(output_dir, "test_corner_watermark.jpg")
    img1.save(img1_path)
    test_images.append(img1_path)
    
    # Image with center watermark
    img2 = Image.new('RGB', (500, 400), 'lightblue')
    for x in range(220, 280):
        for y in range(180, 220):
            img2.putpixel((x, y), (255, 0, 0))  # Red watermark
    
    img2_path = os.path.join(output_dir, "test_center_watermark.png")
    img2.save(img2_path)
    test_images.append(img2_path)
    
    # Small image
    img3 = Image.new('RGB', (150, 100), 'yellow')
    for x in range(100, 140):
        for y in range(10, 30):
            img3.putpixel((x, y), (0, 0, 255))  # Blue watermark
    
    img3_path = os.path.join(output_dir, "test_small_image.webp")
    img3.save(img3_path)
    test_images.append(img3_path)
    
    return test_images

def main():
    """
    Main validation routine
    """
    logger.info("Starting consistency validation between CLI and Web backend")
    
    # Create temporary directory for test images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images
        test_images = create_test_images(temp_dir)
        
        # Define test cases
        test_cases = [
            {"transparent": False, "max_bbox_percent": 10.0},
            {"transparent": True, "max_bbox_percent": 10.0},
            {"transparent": False, "max_bbox_percent": 15.0},
            {"transparent": False, "max_bbox_percent": 5.0, "force_format": "PNG"},
        ]
        
        results = []
        total_tests = len(test_images) * len(test_cases)
        passed_tests = 0
        
        logger.info(f"Running {total_tests} validation tests...")
        
        for test_image in test_images:
            for test_params in test_cases:
                result = validate_single_case(test_image, test_params)
                results.append(result)
                
                if result["success"] and result.get("comparison", {}).get("identical", False):
                    passed_tests += 1
                    logger.info(f"✅ PASS: {Path(test_image).name} with {test_params}")
                else:
                    logger.warning(f"❌ FAIL: {Path(test_image).name} with {test_params}")
                    if "comparison" in result:
                        comp = result["comparison"]
                        logger.warning(f"   Max diff: {comp.get('max_diff', 'N/A'):.4f}, PSNR: {comp.get('psnr', 'N/A'):.2f}")
                    if "error" in result:
                        logger.warning(f"   Error: {result['error']}")
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"VALIDATION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        # Detailed results
        logger.info(f"\nDetailed Results:")
        for i, result in enumerate(results):
            if result["success"] and "comparison" in result:
                comp = result["comparison"]
                status = "✅ IDENTICAL" if comp["identical"] else "❌ DIFFERENT"
                logger.info(f"Test {i+1:2d}: {status} (PSNR: {comp.get('psnr', 0):.2f}dB, Max diff: {comp.get('max_diff', 0):.4f})")
            else:
                logger.info(f"Test {i+1:2d}: ❌ ERROR - {result.get('error', 'Unknown error')}")
        
        return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)