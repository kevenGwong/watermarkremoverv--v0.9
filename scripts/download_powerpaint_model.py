#!/usr/bin/env python3
"""
Download Real PowerPaint v2 Model
This script downloads the correct PowerPaint v2 model with BrushNet architecture
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def check_git_lfs():
    """Check if git-lfs is installed"""
    try:
        result = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ Git LFS is available")
            return True
        else:
            logger.error("❌ Git LFS not found")
            return False
    except FileNotFoundError:
        logger.error("❌ Git not found")
        return False

def install_git_lfs():
    """Install git-lfs using conda"""
    try:
        logger.info("🔧 Installing git-lfs...")
        subprocess.run(['conda', 'install', '-y', 'git-lfs'], check=True)
        subprocess.run(['git', 'lfs', 'install'], check=True)
        logger.info("✅ Git LFS installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install git-lfs: {e}")
        return False

def download_powerpaint_model():
    """Download the real PowerPaint v2 model"""
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # PowerPaint v2 model path
    powerpaint_v2_path = models_dir / "powerpaint_v2_real"
    
    if powerpaint_v2_path.exists():
        logger.info(f"PowerPaint v2 model already exists at: {powerpaint_v2_path}")
        return True
    
    try:
        logger.info("📥 Downloading PowerPaint v2 model from HuggingFace...")
        logger.info("   Model: JunhaoZhuang/PowerPaint_v2")
        logger.info("   This may take several minutes...")
        
        # Clone the PowerPaint v2 model
        cmd = [
            'git', 'lfs', 'clone',
            'https://huggingface.co/JunhaoZhuang/PowerPaint_v2',
            str(powerpaint_v2_path)
        ]
        
        subprocess.run(cmd, check=True, cwd=models_dir.parent)
        
        logger.info(f"✅ PowerPaint v2 model downloaded to: {powerpaint_v2_path}")
        
        # Verify model structure
        required_components = ['brushnet', 'text_encoder', 'unet', 'vae', 'scheduler']
        missing_components = []
        
        for component in required_components:
            component_path = powerpaint_v2_path / component
            if not component_path.exists():
                missing_components.append(component)
        
        if missing_components:
            logger.warning(f"⚠️ Missing components: {missing_components}")
            return False
        else:
            logger.info("✅ PowerPaint v2 model structure verified")
            return True
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to download PowerPaint v2 model: {e}")
        return False

def update_config():
    """Update configuration to use the real PowerPaint model"""
    config_file = Path("web_config.yaml")
    
    if not config_file.exists():
        logger.warning("web_config.yaml not found, skipping config update")
        return
    
    try:
        # Read current config
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update PowerPaint model path
        old_path = "./models/powerpaint_v2/Realistic_Vision_V1.4-inpainting"
        new_path = "./models/powerpaint_v2_real"
        
        if old_path in content:
            content = content.replace(old_path, new_path)
            
            # Write updated config
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ web_config.yaml updated with real PowerPaint model path")
        else:
            logger.info("⚠️ PowerPaint model path not found in config, manual update may be needed")
            
    except Exception as e:
        logger.error(f"❌ Failed to update config: {e}")

def main():
    """Main function"""
    logger.info("🚀 PowerPaint v2 Real Model Download Script")
    logger.info("=" * 60)
    
    # Check git-lfs
    if not check_git_lfs():
        logger.info("Installing git-lfs...")
        if not install_git_lfs():
            logger.error("❌ Failed to install git-lfs. Please install manually:")
            logger.error("   conda install git-lfs")
            logger.error("   git lfs install")
            return False
    
    # Download model
    if download_powerpaint_model():
        logger.info("✅ PowerPaint v2 model download completed successfully!")
        
        # Update config
        update_config()
        
        logger.info("🎉 Setup complete! PowerPaint v2 is ready to use.")
        logger.info("")
        logger.info("📋 Next steps:")
        logger.info("1. Restart the web UI")
        logger.info("2. Select 'powerpaint' model in the interface")
        logger.info("3. PowerPaint will now use proper object removal with task prompts")
        
        return True
    else:
        logger.error("❌ PowerPaint v2 model download failed")
        logger.error("")
        logger.error("🔧 Manual download instructions:")
        logger.error("1. Install git-lfs: conda install git-lfs && git lfs install")
        logger.error("2. Clone model: git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint_v2 ./models/powerpaint_v2_real")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)