#!/usr/bin/env python3
"""
Web UI Launch Script for PowerPaint Integration Testing
"""

import sys
import subprocess
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def launch_streamlit_app():
    """Launch the Streamlit web app"""
    try:
        # Check if main.py exists
        main_path = project_root / "interfaces" / "web" / "main.py"
        if not main_path.exists():
            logger.error(f"Main web interface not found: {main_path}")
            return False
        
        logger.info("🚀 Launching PowerPaint-enabled WatermarkRemover Web UI...")
        logger.info(f"   Main app: {main_path}")
        logger.info("   Features: LaMA + PowerPaint inpainting models")
        logger.info("   Access URL: http://localhost:8501")
        
        # Launch streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(main_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "dark",
            "--server.headless", "false"
        ]
        
        logger.info("Starting Streamlit server...")
        subprocess.run(cmd, cwd=str(project_root))
        
    except KeyboardInterrupt:
        logger.info("🛑 Web UI stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to launch web UI: {e}")
        return False
    
    return True

def check_environment():
    """Check if environment is ready"""
    try:
        # Check PyTorch
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name()}")
        
        # Check Streamlit
        import streamlit
        logger.info(f"✅ Streamlit: {streamlit.__version__}")
        
        # Check diffusers
        import diffusers
        logger.info(f"✅ Diffusers: {diffusers.__version__}")
        
        # Check project modules
        from config.config import ConfigManager
        from core.inference import InferenceManager
        logger.info("✅ Project modules imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔍 Checking environment...")
    
    if not check_environment():
        logger.error("❌ Environment check failed")
        sys.exit(1)
    
    logger.info("✅ Environment check passed")
    
    # Launch the web UI
    success = launch_streamlit_app()
    sys.exit(0 if success else 1)