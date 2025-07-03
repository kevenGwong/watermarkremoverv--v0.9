"""
Main application entry point for AI Watermark Remover
"""

import argparse
import sys
from pathlib import Path
import logging

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))
# 新增：确保watermark_remover_ai包路径在sys.path中
sys.path.insert(0, str(Path(__file__).parent / "watermark_remover_ai"))

# Import functions with graceful error handling
def safe_import():
    """Safely import modules with dependency checking"""
    try:
        from watermark_remover_ai.core.utils.config_utils import load_config, get_default_config
        from watermark_remover_ai.interfaces.cli.watermark_cli import main as cli_main
        from watermark_remover_ai.interfaces.web.frontend.streamlit_app import main as web_main
        return load_config, get_default_config, cli_main, web_main
    except ImportError as e:
        print(f"Dependency missing: {e}")
        print("Some features may not be available.")
        return None, None, None, None

load_config, get_default_config, cli_main, web_main = safe_import()


def setup_logging(log_level: str = "INFO"):
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="AI Watermark Remover - Modular watermark removal tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CLI mode - process single image
  python app.py cli input.jpg output.jpg

  # CLI mode - process directory
  python app.py cli input_dir/ output_dir/ --batch

  # Web interface
  python app.py web

  # Web interface on specific port
  python app.py web --port 8502
        """
    )
    
    # Global options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    # Interface selection
    subparsers = parser.add_subparsers(dest="interface", help="Interface mode")
    
    # CLI interface
    cli_parser = subparsers.add_parser("cli", help="Command line interface")
    cli_parser.add_argument("input", help="Input image or directory")
    cli_parser.add_argument("output", help="Output path or directory")
    cli_parser.add_argument("--batch", action="store_true", help="Process directory")
    cli_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    cli_parser.add_argument("--transparent", action="store_true", help="Create transparent regions")
    cli_parser.add_argument("--mask-method", choices=["florence", "custom", "upload"], 
                          default="auto", help="Mask generation method")
    cli_parser.add_argument("--max-bbox-percent", type=float, default=10.0,
                          help="Maximum bbox percentage for Florence-2")
    cli_parser.add_argument("--mask-threshold", type=float, default=0.5,
                          help="Mask threshold for custom model")
    cli_parser.add_argument("--detection-prompt", type=str, default="watermark",
                          help="Detection prompt for Florence-2")
    cli_parser.add_argument("--force-format", choices=["PNG", "JPEG", "WEBP"],
                          help="Force output format")
    
    # Web interface
    web_parser = subparsers.add_parser("web", help="Web interface (Streamlit)")
    web_parser.add_argument("--port", type=int, default=8501, help="Web server port")
    web_parser.add_argument("--host", type=str, default="localhost", help="Web server host")
    web_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    web_parser.add_argument("--use-existing", action="store_true", help="Use existing debug app directly")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        if load_config is None:
            # Fallback minimal config for direct usage
            config = {
                "interfaces": {
                    "web": {"port": 8501, "host": "localhost"},
                    "cli": {"show_progress": True}
                }
            }
            logger.info("Using minimal fallback configuration")
        elif args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            config = get_default_config()
            logger.info("Using default configuration")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Route to appropriate interface
    if args.interface == "cli":
        if cli_main is None:
            logger.error("CLI interface not available due to missing dependencies")
            sys.exit(1)
        try:
            cli_main(args, config)
        except KeyboardInterrupt:
            logger.info("CLI process interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"CLI execution failed: {e}")
            sys.exit(1)
    
    elif args.interface == "web":
        try:
            if args.use_existing:
                # Use existing debug app directly
                import subprocess
                debug_app_path = Path(__file__).parent / "watermark_web_app_debug.py"
                if not debug_app_path.exists():
                    logger.error(f"Debug app not found: {debug_app_path}")
                    sys.exit(1)
                
                cmd = [
                    sys.executable, "-m", "streamlit", "run",
                    str(debug_app_path),
                    "--server.port", str(args.port),
                    "--server.address", args.host,
                    "--server.headless", "true",
                    "--theme.base", "dark"
                ]
                
                logger.info(f"Starting existing debug app on {args.host}:{args.port}")
                subprocess.run(cmd, check=True)
            else:
                # Use new modular web interface
                web_main(args, config)
        except KeyboardInterrupt:
            logger.info("Web server stopped by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Web server failed: {e}")
            sys.exit(1)
    
    else:
        # No interface specified - show help
        parser.print_help()
        
        # Show available interfaces
        print("\nAvailable interfaces:")
        print("  cli  - Command line interface for batch processing")
        print("  web  - Web interface using Streamlit")
        print("\nUse 'python app.py <interface> --help' for interface-specific options")


if __name__ == "__main__":
    main()