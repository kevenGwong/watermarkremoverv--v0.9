#!/usr/bin/env python3
"""
环境检查脚本
检查WatermarkRemover-AI项目的所有依赖和模型文件
"""

import os
import sys
import importlib
from pathlib import Path

def check_environment():
    """检查环境依赖和模型文件"""
    print("🔍 环境检查开始...")
    print("=" * 50)
    
    checks = {
        "python_version": check_python_version(),
        "torch": check_torch(),
        "iopaint": check_iopaint(),
        "saicinpainting": check_saicinpainting(),
        "other_deps": check_other_dependencies(),
        "models": check_model_files(),
        "config": check_config_files()
    }
    
    # 输出检查结果
    print("\n" + "=" * 50)
    print("📊 检查结果汇总:")
    print("=" * 50)
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
    
    success_count = sum(checks.values())
    total_count = len(checks)
    
    print(f"\n📈 总体状态: {success_count}/{total_count} 项检查通过")
    
    if all(checks.values()):
        print("\n🎉 环境检查通过！可以启动应用。")
        return True
    else:
        print("\n⚠️ 环境检查失败，请按修复方案解决。")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        return False

def check_torch():
    """检查PyTorch"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
        return True
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def check_iopaint():
    """检查IOPaint"""
    try:
        import iopaint
        # IOPaint可能没有__version__属性，检查其他方式
        try:
            version = iopaint.__version__
        except AttributeError:
            version = "已安装(版本未知)"
        print(f"✅ IOPaint: {version}")
        return True
    except ImportError:
        print("❌ IOPaint未安装")
        return False

def check_saicinpainting():
    """检查saicinpainting（可选）"""
    try:
        import saicinpainting
        print("✅ saicinpainting可用（LaMA相关功能可用）")
        return True
    except ImportError:
        print("⚠️ saicinpainting未安装（LaMA相关功能不可用，但主流程不受影响）")
        return True  # 降级为警告，不影响主流程

def check_other_dependencies():
    """检查其他依赖"""
    print("\n📦 其他依赖检查:")
    deps = [
        "segmentation_models_pytorch",
        "albumentations", 
        "transformers",
        "cv2",
        "PIL"
    ]
    
    all_ok = True
    for dep in deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
            all_ok = False
    
    return all_ok

def check_model_files():
    """检查模型文件"""
    print("\n📁 模型文件检查:")
    
    # 检查自定义Mask模型
    mask_path = Path("data/models/epoch=071-valid_iou=0.7267.ckpt")
    if mask_path.exists():
        size_mb = mask_path.stat().st_size / (1024*1024)
        print(f"✅ 自定义Mask模型: {size_mb:.1f}MB")
    else:
        print("❌ 自定义Mask模型缺失")
        return False
    
    # 检查IOPaint模型缓存
    cache_dir = Path.home() / ".cache/torch/hub/checkpoints"
    if cache_dir.exists():
        models = ["big-lama.pt", "zits-inpaint-0717.pt", "Places_512_FullData_G.pth"]
        found_models = []
        for model in models:
            if (cache_dir / model).exists():
                found_models.append(model)
        
        if found_models:
            print(f"✅ IOPaint模型缓存: {len(found_models)}个模型")
            for model in found_models:
                size_mb = (cache_dir / model).stat().st_size / (1024*1024)
                print(f"   - {model}: {size_mb:.1f}MB")
        else:
            print("❌ IOPaint模型缓存为空")
            return False
    else:
        print("❌ IOPaint模型缓存目录不存在")
        return False
    
    return True

def check_config_files():
    """检查配置文件"""
    print("\n⚙️ 配置文件检查:")
    
    config_files = ["web_config.yaml", "config/config.py"]
    all_ok = True
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file}")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1) 