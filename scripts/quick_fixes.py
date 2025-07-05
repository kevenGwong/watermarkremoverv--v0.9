#!/usr/bin/env python3
"""
WatermarkRemover-AI 快速修复脚本
解决CUDA内存管理、LaMA降级和颜色空间问题
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def fix_cuda_memory_management():
    """修复CUDA内存管理问题"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 修复CUDA内存管理...")
    
    # 1. 创建环境变量设置脚本
    env_script = project_root / "scripts" / "set_cuda_env.sh"
    env_content = """#!/bin/bash
# CUDA内存管理优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

# 启动应用
cd "$(dirname "$0")/.."
streamlit run interfaces/web/main.py --server.port 8501
"""
    
    with open(env_script, 'w') as f:
        f.write(env_content)
    
    # 设置执行权限
    os.chmod(env_script, 0o755)
    logger.info(f"✅ 创建CUDA环境脚本: {env_script}")
    
    # 2. 创建内存监控工具
    memory_monitor = project_root / "core" / "utils" / "memory_monitor.py"
    monitor_content = '''"""
GPU内存监控工具
"""

import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """GPU内存监控器"""
    
    @staticmethod
    def get_gpu_memory_info() -> Optional[Dict[str, float]]:
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return None
            
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': free,
                'usage_percent': (reserved / total) * 100
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return None
    
    @staticmethod
    def clear_cache():
        """清理CUDA缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✅ CUDA cache cleared")
    
    @staticmethod
    def check_memory_warning(threshold_gb: float = 2.0) -> bool:
        """检查内存是否不足"""
        info = MemoryMonitor.get_gpu_memory_info()
        if info and info['free_gb'] < threshold_gb:
            logger.warning(f"⚠️ GPU memory low: {info['free_gb']:.2f}GB free")
            return True
        return False
    
    @staticmethod
    def log_memory_status():
        """记录内存状态"""
        info = MemoryMonitor.get_gpu_memory_info()
        if info:
            logger.info(f"GPU Memory: {info['allocated_gb']:.2f}GB allocated, "
                       f"{info['reserved_gb']:.2f}GB reserved, "
                       f"{info['free_gb']:.2f}GB free")
'''
    
    with open(memory_monitor, 'w') as f:
        f.write(monitor_content)
    
    logger.info(f"✅ 创建内存监控工具: {memory_monitor}")

def fix_lama_processor():
    """修复LaMA处理器，实现可选支持"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 修复LaMA处理器...")
    
    # 创建可选的LaMA处理器
    lama_fix = project_root / "core" / "models" / "lama_processor_fixed.py"
    lama_content = '''"""
LaMA处理器修复版本 - 支持可选安装
"""

import logging
import time
import yaml
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)

class OptionalLamaProcessor:
    """可选的LaMA处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = None
        self.model_loaded = False
        self.available = self._check_dependencies()
        
        if self.available:
            self._load_model()
        else:
            logger.warning("⚠️ LaMA dependencies not available, processor disabled")
    
    def _check_dependencies(self) -> bool:
        """检查LaMA依赖是否可用"""
        try:
            import saicinpainting
            return True
        except ImportError:
            logger.warning("❌ saicinpainting not available")
            return False
        except Exception as e:
            logger.warning(f"❌ LaMA dependency check failed: {e}")
            return False
    
    def _load_model(self):
        """加载LaMA模型"""
        if not self.available:
            return
            
        try:
            import torch
            import cv2
            from saicinpainting.evaluation.data import pad_img_to_modulo
            from saicinpainting.evaluation.utils import move_to_device
            from saicinpainting.evaluation.data import load_image, load_mask, get_img
            from saicinpainting.training.trainers import load_checkpoint
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 获取模型配置
            model_config = self.config.get('models', {})
            model_path = model_config.get('lama_model', 'lama')
            
            # 加载模型
            train_config_path = Path(model_path) / 'config.yaml'
            model_path = Path(model_path) / 'models' / 'best.ckpt'
            
            if not model_path.exists():
                logger.error(f"LaMA model not found: {model_path}")
                raise FileNotFoundError(f"LaMA model not found: {model_path}")
            
            with open(train_config_path, 'r') as f:
                train_config = yaml.safe_load(f)
            
            train_config['model']['input_channels'] = 4
            train_config['model']['output_channels'] = 3
            
            # 创建模型
            from saicinpainting.training.data.datasets import make_default_val_dataset
            from saicinpainting.training.models import make_model
            
            model = make_model(train_config['model'], kind='inpainting')
            model.to(self.device)
            
            # 加载checkpoint
            checkpoint = load_checkpoint(train_config, model_path, strict=False, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
            
            self.model = model
            self.model_loaded = True
            logger.info(f"✅ LaMA model loaded from: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load LaMA model: {e}")
            self.model = None
            self.model_loaded = False
    
    def predict(self, image: Image.Image, mask: Image.Image, config: Dict[str, Any] = None) -> np.ndarray:
        """使用LaMA进行inpainting"""
        if not self.available:
            raise RuntimeError("LaMA not available - dependencies missing")
            
        if not self.model_loaded:
            raise RuntimeError("LaMA model not loaded")
        
        try:
            import torch
            import cv2
            from saicinpainting.evaluation.data import pad_img_to_modulo
            from saicinpainting.evaluation.utils import move_to_device
            from saicinpainting.evaluation.data import load_image, load_mask, get_img
            
            # 使用默认配置或自定义配置
            if config is None:
                config = {}
            
            # 获取参数
            ldm_steps = config.get('ldm_steps', 50)
            ldm_sampler = config.get('ldm_sampler', 'ddim')
            hd_strategy = config.get('hd_strategy', 'CROP')
            hd_strategy_crop_margin = config.get('hd_strategy_crop_margin', 64)
            hd_strategy_crop_trigger_size = config.get('hd_strategy_crop_trigger_size', 1024)
            hd_strategy_resize_limit = config.get('hd_strategy_resize_limit', 2048)
            
            # 转换图像格式 - LaMA需要BGR输入
            image_array = np.array(image.convert("RGB"))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # RGB to BGR
            mask_array = np.array(mask.convert("L"))
            
            # 确保mask是二值的
            mask_array = (mask_array > 128).astype(np.uint8) * 255
            
            # 处理高分辨率图像
            if hd_strategy == 'CROP' and max(image_array.shape[:2]) > hd_strategy_crop_trigger_size:
                # 裁剪策略
                image_array, mask_array = self._crop_for_inpainting(
                    image_array, mask_array, hd_strategy_crop_margin
                )
            elif hd_strategy == 'RESIZE' and max(image_array.shape[:2]) > hd_strategy_resize_limit:
                # 缩放策略
                scale = hd_strategy_resize_limit / max(image_array.shape[:2])
                new_h, new_w = int(image_array.shape[0] * scale), int(image_array.shape[1] * scale)
                image_array = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask_array = cv2.resize(mask_array, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # 准备输入
            img = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float() / 255.0
            
            # 移动到设备
            img = move_to_device(img, self.device)
            mask = move_to_device(mask, self.device)
            
            # 填充到模型要求的尺寸
            img = pad_img_to_modulo(img, mod=8)
            mask = pad_img_to_modulo(mask, mod=8)
            
            # 模型推理
            with torch.no_grad():
                inpainted = self.model(img, mask)
                inpainted = torch.clamp(inpainted, 0, 1)
            
            # 后处理 - LaMA输出BGR，转换为RGB
            inpainted = inpainted.cpu().permute(0, 2, 3, 1).numpy()[0]
            inpainted = (inpainted * 255).astype(np.uint8)
            inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)  # BGR to RGB
            
            # 恢复原始尺寸
            if inpainted.shape[:2] != image_array.shape[:2]:
                inpainted = cv2.resize(inpainted, (image_array.shape[1], image_array.shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            
            return inpainted
            
        except Exception as e:
            logger.error(f"LaMA prediction failed: {e}")
            raise
    
    def _crop_for_inpainting(self, image: np.ndarray, mask: np.ndarray, margin: int) -> tuple:
        """为inpainting裁剪图像"""
        import cv2
        
        # 找到mask的边界框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, mask
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contours[0])
        for contour in contours[1:]:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            x = min(x, x1)
            y = min(y, y1)
            w = max(w, x1 + w1 - x)
            h = max(h, y1 + h1 - y)
        
        # 添加边距
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # 裁剪
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
        return cropped_image, cropped_mask
    
    def cleanup_resources(self):
        """清理资源"""
        try:
            if self.model is not None:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
            self.model = None
            self.model_loaded = False
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("✅ LaMA processor resources cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error during LaMA processor cleanup: {e}")
    
    def is_available(self) -> bool:
        """检查LaMA是否可用"""
        return self.available and self.model_loaded
'''
    
    with open(lama_fix, 'w') as f:
        f.write(lama_content)
    
    logger.info(f"✅ 创建修复版LaMA处理器: {lama_fix}")

def fix_color_space_processing():
    """修复颜色空间处理问题"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 修复颜色空间处理...")
    
    # 创建颜色空间处理工具
    color_utils = project_root / "core" / "utils" / "color_utils.py"
    color_content = '''"""
颜色空间处理工具
解决不同模型的颜色格式差异
"""

import cv2
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ColorSpaceProcessor:
    """颜色空间处理器"""
    
    @staticmethod
    def prepare_image_for_model(image: np.ndarray, model_name: str) -> np.ndarray:
        """
        根据模型类型准备图像
        
        Args:
            image: RGB格式的输入图像
            model_name: 模型名称 ('lama', 'zits', 'mat', 'fcf')
            
        Returns:
            处理后的图像数组
        """
        if model_name.lower() == 'lama':
            # LaMA需要BGR输入
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # 其他模型使用RGB
            return image
    
    @staticmethod
    def process_output_for_display(result: np.ndarray, model_name: str) -> np.ndarray:
        """
        处理模型输出用于显示
        
        Args:
            result: 模型输出结果
            model_name: 模型名称
            
        Returns:
            RGB格式的图像数组
        """
        if model_name.lower() == 'lama':
            # LaMA输出BGR，转换为RGB显示
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        else:
            # 其他模型输出RGB
            return result
    
    @staticmethod
    def validate_color_format(image: np.ndarray, expected_format: str = 'RGB') -> bool:
        """
        验证图像颜色格式
        
        Args:
            image: 图像数组
            expected_format: 期望的格式 ('RGB' 或 'BGR')
            
        Returns:
            格式是否正确
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return False
            
        # 简单的颜色分布检查
        if expected_format == 'RGB':
            # RGB图像通常R通道值较高
            return np.mean(image[:, :, 0]) > np.mean(image[:, :, 2])
        else:
            # BGR图像通常B通道值较高
            return np.mean(image[:, :, 2]) > np.mean(image[:, :, 0])
    
    @staticmethod
    def fix_color_channels(image: np.ndarray, current_format: str, target_format: str) -> np.ndarray:
        """
        修复颜色通道
        
        Args:
            image: 输入图像
            current_format: 当前格式
            target_format: 目标格式
            
        Returns:
            修复后的图像
        """
        if current_format == target_format:
            return image
            
        if current_format == 'RGB' and target_format == 'BGR':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif current_format == 'BGR' and target_format == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            logger.warning(f"Unsupported color conversion: {current_format} -> {target_format}")
            return image

class ModelColorConfig:
    """模型颜色配置"""
    
    # 模型颜色格式配置
    MODEL_COLOR_CONFIGS = {
        'lama': {
            'input_format': 'BGR',
            'output_format': 'BGR',
            'display_format': 'RGB'
        },
        'zits': {
            'input_format': 'RGB',
            'output_format': 'RGB',
            'display_format': 'RGB'
        },
        'mat': {
            'input_format': 'RGB',
            'output_format': 'RGB',
            'display_format': 'RGB'
        },
        'fcf': {
            'input_format': 'RGB',
            'output_format': 'RGB',
            'display_format': 'RGB'
        }
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, str]:
        """获取模型颜色配置"""
        return cls.MODEL_COLOR_CONFIGS.get(model_name.lower(), {
            'input_format': 'RGB',
            'output_format': 'RGB',
            'display_format': 'RGB'
        })
    
    @classmethod
    def prepare_input(cls, image: np.ndarray, model_name: str) -> np.ndarray:
        """准备模型输入"""
        config = cls.get_model_config(model_name)
        return ColorSpaceProcessor.prepare_image_for_model(image, model_name)
    
    @classmethod
    def process_output(cls, result: np.ndarray, model_name: str) -> np.ndarray:
        """处理模型输出"""
        config = cls.get_model_config(model_name)
        return ColorSpaceProcessor.process_output_for_display(result, model_name)
'''
    
    with open(color_utils, 'w') as f:
        f.write(color_content)
    
    logger.info(f"✅ 创建颜色空间处理工具: {color_utils}")

def create_installation_guide():
    """创建安装指南"""
    logger = logging.getLogger(__name__)
    logger.info("📝 创建安装指南...")
    
    guide = project_root / "docs" / "INSTALLATION_GUIDE.md"
    guide_content = '''# WatermarkRemover-AI 安装指南

## 快速安装

### 1. 基础环境
```bash
# 克隆项目
git clone <repository_url>
cd WatermarkRemover-AI

# 创建conda环境
conda create -n py310aiwatermark python=3.10
conda activate py310aiwatermark

# 安装基础依赖
pip install -r requirements.txt
```

### 2. 模型文件
项目已包含预训练模型文件，无需额外下载。

### 3. 可选：LaMA支持
如果需要使用LaMA模型（快速处理），安装额外依赖：
```bash
# 方法1: pip安装
pip install saicinpainting

# 方法2: conda安装
conda install -c conda-forge saicinpainting
```

## 启动应用

### 使用优化脚本（推荐）
```bash
# 使用CUDA内存优化脚本
./scripts/set_cuda_env.sh
```

### 手动启动
```bash
# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# 启动应用
streamlit run interfaces/web/main.py --server.port 8501
```

## 故障排除

### CUDA内存不足
1. 降低图像分辨率
2. 使用更小的模型（ZITS → FCF → LaMA）
3. 重启应用清理显存

### LaMA模型不可用
- 检查saicinpainting是否正确安装
- 使用其他可用模型（ZITS/MAT/FCF）

### 颜色异常
- 已修复颜色空间问题
- 如仍有问题，请报告issue

## 性能优化

### 显存管理
- 使用提供的CUDA环境脚本
- 避免同时加载多个大模型
- 定期重启应用清理显存

### 处理速度
- LaMA: 最快（2-5秒）
- FCF: 快速（8-20秒）
- ZITS: 中等（10-30秒）
- MAT: 最慢但质量最好（15-45秒）

## 支持

如遇问题，请：
1. 查看日志输出
2. 检查显存使用情况
3. 提交issue并附上错误信息
'''
    
    with open(guide, 'w') as f:
        f.write(guide_content)
    
    logger.info(f"✅ 创建安装指南: {guide}")

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("🚀 开始应用快速修复...")
    
    try:
        # 1. 修复CUDA内存管理
        fix_cuda_memory_management()
        
        # 2. 修复LaMA处理器
        fix_lama_processor()
        
        # 3. 修复颜色空间处理
        fix_color_space_processing()
        
        # 4. 创建安装指南
        create_installation_guide()
        
        logger.info("✅ 所有修复完成！")
        logger.info("📋 修复内容:")
        logger.info("  - CUDA内存管理优化")
        logger.info("  - LaMA可选支持")
        logger.info("  - 颜色空间处理修复")
        logger.info("  - 安装指南更新")
        logger.info("")
        logger.info("🚀 使用以下命令启动优化版本:")
        logger.info("  ./scripts/set_cuda_env.sh")
        
    except Exception as e:
        logger.error(f"❌ 修复过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 