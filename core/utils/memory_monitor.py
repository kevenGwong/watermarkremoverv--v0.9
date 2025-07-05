"""
GPU内存监控工具
"""

import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """GPU内存监控器 - 增强版本支持模型切换监控"""
    
    def __init__(self):
        self.memory_history = []
        self.warning_threshold = 0.8  # 80%内存使用率警告
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取完整内存信息"""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'gpu_info': self.get_gpu_memory_info(),
            'cpu_info': self.get_cpu_memory_info()
        }
        
        # 记录历史
        if info['gpu_info']:
            self.memory_history.append(info['gpu_info']['usage_percent'])
            if len(self.memory_history) > 10:  # 保留最近10次记录
                self.memory_history.pop(0)
        
        return info
    
    def get_gpu_memory_info(self) -> Optional[Dict[str, float]]:
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
                'usage_percent': (reserved / total) * 100,
                'device_name': torch.cuda.get_device_name(0)
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return None
    
    def get_cpu_memory_info(self) -> Dict[str, float]:
        """获取CPU内存信息"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                'total_gb': memory.total / 1024**3,
                'used_gb': memory.used / 1024**3,
                'free_gb': memory.available / 1024**3,
                'usage_percent': memory.percent
            }
        except ImportError:
            logger.warning("psutil not available, CPU memory info unavailable")
            return {}
    
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
