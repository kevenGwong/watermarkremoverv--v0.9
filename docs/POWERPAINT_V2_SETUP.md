# PowerPaint v2 + Realistic Vision V1.4-inpainting 安装部署指南

## 🎯 **项目目标**

在 PowerPaint v2 框架下，使用轻量版模型 Realistic Vision V1.4-inpainting 实现自定义 mask 的商品图去水印功能，适配本地 GPU 推理。

### ✅ **核心目标**
- 使用自定义 mask，精准修复水印区域
- 替换 PowerPaint 默认重型模型 → 使用 HuggingFace 上的轻量高质量模型
- 保留 PowerPaint v2 的 CLI / UI / 推理兼容结构
- 可在本地 GPU（如 Tesla T4, RTX 3060, 4060）上运行
- 支持未来切换其他 SD1.5 inpainting 模型

## 🔧 **系统要求**

### 硬件要求
- **GPU**: NVIDIA GPU with 8GB+ VRAM (推荐)
  - Tesla T4 (8GB) ✅
  - RTX 3060 (12GB) ✅
  - RTX 4060 (8GB) ✅
  - RTX 4090 (24GB) ✅
- **内存**: 16GB+ RAM
- **存储**: 10GB+ 可用空间

### 软件要求
- **Python**: 3.8-3.11
- **CUDA**: 11.8+ (推荐 12.1+)
- **PyTorch**: 2.0+ with CUDA support

## 📦 **安装步骤**

### 1. 环境准备

```bash
# 激活conda环境
conda activate py310aiwatermark

# 检查CUDA版本
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### 2. 安装PowerPaint v2依赖

```bash
# 安装基础依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors

# 安装PowerPaint v2
pip install git+https://github.com/open-mmlab/PowerPaint.git

# 安装IOPaint (如果未安装)
pip install iopaint>=1.6.0
```

### 3. 下载Realistic Vision V1.4-inpainting模型

```bash
# 创建模型目录
mkdir -p models/powerpaint_v2
cd models/powerpaint_v2

# 下载模型 (使用git-lfs)
git lfs install
git clone https://huggingface.co/Sanster/Realistic_Vision_V1.4-inpainting
### 若需要huggingface token , 则填入REMOVED_TOKEN
# 或者使用huggingface_hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Sanster/Realistic_Vision_V1.4-inpainting',
    local_dir='./Realistic_Vision_V1.4-inpainting',
    local_dir_use_symlinks=False
)
"
```

### 4. 验证安装

```bash
# 测试模型加载
python -c "
import torch
from diffusers import StableDiffusionInpaintPipeline

# 加载模型
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    './models/powerpaint_v2/Realistic_Vision_V1.4-inpainting',
    torch_dtype=torch.float16,
    use_safetensors=True
)

# 移动到GPU
pipe = pipe.to('cuda')
print('✅ Realistic Vision V1.4-inpainting 加载成功!')
print(f'模型设备: {pipe.device}')
print(f'模型精度: {pipe.unet.dtype}')
"
```

## 🏗️ **项目集成**

### 1. 创建PowerPaint Inpainter模块

```python
# watermark_remover_ai/core/models/powerpaint_inpainter.py
"""
PowerPaint v2 + Realistic Vision V1.4-inpainting 图像修复模型
"""

from typing import Any, Dict, Optional, Union
from PIL import Image
import numpy as np
import cv2
import logging
import torch
from pathlib import Path

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class PowerPaintInpainter(BaseModel):
    """PowerPaint v2 图像修复模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化PowerPaint修复器
        
        Args:
            config: 包含powerpaint配置的字典
        """
        super().__init__(config)
        self.pipe = None
        self._load_model()
    
    def _load_model(self) -> None:
        """加载PowerPaint模型"""
        try:
            from diffusers import StableDiffusionInpaintPipeline
            
            # 获取模型路径
            model_path = self.config.get('models', {}).get('powerpaint_model_path', 
                './models/powerpaint_v2/Realistic_Vision_V1.4-inpainting')
            
            # 检查模型路径
            if not Path(model_path).exists():
                raise FileNotFoundError(f"PowerPaint模型路径不存在: {model_path}")
            
            # 加载配置
            powerpaint_config = self.config.get('powerpaint_config', {})
            use_fp16 = powerpaint_config.get('use_fp16', True)
            enable_attention_slicing = powerpaint_config.get('enable_attention_slicing', True)
            
            # 加载模型
            logger.info(f"正在加载PowerPaint模型: {model_path}")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
                use_safetensors=True
            )
            
            # 移动到设备
            self.pipe = self.pipe.to(self.device)
            
            # 启用内存优化
            if enable_attention_slicing:
                self.pipe.enable_attention_slicing()
            
            logger.info(f"✅ PowerPaint模型加载成功: {model_path}")
            logger.info(f"   设备: {self.device}")
            logger.info(f"   精度: {self.pipe.unet.dtype}")
            
        except Exception as e:
            logger.error(f"PowerPaint模型加载失败: {e}")
            raise
    
    def predict(self, 
                image: Union[Image.Image, np.ndarray], 
                mask: Union[Image.Image, np.ndarray],
                custom_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        执行图像修复
        
        Args:
            image: 输入图像 (PIL Image)
            mask: 掩码图像 (PIL Image)
            custom_config: 自定义配置
            
        Returns:
            修复后的图像数组 (RGB格式)
        """
        try:
            # 默认配置
            default_config = {
                'num_inference_steps': 50,
                'guidance_scale': 7.5,
                'negative_prompt': 'watermark, logo, text, signature, blurry, low quality',
                'prompt': 'high quality, detailed, clean',
                'seed': -1
            }
            
            # 合并自定义配置
            if custom_config:
                default_config.update(custom_config)
                logger.info(f"📋 使用自定义配置: {custom_config}")
            
            # 设置随机种子
            if default_config['seed'] >= 0:
                torch.manual_seed(default_config['seed'])
                logger.info(f"🎲 使用固定种子: {default_config['seed']}")
            
            # 确保输入格式正确
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)
            
            # 确保mask是二值的
            mask_np = np.array(mask.convert('L'))
            mask_np = (mask_np > 128).astype(np.uint8) * 255
            mask = Image.fromarray(mask_np)
            
            logger.info(f"🖼️ 输入图像: {image.size}, 模式: {image.mode}")
            logger.info(f"🎭 输入mask: {mask.size}, 覆盖率: {np.sum(mask_np > 0) / mask_np.size * 100:.2f}%")
            
            # PowerPaint推理
            logger.info("🤖 PowerPaint模型推理...")
            result = self.pipe(
                prompt=default_config['prompt'],
                negative_prompt=default_config['negative_prompt'],
                image=image,
                mask_image=mask,
                num_inference_steps=default_config['num_inference_steps'],
                guidance_scale=default_config['guidance_scale'],
                generator=torch.Generator(device=self.device).manual_seed(default_config['seed']) if default_config['seed'] >= 0 else None
            ).images[0]
            
            logger.info("✅ PowerPaint推理完成!")
            
            # 转换为numpy数组
            result_np = np.array(result)
            logger.info(f"   输出图像: shape={result_np.shape}, dtype={result_np.dtype}")
            
            return result_np
            
        except Exception as e:
            logger.error(f"PowerPaint修复失败: {e}")
            raise
    
    def inpaint_image(self, 
                     image: Union[Image.Image, np.ndarray], 
                     mask: Union[Image.Image, np.ndarray],
                     custom_config: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        修复图像并返回PIL图像
        
        Args:
            image: 输入图像
            mask: 掩码图像
            custom_config: 自定义配置
            
        Returns:
            修复后的PIL图像
        """
        result_np = self.predict(image, mask, custom_config)
        return Image.fromarray(result_np)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            "model_type": "PowerPaint_v2_RealisticVision",
            "base_model": "Realistic_Vision_V1.4-inpainting",
            "framework": "diffusers",
            "supports_fp16": True,
            "supports_attention_slicing": True
        })
        return info
```

### 2. 更新配置文件

```yaml
# data/configs/web_config_powerpaint.yaml
# PowerPaint v2 专用配置

app:
  title: "AI Watermark Remover (PowerPaint v2)"
  host: "0.0.0.0"
  port: 8501
  debug: false

# 处理选项
processing:
  default_max_bbox_percent: 10.0
  default_transparent: false
  default_overwrite: false
  default_force_format: "PNG"
  supported_formats: ["jpg", "jpeg", "png", "webp"]

# Mask生成器配置
mask_generator:
  model_type: "custom"  # "custom" or "florence"
  mask_model_path: "./models/epoch=071-valid_iou=0.7267.ckpt"
  image_size: 768
  imagenet_mean: [0.485, 0.456, 0.406]
  imagenet_std: [0.229, 0.224, 0.225]
  mask_threshold: 0.5
  mask_dilate_kernel_size: 3

# 模型配置
models:
  florence_model: "microsoft/Florence-2-large"
  lama_model: "lama"  # 保留LaMA作为备选
  powerpaint_model_path: "./models/powerpaint_v2/Realistic_Vision_V1.4-inpainting"
  default_inpainting: "powerpaint"  # 默认使用PowerPaint

# PowerPaint专用配置
powerpaint_config:
  # 基础参数
  num_inference_steps: 50  # 10-200, 推理步数
  guidance_scale: 7.5  # 1.0-20.0, 引导尺度
  negative_prompt: "watermark, logo, text, signature, blurry, low quality, artifacts"
  prompt: "high quality, detailed, clean, professional"
  
  # 性能优化
  use_fp16: true  # 使用半精度
  enable_attention_slicing: true  # 启用注意力切片
  enable_memory_efficient_attention: true  # 内存高效注意力
  
  # 高级参数
  seed: -1  # -1为随机, >=0为固定种子
  eta: 0.0  # 0.0-1.0, 噪声调度器参数
  
  # 图像预处理
  resize_to_512: true  # 是否调整到512x512
  crop_strategy: "center"  # center, random, none
  
  # 后处理
  enable_face_restoration: false  # 是否启用面部修复
  enable_color_correction: true  # 是否启用颜色校正

# 文件处理
files:
  max_upload_size: 20  # MB
  temp_dir: "./temp"
  output_dir: "./output"
  allowed_extensions: [".jpg", ".jpeg", ".png", ".webp"]

# UI设置
ui:
  show_advanced_options: true
  show_system_info: true
  enable_batch_processing: true
  default_download_format: "png"
  
  # PowerPaint专用UI选项
  show_powerpaint_params: true
  show_prompt_editor: true
  show_seed_control: true
```

### 3. 更新推理管理器

```python
# inference.py 中添加PowerPaint支持
class EnhancedWatermarkProcessor:
    def __init__(self, base_processor):
        self.base_processor = base_processor
        self.powerpaint_inpainter = None
        self._load_powerpaint_model()
    
    def _load_powerpaint_model(self):
        """加载PowerPaint模型"""
        try:
            from watermark_remover_ai.core.models.powerpaint_inpainter import PowerPaintInpainter
            self.powerpaint_inpainter = PowerPaintInpainter(self.base_processor.config)
            logger.info("PowerPaint模型加载成功")
        except Exception as e:
            logger.warning(f"PowerPaint模型加载失败，将使用LaMA: {e}")
    
    def process_image_with_powerpaint(self, image, mask, inpaint_params):
        """使用PowerPaint处理图像"""
        if self.powerpaint_inpainter is None:
            logger.warning("PowerPaint模型未加载，回退到LaMA")
            return self.base_processor._process_with_lama(image, mask, inpaint_params)
        
        # 转换PowerPaint参数
        powerpaint_config = {
            'num_inference_steps': inpaint_params.get('num_inference_steps', 50),
            'guidance_scale': inpaint_params.get('guidance_scale', 7.5),
            'negative_prompt': inpaint_params.get('negative_prompt', 'watermark, logo, text'),
            'prompt': inpaint_params.get('prompt', 'high quality, clean'),
            'seed': inpaint_params.get('seed', -1)
        }
        
        return self.powerpaint_inpainter.inpaint_image(image, mask, powerpaint_config)
```

### 4. 更新UI界面

```python
# ui.py 中添加PowerPaint参数控制
class ParameterPanel:
    def _render_powerpaint_section(self) -> Dict[str, Any]:
        """渲染PowerPaint参数面板"""
        st.sidebar.subheader("🎨 PowerPaint v2 参数")
        
        # 基础参数
        num_inference_steps = st.slider(
            "推理步数", 10, 200, 50, 5,
            help="更高的步数 = 更好的质量，但更慢"
        )
        
        guidance_scale = st.slider(
            "引导尺度", 1.0, 20.0, 7.5, 0.5,
            help="控制生成图像的创造性 vs 准确性"
        )
        
        # 提示词编辑
        st.sidebar.subheader("📝 提示词")
        prompt = st.text_area(
            "正向提示词", 
            "high quality, detailed, clean, professional",
            help="描述期望的图像质量"
        )
        
        negative_prompt = st.text_area(
            "负向提示词",
            "watermark, logo, text, signature, blurry, low quality, artifacts",
            help="避免的特征"
        )
        
        # 随机种子
        seed = st.number_input(
            "随机种子", -1, 999999, -1,
            help="-1为随机，>=0为固定种子"
        )
        
        # 性能选项
        st.sidebar.subheader("⚡ 性能选项")
        use_fp16 = st.checkbox("使用FP16", True, help="减少显存占用，提升速度")
        enable_attention_slicing = st.checkbox("注意力切片", True, help="减少显存占用")
        
        return {
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'seed': seed,
            'use_fp16': use_fp16,
            'enable_attention_slicing': enable_attention_slicing
        }
```

## 🧪 **测试验证**

### 1. 基础功能测试

```bash
# 创建测试脚本
cat > test_powerpaint.py << 'EOF'
#!/usr/bin/env python3
"""
PowerPaint v2 功能测试
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from watermark_remover_ai.core.models.powerpaint_inpainter import PowerPaintInpainter

def test_powerpaint_loading():
    """测试模型加载"""
    print("🧪 测试PowerPaint模型加载...")
    
    config = {
        'models': {
            'powerpaint_model_path': './models/powerpaint_v2/Realistic_Vision_V1.4-inpainting'
        },
        'powerpaint_config': {
            'use_fp16': True,
            'enable_attention_slicing': True
        }
    }
    
    try:
        inpainter = PowerPaintInpainter(config)
        print("✅ PowerPaint模型加载成功!")
        print(f"   模型信息: {inpainter.get_model_info()}")
        return inpainter
    except Exception as e:
        print(f"❌ PowerPaint模型加载失败: {e}")
        return None

def test_powerpaint_inference(inpainter):
    """测试推理功能"""
    print("\n🧪 测试PowerPaint推理...")
    
    # 创建测试图像和mask
    test_image = Image.new('RGB', (512, 512), color='white')
    test_mask = Image.new('L', (512, 512), color=0)
    
    # 在中心区域创建mask
    mask_array = np.array(test_mask)
    mask_array[200:300, 200:300] = 255
    test_mask = Image.fromarray(mask_array)
    
    # 测试推理
    try:
        result = inpainter.inpaint_image(
            test_image, 
            test_mask,
            {
                'num_inference_steps': 20,  # 快速测试
                'guidance_scale': 7.5,
                'prompt': 'high quality, clean',
                'negative_prompt': 'watermark, text',
                'seed': 42
            }
        )
        
        print("✅ PowerPaint推理成功!")
        print(f"   输出图像尺寸: {result.size}")
        
        # 保存结果
        result.save('test_powerpaint_result.png')
        print("   结果已保存到: test_powerpaint_result.png")
        
        return True
    except Exception as e:
        print(f"❌ PowerPaint推理失败: {e}")
        return False

if __name__ == "__main__":
    # 测试模型加载
    inpainter = test_powerpaint_loading()
    
    if inpainter:
        # 测试推理
        test_powerpaint_inference(inpainter)
    
    print("\n🎉 PowerPaint v2 测试完成!")
EOF

# 运行测试
python test_powerpaint.py
```

### 2. 性能基准测试

```bash
# 创建性能测试脚本
cat > benchmark_powerpaint.py << 'EOF'
#!/usr/bin/env python3
"""
PowerPaint v2 性能基准测试
"""

import time
import torch
import psutil
from PIL import Image
import numpy as np
from watermark_remover_ai.core.models.powerpaint_inpainter import PowerPaintInpainter

def benchmark_powerpaint():
    """性能基准测试"""
    print("🚀 PowerPaint v2 性能基准测试")
    print("="*50)
    
    # 配置
    config = {
        'models': {
            'powerpaint_model_path': './models/powerpaint_v2/Realistic_Vision_V1.4-inpainting'
        },
        'powerpaint_config': {
            'use_fp16': True,
            'enable_attention_slicing': True
        }
    }
    
    # 加载模型
    print("📦 加载PowerPaint模型...")
    start_time = time.time()
    inpainter = PowerPaintInpainter(config)
    load_time = time.time() - start_time
    print(f"   加载时间: {load_time:.2f}秒")
    
    # 创建测试图像
    test_image = Image.new('RGB', (512, 512), color='white')
    test_mask = Image.new('L', (512, 512), color=0)
    mask_array = np.array(test_mask)
    mask_array[200:300, 200:300] = 255
    test_mask = Image.fromarray(mask_array)
    
    # 测试不同参数的性能
    test_configs = [
        {'num_inference_steps': 20, 'name': '快速模式'},
        {'num_inference_steps': 50, 'name': '标准模式'},
        {'num_inference_steps': 100, 'name': '高质量模式'}
    ]
    
    for test_config in test_configs:
        print(f"\n🔍 测试 {test_config['name']}...")
        
        # 记录开始状态
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        start_time = time.time()
        
        # 执行推理
        result = inpainter.inpaint_image(
            test_image,
            test_mask,
            {
                'num_inference_steps': test_config['num_inference_steps'],
                'guidance_scale': 7.5,
                'prompt': 'high quality',
                'negative_prompt': 'watermark',
                'seed': 42
            }
        )
        
        # 记录结束状态
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 计算性能指标
        inference_time = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024**3  # GB
        
        print(f"   推理时间: {inference_time:.2f}秒")
        print(f"   显存使用: {memory_used:.2f}GB")
        print(f"   处理速度: {test_config['num_inference_steps']/inference_time:.1f}步/秒")
    
    print("\n✅ 性能基准测试完成!")

if __name__ == "__main__":
    benchmark_powerpaint()
EOF

# 运行性能测试
python benchmark_powerpaint.py
```

## 🚀 **启动应用**

### 1. 使用PowerPaint配置启动

```bash
# 使用PowerPaint配置启动Web UI
python app.py web --config data/configs/web_config_powerpaint.yaml

# 或者修改默认配置
cp data/configs/web_config_powerpaint.yaml data/configs/web_config.yaml
python app.py web
```

### 2. 验证功能

1. **访问Web界面**: http://localhost:8501
2. **上传测试图像**
3. **选择PowerPaint v2作为inpainting模型**
4. **调整参数并处理**
5. **检查结果质量**

## 🔧 **故障排除**

### 常见问题

#### 1. 模型加载失败
```bash
# 检查模型路径
ls -la models/powerpaint_v2/Realistic_Vision_V1.4-inpainting/

# 重新下载模型
rm -rf models/powerpaint_v2/Realistic_Vision_V1.4-inpainting/
git clone https://huggingface.co/Sanster/Realistic_Vision_V1.4-inpainting models/powerpaint_v2/Realistic_Vision_V1.4-inpainting
```

#### 2. 显存不足
```python
# 在配置中启用更多优化
powerpaint_config:
  use_fp16: true
  enable_attention_slicing: true
  enable_memory_efficient_attention: true
  enable_vae_slicing: true
```

#### 3. 推理速度慢
```python
# 调整参数
powerpaint_config:
  num_inference_steps: 30  # 减少步数
  enable_attention_slicing: false  # 关闭注意力切片
```

## 📊 **性能优化建议**

### 1. 显存优化
- 启用FP16: `use_fp16: true`
- 启用注意力切片: `enable_attention_slicing: true`
- 启用VAE切片: `enable_vae_slicing: true`

### 2. 速度优化
- 减少推理步数: `num_inference_steps: 30-50`
- 关闭不必要的优化选项
- 使用较小的输入图像尺寸

### 3. 质量优化
- 增加推理步数: `num_inference_steps: 50-100`
- 调整引导尺度: `guidance_scale: 7.5-10.0`
- 优化提示词

## 🎉 **总结**

通过以上步骤，您已成功：

1. ✅ 安装了PowerPaint v2框架
2. ✅ 集成了Realistic Vision V1.4-inpainting模型
3. ✅ 适配了项目架构和UI设计
4. ✅ 添加了完整的调试参数控制
5. ✅ 实现了本地GPU推理支持

现在您可以使用PowerPaint v2 + Realistic Vision进行高质量的水印去除任务，同时保持与现有系统的完全兼容性！

## 🔗 **参考链接**

- [PowerPaint v2 官方文档](https://www.iopaint.com/models/diffusion/powerpaint_v2)
- [Realistic Vision V1.4-inpainting 模型](https://huggingface.co/Sanster/Realistic_Vision_V1.4-inpainting)
- [PowerPaint GitHub 仓库](https://github.com/open-mmlab/PowerPaint) 