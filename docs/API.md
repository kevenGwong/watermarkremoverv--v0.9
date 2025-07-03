# 🔌 API 文档

## 📋 概述

本文档描述了 AI Watermark Remover v2.0 的各个模块的API接口，包括核心模型、处理器、工具函数等。

## 🏗️ 核心模块 API

### 1. 模型层 (core/models/)

#### BaseModel 基类

所有AI模型的抽象基类，定义了统一的接口。

```python
from core.models.base_model import BaseModel

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        """初始化模型"""
        pass
    
    @abstractmethod
    def _load_model(self) -> None:
        """加载模型的具体实现"""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """模型推理的具体实现"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass
```

#### FlorenceDetector

Florence-2检测模型的实现。

```python
from core.models.florence_detector import FlorenceDetector

# 创建检测器
config = {
    'models': {
        'florence_model': 'microsoft/Florence-2-large'
    }
}
detector = FlorenceDetector(config)

# 检测水印
result = detector.predict(
    image=image,
    text_input="watermark",
    max_bbox_percent=10.0
)

# 生成掩码
mask = detector.generate_mask(
    image=image,
    max_bbox_percent=10.0,
    detection_prompt="watermark"
)
```

**参数说明**:
- `image`: PIL.Image - 输入图像
- `text_input`: str - 检测提示词
- `max_bbox_percent`: float - 最大边界框百分比

**返回值**:
- `result`: Dict - 检测结果字典
- `mask`: PIL.Image - 生成的掩码图像

#### CustomSegmenter

自定义分割模型的实现。

```python
from core.models.custom_segmenter import CustomSegmenter

# 创建分割器
config = {
    'mask_generator': {
        'mask_model_path': './models/epoch=071-valid_iou=0.7267.ckpt',
        'image_size': 768,
        'mask_threshold': 0.5,
        'mask_dilate_kernel_size': 3
    }
}
segmenter = CustomSegmenter(config)

# 生成掩码
mask = segmenter.generate_mask(image)
```

**参数说明**:
- `image`: PIL.Image - 输入图像

**返回值**:
- `mask`: PIL.Image - 生成的掩码图像

#### LamaInpainter

LaMA修复模型的实现。

```python
from core.models.lama_inpainter import LamaInpainter

# 创建修复器
config = {
    'models': {
        'lama_model': 'lama'
    }
}
inpainter = LamaInpainter(config)

# 修复图像
result = inpainter.inpaint_image(
    image=image,
    mask=mask,
    custom_config={
        'ldm_steps': 50,
        'ldm_sampler': 'ddim',
        'hd_strategy': 'CROP'
    }
)
```

**参数说明**:
- `image`: Union[PIL.Image, np.ndarray, bytes] - 输入图像
- `mask`: Union[PIL.Image, np.ndarray] - 掩码图像
- `custom_config`: Dict - 自定义配置

**返回值**:
- `result`: PIL.Image - 修复后的图像

### 2. 处理器层 (core/processors/)

#### WatermarkProcessor

主处理器，协调各个组件完成水印去除任务。

```python
from core.processors.watermark_processor import WatermarkProcessor

# 创建处理器
processor = WatermarkProcessor(config)

# 处理图像
result = processor.process_image(
    image=image,
    mask_method="custom",  # "custom", "florence", "upload"
    transparent=False,
    max_bbox_percent=10.0,
    force_format="PNG"
)
```

**参数说明**:
- `image`: Union[PIL.Image, str, bytes] - 输入图像
- `mask_method`: str - 掩码生成方法
- `transparent`: bool - 是否使用透明模式
- `max_bbox_percent`: float - 最大边界框百分比
- `force_format`: str - 强制输出格式

**返回值**:
- `result`: ProcessingResult - 处理结果对象

### 3. 工具层 (core/utils/)

#### ImageUtils

图像处理工具函数。

```python
from core.utils.image_utils import (
    load_image_opencv,
    convert_bgr_to_rgb,
    resize_image,
    add_background,
    make_region_transparent
)

# 加载图像
image = load_image_opencv("input.jpg")

# 转换颜色空间
rgb_image = convert_bgr_to_rgb(bgr_image)

# 调整图像大小
resized_image = resize_image(image, (800, 600), keep_aspect_ratio=True)

# 添加背景
image_with_bg = add_background(rgba_image, bg_type="white")

# 使区域透明
transparent_image = make_region_transparent(image, mask)
```

#### ConfigUtils

配置管理工具函数。

```python
from core.utils.config_utils import (
    load_config,
    save_config,
    merge_configs,
    get_config_value,
    set_config_value
)

# 加载配置
config = load_config("config.yaml")

# 保存配置
save_config(config, "output.yaml")

# 合并配置
merged_config = merge_configs(base_config, override_config)

# 获取配置值
model_path = get_config_value(config, "models.lama_model")

# 设置配置值
set_config_value(config, "processing.max_bbox_percent", 15.0)
```

#### FlorenceUtils

Florence-2专用工具函数。

```python
from core.utils.florence_utils import (
    TaskType,
    identify,
    draw_polygons,
    convert_bbox_to_relative
)

# 使用Florence-2识别
result = identify(
    TaskType.OPEN_VOCAB_DETECTION,
    image,
    "watermark",
    model,
    processor,
    device
)

# 绘制多边形
annotated_image = draw_polygons(image, prediction, fill_mask=False)

# 转换坐标
relative_coords = convert_bbox_to_relative(box, image)
```

## 🌐 Web界面 API

### Streamlit应用

```python
from interfaces.web.frontend.streamlit_app import main

# 启动Web应用
main()
```

### 组件API

#### ParameterPanel

参数控制面板组件。

```python
from interfaces.web.frontend.components.parameter_panel import ParameterPanel

# 创建参数面板
panel = ParameterPanel()

# 渲染参数控制
params = panel.render()
```

#### ImageComparison

图像对比组件。

```python
from interfaces.web.frontend.components.image_comparison import ImageComparison

# 创建对比组件
comparison = ImageComparison()

# 显示对比
comparison.show_comparison(original_image, processed_image)
```

## 💻 命令行界面 API

### CLI应用

```python
from interfaces.cli.cli_app import CLIApp

# 创建CLI应用
app = CLIApp()

# 处理单张图像
app.process_single("input.jpg", "output.jpg", params)

# 批量处理
app.process_batch("input_dir/", "output_dir/", params)
```

## 🖥️ GUI界面 API

### Qt应用

```python
from interfaces.gui.qt_app import QtApp

# 创建GUI应用
app = QtApp()

# 启动应用
app.run()
```

## 🔧 配置API

### 配置文件结构

```yaml
# config/default_config.yaml
app:
  title: "AI Watermark Remover"
  host: "0.0.0.0"
  port: 8501
  debug: false

models:
  florence_model: "microsoft/Florence-2-large"
  lama_model: "lama"

mask_generator:
  model_type: "custom"
  mask_model_path: "./models/epoch=071-valid_iou=0.7267.ckpt"
  image_size: 768
  imagenet_mean: [0.485, 0.456, 0.406]
  imagenet_std: [0.229, 0.224, 0.225]
  mask_threshold: 0.5
  mask_dilate_kernel_size: 3

processing:
  default_max_bbox_percent: 10.0
  default_transparent: false
  default_overwrite: false
  default_force_format: "None"
  supported_formats: ["jpg", "jpeg", "png", "webp"]

files:
  max_upload_size: 10
  temp_dir: "./temp"
  output_dir: "./output"
  allowed_extensions: [".jpg", ".jpeg", ".png", ".webp"]

ui:
  show_advanced_options: true
  show_system_info: true
  enable_batch_processing: true
  default_download_format: "png"
```

### 环境配置

```python
from core.utils.config_utils import get_environment_config

# 获取环境配置
dev_config = get_environment_config("development")
prod_config = get_environment_config("production")
```

## 🧪 测试API

### 单元测试

```python
from tests.unit.test_models import TestFlorenceDetector

# 运行模型测试
test = TestFlorenceDetector()
test.test_model_loading()
test.test_prediction()
```

### 集成测试

```python
from tests.integration.test_processing_pipeline import TestProcessingPipeline

# 运行集成测试
test = TestProcessingPipeline()
test.test_complete_workflow()
```

## 📊 数据模型

### ProcessingResult

处理结果数据模型。

```python
from dataclasses import dataclass
from typing import Optional
from PIL import Image

@dataclass
class ProcessingResult:
    success: bool
    result_image: Optional[Image.Image] = None
    mask_image: Optional[Image.Image] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
```

### ModelConfig

模型配置数据模型。

```python
@dataclass
class ModelConfig:
    model_type: str
    model_path: str
    device: str = "auto"
    batch_size: int = 1
    max_memory: Optional[int] = None
```

## 🔄 使用示例

### 完整处理流程

```python
import yaml
from PIL import Image
from core.processors.watermark_processor import WatermarkProcessor

# 加载配置
with open("web_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 创建处理器
processor = WatermarkProcessor(config)

# 加载图像
image = Image.open("input.jpg")

# 处理图像
result = processor.process_image(
    image=image,
    mask_method="custom",
    transparent=False,
    max_bbox_percent=10.0
)

# 检查结果
if result.success:
    # 保存结果
    result.result_image.save("output.jpg")
    result.mask_image.save("mask.jpg")
    print(f"处理完成，耗时: {result.processing_time:.2f}秒")
else:
    print(f"处理失败: {result.error_message}")
```

### 自定义模型集成

```python
from core.models.base_model import BaseModel, ModelFactory

# 定义新模型
class MyCustomModel(BaseModel):
    def _load_model(self):
        # 实现模型加载逻辑
        pass
    
    def predict(self, image, **kwargs):
        # 实现推理逻辑
        pass

# 注册模型
ModelFactory.register("my_model", MyCustomModel)

# 使用模型
config = {"my_model": {"path": "path/to/model"}}
model = ModelFactory.create("my_model", config)
result = model.predict(image)
```

### 批量处理

```python
import os
from pathlib import Path

def batch_process(input_dir: str, output_dir: str, processor: WatermarkProcessor):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for image_file in input_path.glob("*.jpg"):
        # 处理单张图像
        result = processor.process_image(
            image=str(image_file),
            mask_method="custom"
        )
        
        if result.success:
            # 保存结果
            output_file = output_path / f"processed_{image_file.name}"
            result.result_image.save(output_file)
            print(f"处理完成: {image_file.name}")
        else:
            print(f"处理失败: {image_file.name} - {result.error_message}")

# 使用批量处理
processor = WatermarkProcessor(config)
batch_process("input_images/", "output_images/", processor)
```

## 🚨 错误处理

### 常见异常

```python
from core.models.base_model import ModelLoadError
from core.processors.watermark_processor import ProcessingError

try:
    # 模型操作
    model = FlorenceDetector(config)
except ModelLoadError as e:
    print(f"模型加载失败: {e}")
except ProcessingError as e:
    print(f"处理失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 错误码

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| MODEL_LOAD_ERROR | 模型加载失败 | 检查模型路径和依赖 |
| PROCESSING_ERROR | 处理失败 | 检查输入数据和参数 |
| CONFIG_ERROR | 配置错误 | 验证配置文件格式 |
| MEMORY_ERROR | 内存不足 | 减少批处理大小或图像尺寸 |

## 📈 性能优化

### 内存优化

```python
# 使用生成器处理大文件
def process_large_directory(input_dir: str, output_dir: str):
    for image_file in Path(input_dir).glob("*.jpg"):
        # 逐个处理，避免内存溢出
        yield process_single_image(image_file, output_dir)

# 使用上下文管理器
with torch.no_grad():
    result = model.predict(image)
```

### GPU优化

```python
# 混合精度训练
from torch.cuda.amp import autocast

with autocast():
    result = model.predict(image)

# 多GPU支持
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

**📝 本文档会随着API的更新持续维护，请关注最新版本。** 