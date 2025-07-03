# 🎨 AI Watermark Remover v2.0

> 基于Florence-2 + LaMA的智能水印去除系统，支持多种检测模型和修复技术

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目概述

AI Watermark Remover 是一个功能强大的水印去除工具，采用模块化架构设计，支持多种AI模型和用户界面。项目集成了最新的计算机视觉技术，提供高质量的水印检测和图像修复功能。

### ✨ 核心特性

- 🔍 **多模型检测**: Florence-2开放词汇检测 + 自定义分割模型
- 🎨 **高质量修复**: LaMA + Stable Diffusion 1.5 图像修复
- 🖥️ **多界面支持**: Web UI、命令行、桌面GUI
- ⚙️ **参数控制**: 精细的参数调节和实时预览
- 📦 **批量处理**: 支持文件夹批量处理
- 🎯 **透明模式**: 支持透明背景输出

## 🏗️ 系统架构

### 模块化设计

```
watermark_remover_ai/
├── 📁 core/                          # 核心业务逻辑
│   ├── 📁 models/                    # AI模型实现
│   │   ├── base_model.py            # 模型基类
│   │   ├── florence_detector.py     # Florence-2检测模型
│   │   ├── custom_segmenter.py      # 自定义分割模型
│   │   └── lama_inpainter.py        # LaMA修复模型
│   ├── 📁 processors/               # 处理管道
│   │   └── watermark_processor.py   # 主处理器
│   └── 📁 utils/                    # 核心工具
│       ├── image_utils.py           # 图像处理工具
│       ├── mask_utils.py            # 掩码处理工具
│       ├── config_utils.py          # 配置管理
│       └── florence_utils.py        # Florence-2专用工具
├── 📁 interfaces/                    # 用户界面
│   ├── 📁 web/                      # Web界面
│   │   ├── 📁 frontend/             # 前端组件
│   │   ├── 📁 backend/              # 后端API
│   │   └── 📁 services/             # 业务服务
│   ├── 📁 cli/                      # 命令行界面
│   └── 📁 gui/                      # 桌面GUI
├── 📁 config/                       # 配置管理
├── 📁 data/                         # 数据处理
├── 📁 tests/                        # 测试模块
├── 📁 scripts/                      # 脚本工具
├── 📁 docs/                         # 文档
└── 📁 archive/                      # 归档文件
```

### 技术栈

- **AI框架**: PyTorch, Transformers, iopaint
- **图像处理**: OpenCV, PIL, Albumentations
- **Web框架**: Streamlit, Streamlit-image-comparison
- **GUI框架**: PyQt6
- **配置管理**: PyYAML
- **测试框架**: pytest

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **内存**: 4GB+ RAM
- **GPU**: CUDA兼容GPU (推荐)
- **存储**: 2GB+ 可用空间

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/kevenGwong/watermarkremoverv--v0.9.git
cd WatermarkRemover-AI
```

2. **创建虚拟环境**
```bash
# 使用conda (推荐)
conda env create -f environment.yml
conda activate py310aiwatermark

# 或使用pip
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **下载模型**
```bash
# 下载LaMA模型
iopaint download --model lama

# 下载Florence-2模型 (自动下载)
python -c "from transformers import AutoProcessor, AutoModelForCausalLM; AutoProcessor.from_pretrained('microsoft/Florence-2-large')"
```

### 基本使用

#### 🌐 Web界面 (推荐)
```bash
python app.py web --port 8501
```
访问: http://localhost:8501

#### 💻 命令行
```bash
# 单张图片处理
python app.py cli input.jpg output.jpg

# 批量处理
python app.py cli input_dir/ output_dir/ --batch

# 使用自定义参数
python app.py cli input.jpg output.jpg --mask-method custom --transparent
```

#### 🖥️ 桌面GUI
```bash
python app.py gui
```

## 🎯 功能详解

### 检测模型

#### 1. Florence-2检测
- **特点**: 开放词汇检测，支持自然语言描述
- **适用**: 复杂水印、文字水印、Logo检测
- **参数**: 检测提示词、置信度阈值、最大边界框比例

#### 2. 自定义分割模型
- **特点**: 基于FPN + MIT-B5的专用水印分割
- **适用**: 精确的水印区域分割
- **参数**: 分割阈值、膨胀核大小、膨胀迭代次数

#### 3. 手动上传
- **特点**: 支持用户自定义掩码
- **适用**: 特殊水印、复杂场景
- **功能**: 掩码编辑、膨胀处理

### 修复模型

#### 1. LaMA修复
- **特点**: 高质量图像修复，适合大面积区域
- **参数**: LDM步数、采样器、高清策略
- **优势**: 速度快、质量高

#### 2. Stable Diffusion 1.5 (计划中)
- **特点**: 可控的生成修复，支持提示词控制
- **参数**: 正面/负面提示词、采样步数
- **优势**: 更自然的修复效果

### 处理模式

#### 1. 修复模式
- 使用AI模型填充水印区域
- 保持图像整体一致性
- 适合大多数水印场景

#### 2. 透明模式
- 将水印区域设为透明
- 支持背景替换
- 适合Logo去除

## ⚙️ 配置说明

### 配置文件结构
```yaml
# web_config.yaml
app:
  title: "AI Watermark Remover"
  host: "0.0.0.0"
  port: 8501

models:
  florence_model: "microsoft/Florence-2-large"
  lama_model: "lama"

mask_generator:
  model_type: "custom"
  mask_model_path: "./models/epoch=071-valid_iou=0.7267.ckpt"
  mask_threshold: 0.5
  mask_dilate_kernel_size: 3

processing:
  default_max_bbox_percent: 10.0
  supported_formats: ["jpg", "jpeg", "png", "webp"]
```

### 环境配置
```bash
# 开发环境
export ENVIRONMENT=development

# 生产环境
export ENVIRONMENT=production
```

## 📊 性能指标

### 处理速度
- **LaMA修复**: 3-6秒/张 (2000x1500分辨率)
- **Florence-2检测**: 1-2秒/张
- **自定义分割**: 2-3秒/张

### 内存使用
- **GPU内存**: 2-4GB (推荐)
- **系统内存**: 1-2GB
- **存储空间**: 模型文件约1GB

### 支持格式
- **输入**: JPG, JPEG, PNG, WEBP
- **输出**: PNG, JPEG, WEBP (支持透明)

## 🧪 测试

### 运行测试
```bash
# 单元测试
python -m pytest tests/unit/

# 集成测试
python -m pytest tests/integration/

# 完整测试套件
python -m pytest tests/ -v
```

### 测试覆盖率
```bash
python -m pytest tests/ --cov=core --cov=interfaces --cov-report=html
```

## 🔧 开发指南

### 项目结构说明

#### Core模块
- **models/**: AI模型实现，遵循BaseModel接口
- **processors/**: 业务逻辑处理，协调各个组件
- **utils/**: 工具函数，按功能分类

#### Interfaces模块
- **web/**: Web界面，前后端分离设计
- **cli/**: 命令行界面，支持批处理
- **gui/**: 桌面GUI，基于PyQt6

### 扩展开发

#### 添加新模型
```python
from core.models.base_model import BaseModel

class NewModel(BaseModel):
    def _load_model(self):
        # 实现模型加载逻辑
        pass
    
    def predict(self, *args, **kwargs):
        # 实现推理逻辑
        pass

# 注册到工厂
from core.models.base_model import ModelFactory
ModelFactory.register("new_model", NewModel)
```

#### 添加新界面
```python
# 在interfaces/下创建新目录
# 实现界面逻辑
# 在app.py中添加启动选项
```

### 代码规范
- **命名**: 使用snake_case (函数/变量) 和 PascalCase (类)
- **文档**: 所有公共接口必须有文档字符串
- **测试**: 新功能必须包含测试用例
- **类型**: 使用类型注解提高代码可读性

## 📈 开发路线图

### 当前版本 (v2.0)
- ✅ 模块化架构重构
- ✅ 多模型支持
- ✅ Web界面优化
- ✅ 参数控制系统

### 短期计划 (1-2个月)
- 🎯 **SD1.5集成**: 添加Stable Diffusion 1.5修复模型
- 🎯 **批量处理**: 支持文件夹批量处理
- 🎯 **手绘Mask**: 交互式掩码编辑功能
- 🎯 **二次修正**: 支持多次处理和局部调整

### 中期计划 (3-6个月)
- 🎯 **模型扩展**: 集成YOLO、SAM等检测模型
- 🎯 **性能优化**: GPU加速、内存优化
- 🎯 **API服务**: 提供RESTful API接口
- 🎯 **云部署**: 支持云端部署和分布式处理

### 长期计划 (6个月+)
- 🎯 **商业化功能**: 用户管理、计费系统
- 🎯 **模型训练**: 支持自定义模型训练
- 🎯 **多语言支持**: 国际化界面
- 🎯 **移动端**: 移动应用开发

## 🤝 贡献指南

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 设置pre-commit钩子
pre-commit install
```

### 提交规范
- **feat**: 新功能
- **fix**: 错误修复
- **docs**: 文档更新
- **style**: 代码格式调整
- **refactor**: 代码重构
- **test**: 测试相关
- **chore**: 构建过程或辅助工具的变动

### 问题反馈
- 使用GitHub Issues报告问题
- 提供详细的错误信息和复现步骤
- 包含系统环境和版本信息

## 📄 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 🙏 致谢

- **Microsoft Florence-2**: 开放词汇检测模型
- **LaMA**: 高质量图像修复模型
- **iopaint**: 图像修复框架
- **Streamlit**: Web应用框架
- **PyTorch**: 深度学习框架

## 📞 联系方式

- **项目地址**: https://github.com/kevenGwong/watermarkremoverv--v0.9
- **问题反馈**: GitHub Issues
- **功能建议**: GitHub Discussions

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！**