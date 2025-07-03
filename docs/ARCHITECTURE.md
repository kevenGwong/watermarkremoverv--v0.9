# 🏗️ 系统架构说明

## 📋 概述

AI Watermark Remover v2.0 采用模块化、可扩展的架构设计，实现了核心业务逻辑与用户界面的完全分离。系统支持多种AI模型和用户界面，便于维护、测试和扩展。

## 🎯 设计原则

### 1. 单一职责原则
- 每个模块只负责一个特定功能
- 模型层只处理AI推理，不涉及UI逻辑
- 界面层只处理用户交互，不包含业务逻辑

### 2. 开闭原则
- 对扩展开放，对修改封闭
- 新模型只需实现BaseModel接口
- 新界面只需在interfaces下添加

### 3. 依赖倒置原则
- 高层模块不依赖低层模块
- 都依赖抽象接口
- 便于单元测试和模块替换

### 4. 接口隔离原则
- 客户端不应该依赖它不需要的接口
- 每个接口功能单一，职责明确

## 🏛️ 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interfaces Layer                    │
├─────────────────────────────────────────────────────────────┤
│  Web UI  │  CLI  │  GUI  │  API  │  Mobile  │  Desktop     │
├─────────────────────────────────────────────────────────────┤
│                  Business Logic Layer                       │
├─────────────────────────────────────────────────────────────┤
│  WatermarkProcessor  │  ImageProcessor  │  MaskProcessor   │
├─────────────────────────────────────────────────────────────┤
│                    Model Layer                              │
├─────────────────────────────────────────────────────────────┤
│  Florence-2  │  Custom Seg  │  LaMA  │  SD1.5  │  YOLO      │
├─────────────────────────────────────────────────────────────┤
│                   Utility Layer                             │
├─────────────────────────────────────────────────────────────┤
│  Image Utils  │  Config Utils  │  Mask Utils  │  Florence   │
├─────────────────────────────────────────────────────────────┤
│                  Configuration Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Settings  │  Environment  │  Model Config  │  UI Config   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 模块详解

### 1. Core模块 (`core/`)

#### 1.1 Models (`core/models/`)

**职责**: AI模型的抽象和实现

**核心组件**:
- `base_model.py`: 所有模型的抽象基类
- `florence_detector.py`: Florence-2检测模型
- `custom_segmenter.py`: 自定义分割模型
- `lama_inpainter.py`: LaMA修复模型

**设计模式**: 工厂模式 + 策略模式

```python
# 模型工厂使用示例
from core.models.base_model import ModelFactory

# 注册模型
ModelFactory.register("florence", FlorenceDetector)
ModelFactory.register("custom", CustomSegmenter)
ModelFactory.register("lama", LamaInpainter)

# 创建模型实例
detector = ModelFactory.create("florence", config)
segmenter = ModelFactory.create("custom", config)
inpainter = ModelFactory.create("lama", config)
```

#### 1.2 Processors (`core/processors/`)

**职责**: 业务逻辑处理和流程协调

**核心组件**:
- `watermark_processor.py`: 主处理器，协调各个组件
- `image_processor.py`: 图像预处理和后处理
- `mask_processor.py`: 掩码生成和处理

**设计模式**: 管道模式 + 责任链模式

```python
# 处理流程示例
class WatermarkProcessor:
    def process_image(self, image, params):
        # 1. 图像预处理
        processed_image = self.image_processor.preprocess(image)
        
        # 2. 生成掩码
        mask = self.mask_processor.generate_mask(processed_image, params)
        
        # 3. 图像修复
        result = self.inpainter.inpaint(processed_image, mask, params)
        
        # 4. 后处理
        final_result = self.image_processor.postprocess(result)
        
        return final_result
```

#### 1.3 Utils (`core/utils/`)

**职责**: 通用工具函数和辅助功能

**核心组件**:
- `image_utils.py`: 图像处理工具
- `mask_utils.py`: 掩码处理工具
- `config_utils.py`: 配置管理工具
- `florence_utils.py`: Florence-2专用工具

### 2. Interfaces模块 (`interfaces/`)

#### 2.1 Web界面 (`interfaces/web/`)

**架构**: 前后端分离

**前端组件** (`frontend/`):
- `streamlit_app.py`: 主应用入口
- `components/`: UI组件库
  - `parameter_panel.py`: 参数控制面板
  - `image_comparison.py`: 图像对比组件
  - `download_buttons.py`: 下载按钮组件
- `layouts/`: 布局组件
  - `main_layout.py`: 主布局
  - `sidebar_layout.py`: 侧边栏布局

**后端服务** (`backend/`):
- `api_routes.py`: API路由定义
- `request_handler.py`: 请求处理器

**业务服务** (`services/`):
- `processing_service.py`: 处理服务
- `file_service.py`: 文件服务

#### 2.2 CLI界面 (`interfaces/cli/`)

**特点**: 命令行工具，支持批处理

**功能**:
- 单文件处理
- 批量处理
- 参数配置
- 进度显示

#### 2.3 GUI界面 (`interfaces/gui/`)

**特点**: 桌面应用，基于PyQt6

**功能**:
- 拖拽上传
- 实时预览
- 参数调节
- 结果对比

### 3. Config模块 (`config/`)

**职责**: 配置管理和环境设置

**组件**:
- `settings.py`: 应用设置和常量
- `default_config.yaml`: 默认配置
- `environments/`: 环境特定配置
  - `development.yaml`: 开发环境
  - `production.yaml`: 生产环境

### 4. Data模块 (`data/`)

**职责**: 数据存储和管理

**结构**:
- `models/`: 模型文件存储
- `temp/`: 临时文件
- `output/`: 输出文件

### 5. Tests模块 (`tests/`)

**职责**: 测试用例和测试数据

**结构**:
- `unit/`: 单元测试
- `integration/`: 集成测试
- `fixtures/`: 测试数据

## 🔄 数据流

### 1. Web界面数据流

```
用户上传 → Streamlit前端 → 后端API → 处理服务 → 模型层 → 结果返回 → 前端显示
```

### 2. CLI界面数据流

```
命令行参数 → CLI处理器 → 主处理器 → 模型层 → 结果保存
```

### 3. 处理流程

```
输入图像 → 预处理 → 检测/分割 → 掩码生成 → 图像修复 → 后处理 → 输出
```

## 🔧 扩展机制

### 1. 添加新模型

```python
# 1. 继承BaseModel
class NewModel(BaseModel):
    def _load_model(self):
        # 实现模型加载
        pass
    
    def predict(self, *args, **kwargs):
        # 实现推理逻辑
        pass

# 2. 注册到工厂
ModelFactory.register("new_model", NewModel)

# 3. 在配置中添加
models:
  new_model: "path/to/model"
```

### 2. 添加新界面

```python
# 1. 在interfaces/下创建新目录
interfaces/new_interface/
├── __init__.py
├── main.py
└── components/

# 2. 实现界面逻辑
# 3. 在app.py中添加启动选项
```

### 3. 添加新工具

```python
# 1. 在core/utils/下创建新文件
# 2. 实现工具函数
# 3. 在需要的地方导入使用
```

## 🧪 测试架构

### 1. 单元测试

- **模型测试**: 测试每个模型的加载和推理
- **工具测试**: 测试工具函数的正确性
- **配置测试**: 测试配置加载和验证

### 2. 集成测试

- **处理流程测试**: 测试完整的处理流程
- **界面测试**: 测试用户界面的功能
- **API测试**: 测试后端API的正确性

### 3. 性能测试

- **内存使用测试**: 监控内存使用情况
- **处理速度测试**: 测试处理速度
- **并发测试**: 测试并发处理能力

## 🔒 安全考虑

### 1. 输入验证

- 文件类型验证
- 文件大小限制
- 恶意文件检测

### 2. 错误处理

- 异常捕获和处理
- 用户友好的错误信息
- 日志记录和监控

### 3. 资源管理

- 内存使用监控
- GPU资源管理
- 临时文件清理

## 📊 性能优化

### 1. 模型优化

- 模型量化
- 混合精度训练
- 模型剪枝

### 2. 内存优化

- 流式处理
- 内存池管理
- 垃圾回收优化

### 3. 并发优化

- 异步处理
- 线程池管理
- 负载均衡

## 🚀 部署架构

### 1. 单机部署

```
用户 → Nginx → Streamlit → Python应用 → GPU
```

### 2. 分布式部署

```
用户 → 负载均衡器 → 多个应用实例 → 模型服务 → GPU集群
```

### 3. 容器化部署

```
Docker容器 → Kubernetes集群 → GPU节点
```

## 📈 监控和日志

### 1. 日志系统

- 结构化日志
- 日志级别管理
- 日志轮转

### 2. 监控指标

- 处理速度
- 内存使用
- GPU利用率
- 错误率

### 3. 告警系统

- 性能告警
- 错误告警
- 资源告警

## 🔮 未来架构演进

### 1. 微服务化

- 模型服务独立部署
- 处理服务独立部署
- 界面服务独立部署

### 2. 云原生

- Kubernetes部署
- 服务网格
- 云存储集成

### 3. 边缘计算

- 轻量级模型
- 本地推理
- 云端协同

---

**📝 本文档会随着项目发展持续更新，请关注最新版本。** 