# 🏗️ 重构后的项目结构

```
watermark_remover_ai/
├── 📁 core/                          # 核心业务逻辑
│   ├── 📁 models/                    # AI模型实现
│   │   ├── __init__.py
│   │   ├── base_model.py            # 模型基类
│   │   ├── florence_detector.py     # Florence-2检测模型
│   │   ├── custom_segmenter.py      # 自定义分割模型
│   │   └── lama_inpainter.py        # LaMA修复模型
│   ├── 📁 processors/               # 处理管道
│   │   ├── __init__.py
│   │   ├── watermark_processor.py   # 主处理器
│   │   ├── image_processor.py       # 图像处理
│   │   └── mask_processor.py        # 掩码处理
│   └── 📁 utils/                    # 核心工具
│       ├── __init__.py
│       ├── image_utils.py           # 图像处理工具
│       ├── mask_utils.py            # 掩码处理工具
│       ├── config_utils.py          # 配置管理
│       └── florence_utils.py        # Florence-2专用工具
├── 📁 interfaces/                    # 用户界面
│   ├── 📁 web/                      # Web界面
│   │   ├── __init__.py
│   │   ├── 📁 frontend/             # 前端组件
│   │   │   ├── __init__.py
│   │   │   ├── streamlit_app.py     # Streamlit主应用
│   │   │   ├── 📁 components/       # UI组件
│   │   │   │   ├── __init__.py
│   │   │   │   ├── parameter_panel.py    # 参数控制面板
│   │   │   │   ├── image_comparison.py   # 图像对比组件
│   │   │   │   └── download_buttons.py   # 下载按钮组件
│   │   │   └── 📁 layouts/          # 布局组件
│   │   │       ├── __init__.py
│   │   │       ├── main_layout.py        # 主布局
│   │   │       └── sidebar_layout.py     # 侧边栏布局
│   │   ├── 📁 backend/              # 后端API
│   │   │   ├── __init__.py
│   │   │   ├── api_routes.py        # API路由
│   │   │   └── request_handler.py   # 请求处理器
│   │   └── 📁 services/             # 业务服务
│   │       ├── __init__.py
│   │       ├── processing_service.py     # 处理服务
│   │       └── file_service.py           # 文件服务
│   ├── 📁 cli/                      # 命令行界面
│   │   ├── __init__.py
│   │   └── cli_app.py               # CLI应用
│   └── 📁 gui/                      # 桌面GUI
│       ├── __init__.py
│       └── qt_app.py                # Qt应用
├── 📁 config/                       # 配置管理
│   ├── __init__.py
│   ├── settings.py                  # 应用设置
│   ├── default_config.yaml          # 默认配置
│   └── 📁 environments/             # 环境配置
│       ├── __init__.py
│       ├── development.yaml         # 开发环境
│       └── production.yaml          # 生产环境
├── 📁 data/                         # 数据处理
│   ├── 📁 models/                   # 模型文件
│   ├── 📁 temp/                     # 临时文件
│   └── 📁 output/                   # 输出文件
├── 📁 tests/                        # 测试模块
│   ├── __init__.py
│   ├── 📁 unit/                     # 单元测试
│   │   ├── test_models.py
│   │   ├── test_processors.py
│   │   └── test_utils.py
│   ├── 📁 integration/              # 集成测试
│   │   ├── test_web_interface.py
│   │   └── test_processing_pipeline.py
│   └── 📁 fixtures/                 # 测试数据
├── 📁 scripts/                      # 脚本和工具
│   ├── start_web.sh                 # Web启动脚本
│   ├── start_cli.sh                 # CLI启动脚本
│   └── 📁 deployment/               # 部署脚本
├── 📁 docs/                         # 文档
│   ├── README.md                    # 项目说明
│   ├── API.md                       # API文档
│   ├── DEPLOYMENT.md                # 部署指南
│   └── 📁 development/              # 开发文档
├── 📁 archive/                      # 归档文件
│   └── original_files/              # 原始文件备份
├── app.py                           # 主应用入口
├── requirements.txt                 # Python依赖
├── setup.py                         # 安装脚本
└── .gitignore                       # Git忽略文件
```

## 🎯 **模块职责划分**

### 📁 **core/ - 核心业务逻辑**

#### 📁 **models/ - AI模型层**
- **base_model.py**: 所有模型的抽象基类，定义统一接口
- **florence_detector.py**: Florence-2检测模型封装
- **custom_segmenter.py**: 自定义分割模型封装
- **lama_inpainter.py**: LaMA修复模型封装

#### 📁 **processors/ - 处理管道**
- **watermark_processor.py**: 主处理器，协调各个组件
- **image_processor.py**: 图像预处理和后处理
- **mask_processor.py**: 掩码生成和处理

#### 📁 **utils/ - 工具函数**
- **image_utils.py**: 图像处理工具函数
- **mask_utils.py**: 掩码处理工具函数
- **config_utils.py**: 配置加载和管理
- **florence_utils.py**: Florence-2专用工具

### 📁 **interfaces/ - 用户界面层**

#### 📁 **web/ - Web界面**
- **frontend/**: Streamlit前端组件
- **backend/**: 后端API服务
- **services/**: 业务服务层

#### 📁 **cli/ - 命令行界面**
- **cli_app.py**: 命令行应用入口

#### 📁 **gui/ - 桌面GUI**
- **qt_app.py**: Qt桌面应用

### 📁 **config/ - 配置管理**
- **settings.py**: 应用设置和常量
- **default_config.yaml**: 默认配置文件
- **environments/**: 不同环境的配置

## 🔧 **重构优势**

### ✅ **清晰的职责分离**
- **模型层**: 纯AI模型实现，无UI逻辑
- **处理层**: 业务逻辑处理，无模型依赖
- **界面层**: 纯UI实现，无业务逻辑
- **配置层**: 集中配置管理

### ✅ **易于扩展**
- 新模型只需实现 `BaseModel` 接口
- 新界面只需在 `interfaces/` 下添加
- 新功能模块化，不影响其他部分

### ✅ **便于测试**
- 每个模块可独立测试
- 清晰的依赖关系
- 模拟接口便于单元测试

### ✅ **维护友好**
- 统一的命名规范
- 清晰的文档结构
- 模块化的错误处理

## 🚀 **迁移计划**

1. **第一阶段**: 创建新的目录结构
2. **第二阶段**: 迁移核心模块 (core/)
3. **第三阶段**: 重构界面层 (interfaces/)
4. **第四阶段**: 完善配置和文档
5. **第五阶段**: 添加测试和部署脚本

## 📋 **命名规范**

### 🏷️ **文件命名**
- 使用小写字母和下划线: `watermark_processor.py`
- 描述性名称: `florence_detector.py` 而不是 `utils.py`
- 模块名清晰表达功能

### 🏷️ **类命名**
- 使用PascalCase: `WatermarkProcessor`
- 描述性名称: `FlorenceDetector`
- 避免缩写: `ImageProcessor` 而不是 `ImgProc`

### 🏷️ **函数命名**
- 使用snake_case: `process_image()`
- 动词开头: `generate_mask()`, `load_model()`
- 描述性名称: `convert_bbox_to_relative()`

### 🏷️ **变量命名**
- 使用snake_case: `mask_threshold`
- 避免单字母变量: `image` 而不是 `img`
- 常量使用大写: `MAX_FILE_SIZE` 