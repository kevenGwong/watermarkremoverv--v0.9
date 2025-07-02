# 📁 Directory Structure

## 🎯 Core Files (Root Directory)

### 🚀 Main Applications
- **`watermark_web_app_debug.py`** - 🔬 **主要程序** - Debug版本Web UI（推荐使用）
- **`run_debug_app.sh`** - 🚀 Debug版本启动脚本
- **`remwm.py`** - 🖥️ 原始CLI版本
- **`remwmgui.py`** - 🖥️ 原始GUI版本

### ⚙️ Core Backend & Config
- **`web_backend.py`** - 🔧 核心后端处理器
- **`web_config.yaml`** - ⚙️ 核心配置文件
- **`utils.py`** - 🛠️ 工具函数

### 📋 Documentation
- **`README.md`** - 📖 项目主要说明
- **`DEBUG_UI_GUIDE.md`** - 🔬 Debug UI使用指南
- **`PARAMETER_RANGES_UPDATE.md`** - 📊 参数范围说明
- **`DIRECTORY_STRUCTURE.md`** - 📁 本文件

### 🏗️ Setup & Requirements
- **`setup.sh`** - 🏗️ 环境安装脚本
- **`environment.yml`** - 🐍 Conda环境文件
- **`requirements_web.txt`** - 📦 Web依赖

### 📂 Data Directories
- **`models/`** - 🤖 AI模型文件
  - `epoch=071-valid_iou=0.7267.ckpt` - Custom watermark model
- **`test/`** - 📊 测试数据集
- **`test_images/`** - 🖼️ 测试图片
- **`debug_output/`** - 🔍 调试输出
- **`归档/`** - 📦 历史归档

---

## 📦 Scripts Directory

### 🧪 Test Scripts (`scripts/test_scripts/`)
- `test_functionality.py` - 功能测试
- `test_image_processing.py` - 图像处理测试
- `test_parameter_effects.py` - 参数效果测试
- `test_web_startup.py` - Web启动测试
- `test_web_backend.py` - 后端测试
- `quick_test.py` - 快速测试
- `debug_transparent_issue.py` - 透明问题调试
- `validate_consistency.py` - 一致性验证

### 📊 Test Outputs (`scripts/test_outputs/`)
- `test_*.png` - 各种测试输出图片
- `parameter_test_report.md` - 参数测试报告

### 📱 Deprecated Apps (`scripts/deprecated_apps/`)
- `watermark_web_app.py` - 原始Web版本
- `watermark_web_app_enhanced.py` - 增强版本
- `watermark_web_app_pro.py` - 专业版本
- `watermark_web_app_simple.py` - 简化版本
- `watermark_web_app_v2.py` - V2版本
- `run_*_app.sh` - 对应启动脚本
- `web_backend_advanced.py` - 高级后端
- `web_config_advanced.yaml` - 高级配置

### 📖 Documentation Archive (`scripts/docs/`)
- `DEVELOPMENT_SUMMARY.md` - 开发总结
- `PARAMETER_GUIDE.md` - 参数指南
- `TEST_REPORT.md` - 测试报告
- `WEB_UI_README.md` - Web UI说明

---

## 🎯 Recommended Usage

### 🚀 Quick Start
```bash
# 激活环境
conda activate py310aiwatermark

# 启动主要应用
./run_debug_app.sh

# 访问Web UI
http://localhost:8506
```

### 🔧 Development
```bash
# 查看测试脚本
ls scripts/test_scripts/

# 运行功能测试
python scripts/test_scripts/test_functionality.py

# 查看历史版本
ls scripts/deprecated_apps/
```

### 📚 Documentation
- 主要使用指南: `DEBUG_UI_GUIDE.md`
- 参数说明: `PARAMETER_RANGES_UPDATE.md`
- 历史文档: `scripts/docs/`

---

## 🎉 Benefits of This Structure

### ✅ Clean Root Directory
- 只保留核心、常用文件
- 启动和使用更简单
- 避免混淆

### 📦 Organized Scripts
- 测试脚本分类存放
- 弃用版本归档保留
- 文档集中管理

### 🚀 Easy Maintenance
- 清晰的文件职责
- 便于版本管理
- 简化部署流程

---

**🎯 现在目录结构清晰，专注于Debug UI的使用和改进！**