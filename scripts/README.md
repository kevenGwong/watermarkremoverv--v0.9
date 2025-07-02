# 📦 Scripts Directory

此目录包含测试脚本、弃用应用和历史文档。

## 📂 Directory Structure

### 🧪 `test_scripts/` - 测试脚本
包含各种功能测试和验证脚本：
- `test_functionality.py` - 模块功能测试
- `test_image_processing.py` - 图像处理测试  
- `test_parameter_effects.py` - 参数效果验证
- `test_web_startup.py` - Web应用启动测试
- `quick_test.py` - 快速验证脚本

### 📊 `test_outputs/` - 测试输出
包含测试生成的图片和报告：
- `test_*.png` - 各种参数测试的输出图片
- `parameter_test_report.md` - 参数效果测试报告

### 📱 `deprecated_apps/` - 弃用应用
包含历史版本的Web应用：
- `watermark_web_app*.py` - 各个版本的Web UI
- `run_*_app.sh` - 对应的启动脚本
- `web_backend_advanced.py` - 复杂版本的后端
- `web_config_advanced.yaml` - 高级配置文件

### 📖 `docs/` - 文档归档
包含开发过程中的文档：
- `DEVELOPMENT_SUMMARY.md` - 开发历程总结
- `PARAMETER_GUIDE.md` - 详细参数说明
- `TEST_REPORT.md` - 测试结果报告

## 🎯 Usage

### 运行测试
```bash
# 从根目录运行
python scripts/test_scripts/test_functionality.py
```

### 查看历史版本
```bash
# 查看弃用的应用
ls scripts/deprecated_apps/
```

### 参考文档
```bash
# 查看历史文档
ls scripts/docs/
```

## ⚠️ Note

这些文件主要用于：
- 🔍 开发调试和测试
- 📚 学习和参考历史实现
- 🗄️ 保留完整的开发记录

**主要使用请返回根目录的 `watermark_web_app_debug.py`**