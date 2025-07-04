# Claude 上下文提示词 - WatermarkRemover-AI项目

## 🎯 项目核心信息

### 项目定位
你正在协助优化一个**基于IOPaint的自定义WebUI水印去除项目**。这是一个模块化重构后的系统，主要功能是利用自定义模型生成水印mask，然后使用ZITS/MAT/FCF/LaMA等处理模型进行水印修复。

### 技术栈
- **前端**: Streamlit WebUI
- **后端**: IOPaint统一处理框架  
- **模型**: ZITS(结构保持) + MAT(最佳质量) + FCF(快速修复) + LaMA(最快速度)
- **Mask生成**: 自定义FPN+MIT-B5模型 + Florence-2 + 手动上传
- **环境**: conda py310aiwatermark

### 项目结构
```
WatermarkRemover-AI/
├── core/                           # 核心推理逻辑
│   ├── inference.py                # 主入口API
│   ├── inference_manager.py        # 推理管理器(调度)
│   ├── models/                     # 模型处理器
│   │   ├── unified_processor.py    # 统一模型管理
│   │   ├── zits_processor.py       # ZITS专用处理器
│   │   ├── mat_processor.py        # MAT专用处理器  
│   │   ├── fcf_processor.py        # FCF专用处理器
│   │   ├── lama_processor.py       # LaMA修复处理器
│   │   └── mask_generators.py      # Mask生成器集合
│   └── processors/                 # 处理器和结果
│       ├── watermark_processor.py  # 主处理器
│       └── processing_result.py    # 结果数据类
├── interfaces/web/                 # Web界面
│   ├── main.py                     # Streamlit主入口
│   └── ui.py                       # UI组件和参数面板
├── config/                         # 配置管理
│   ├── config.py                   # 配置管理器
│   └── iopaint_config.yaml         # IOPaint配置
└── data/models/                    # 模型文件存储
```

## 📋 当前项目状态 (2025-07-05 2:27 AM)

### ✅ 已完成的重大工作
1. **模块化重构完成** - 从单文件脚本重构为清晰的模块化架构
2. **IOPaint集成** - 完全移除PowerPaint，集成IOPaint统一框架
3. **多模型支持** - 支持ZITS、MAT、FCF、LaMA四种先进模型
4. **关键bug修复** - 递归调用、方法名统一、参数传递等问题已解决
5. **功能验证通过** - 所有核心功能测试成功

### ✅ 验证通过的功能
- **Web UI启动** - Streamlit界面正常，无import错误
- **模型加载** - ZITS、MAT、FCF成功加载(LaMA缺saicinpainting依赖)
- **图像处理** - MAT模型测试成功，处理时间1.76秒
- **参数传递** - UI参数正确传递到底层处理模块
- **Mask功能** - 自定义mask上传和透明处理正常
- **输出验证** - 支持PNG/JPEG/WebP保存，修复效果显著

### ⚠️ 已知限制和待优化项
- **LaMA模型** - 需要安装saicinpainting依赖
- **自定义分割模型** - 模型文件路径需要配置
- **UI刷新优化** - 模型切换时需要确保预览刷新
- **高清处理策略** - 需要验证无压缩、无不必要resize

### 🔧 关键修复记录
1. **递归调用问题** - `inference.py`中使用`RealInferenceManager`避免无限循环
2. **方法名统一** - 所有mask生成器统一使用`generate_mask()`方法
3. **参数修正** - `ProcessingResult`类参数: success/result_image/mask_image/processing_time
4. **手动选择** - 移除智能模型选择，使用`force_model`手动指定模型

## 🎯 明天的主要任务

### 1. 功能全面测试
- 测试所有三个模型的完整处理流程
- 验证不同图像尺寸和格式的处理能力  
- 测试边界情况和错误处理

### 2. IOPaint标准对齐
**参考**: https://www.iopaint.com/
- 确认图像输入格式(RGB vs RGBA vs BGR)
- 验证模型配置和预处理流程
- 对比官方IOPaint的处理pipeline

### 3. UI交互优化
- 确保模型切换时图片预览实时刷新
- 优化st.session_state状态管理
- 验证Before/After对比组件

### 4. 高清处理策略
- 验证ORIGINAL策略下图像不被resize
- 确认hd_strategy参数正确传递
- 测试大尺寸图像处理质量

### 5. 数据流验证
- UI → inference.py → inference_manager.py → unified_processor.py
- 确保mask和图片正确加载到模型中

### 6. 模型路径配置
- 检查IOPaint模型缓存路径
- 验证自定义mask模型路径配置

## 💡 工作方式指南

### 启动和测试流程
```bash
# 1. 环境激活
conda activate py310aiwatermark

# 2. 启动应用
streamlit run interfaces/web/main.py --server.port 8501

# 3. 测试命令示例
python -c "
from config.config import ConfigManager
from core.inference import process_image
from PIL import Image
config_manager = ConfigManager()
test_image = Image.new('RGB', (100, 100), 'red')
result = process_image(test_image, mask_model='custom', 
                      inpaint_params={'force_model': 'mat'}, 
                      config_manager=config_manager)
print(f'Success: {result.success}')
"
```

### 调试要点
- **模型加载问题** - 检查`unified_processor.py`中的processors字典
- **UI刷新问题** - 关注`st.session_state`和`st.rerun()`调用  
- **参数传递问题** - 验证`_generate_mask()`方法的参数传递
- **图像质量问题** - 检查`hd_strategy`设置和resize逻辑

### 常见错误模式
- `RecursionError` → 检查inference.py中的循环调用
- `AttributeError: generate` → 确认mask生成器方法名为generate_mask
- `ProcessingResult unexpected argument` → 检查参数名是否匹配数据类定义
- `model not loaded` → 验证模型路径和依赖安装

## 🎨 用户体验目标

### 核心价值主张
**我们的目的是测试不同模型**，所以必须确保：
1. 每次切换模型时图片预览和结果都要刷新
2. 图片是高清处理策略，无压缩，无resize(除非选择crop策略)
3. Mask和图片正确加载到模型中处理
4. 模型路径设置正确

### 用户工作流
```
用户上传图像 → 选择mask生成方式(custom/florence2/upload) → 
选择inpainting模型(zits/mat/fcf) → 调整参数 → 
点击处理 → 查看Before/After对比 → 下载结果
```

### 期望的技术指标
- **处理时间**: ZITS<4s, MAT<2s, FCF<2s
- **内存使用**: <8GB GPU, <4GB RAM  
- **图像质量**: 无明显伪影或失真
- **UI响应**: 切换无延迟，预览实时更新

## 🔄 重启后快速检查清单

### 立即验证点
1. [ ] `streamlit run interfaces/web/main.py` 能否正常启动
2. [ ] UI界面是否完整显示（无组件缺失）
3. [ ] 模型选择器是否显示ZITS/MAT/FCF选项
4. [ ] 测试一次完整的图像处理流程

### 如果遇到问题
- **Import错误** → 检查conda环境激活
- **模型加载失败** → 验证IOPaint依赖和网络连接
- **UI显示异常** → 清除streamlit缓存，重启服务

### 项目健康度快速检测
```python
# 5分钟快速健康检查脚本
from config.config import ConfigManager
from core.inference import get_inference_manager

config = ConfigManager()
manager = get_inference_manager(config)
if manager:
    available = manager.unified_processor.get_available_models()
    print(f"✅ Available models: {available}")
else:
    print("❌ Manager initialization failed")
```

---

**这个提示词确保你在重启后能够：**
1. **快速理解项目** - 清楚技术栈、架构、当前状态
2. **了解进度** - 知道已完成什么、遇到什么问题、下一步做什么  
3. **无缝连接工作** - 有具体的任务清单、测试方法、调试指南

**使用方式**: 将此提示词作为对话开始时的上下文，确保工作连续性和效率。