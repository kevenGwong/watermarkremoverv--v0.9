# WatermarkRemover-AI

## 项目背景
WatermarkRemover-AI 致力于为商品图像、照片等提供高效、智能的水印去除能力。项目经历了从早期单文件脚本到模块化架构的重构，现已支持灵活的参数调节和多种去水印模型，具备Web UI交互能力。

## 当前架构
项目采用模块化设计，核心目录结构如下：

```
WatermarkRemover-AI/
├── core/           # 推理与图像处理核心逻辑
│   ├── inference.py         # 推理主流程，管理AI模型调用
│   └── utils/
│       └── image_utils.py   # 图像处理工具（如mask膨胀、格式转换等）
├── config/         # 配置与参数管理
│   └── config.py           # 配置管理器，参数校验与默认值
├── interfaces/     # 各类交互界面
│   └── web/
│       ├── main.py         # Streamlit Web UI 主入口
│       └── ui.py           # Web UI 交互与参数面板
├── models/         # AI模型文件（如PowerPaint v2等）
└── README.md       # 项目说明
```

> **注意：app.py 已废弃，无需再使用。主入口为 `interfaces/web/main.py`。**

## 各模块/子文件夹作用
- **core/**：AI推理主流程与图像处理工具，所有AI相关逻辑集中于此。
- **core/utils/**：图像处理相关的静态工具函数。
- **config/**：配置管理与参数校验，支持自定义和默认参数。
- **interfaces/web/**：基于Streamlit的Web UI，负责用户交互、参数输入、结果展示。
- **models/**：存放AI模型权重文件（如PowerPaint v2、LaMA等）。

## 已解决的问题

### 架构重构与代码优化
- 项目结构混乱、历史与新文件混杂 → 已彻底模块化、归档/删除历史文件。
- 旧依赖（如web_backend.py）已弃用，所有import路径已统一修正。
- Streamlit Web UI 可正常启动，主流程可用。
- 代码已用.gitignore排除缓存和临时文件，核心代码已备份。

### 关键缺陷修复 (2025-01-04)
- **模块耦合问题**：配置文件路径硬编码 → 支持依赖注入，配置路径可自定义
- **错误处理缺陷**：模型加载失败导致崩溃 → 增加降级方案和详细错误提示
- **资源管理问题**：CUDA内存泄漏 → 实现自动资源清理和上下文管理器
- **功能逻辑缺陷**：上传mask无验证 → 增加完整的输入验证和错误处理
- **配置管理混乱**：重复定义配置 → 统一使用ConfigManager，消除重复配置
- **废弃文件清理**：删除已废弃的app.py文件

### PowerPaint水印去除问题修复 (2025-01-04 下午)
- **🚨 致命问题**：PowerPaint V2 processor中mask逻辑完全颠倒 → 修正mask composite逻辑
- **Mask处理问题**：LANCZOS resize导致mask边缘模糊 → 统一使用NEAREST保持锐利边缘  
- **模型路径错误**：硬编码错误路径导致模型加载失败 → 修正为正确路径
- **验证缺失**：缺少mask有效性检查可能传入空mask → 添加完整验证和警告
- **参数传递问题**：UI设置被默认值覆盖 → 确保UI参数优先级最高
- **静默回滚**：PowerPaint失败时静默回滚到LaMA → 详细日志记录真实执行路径

## PowerPaint状态更新 (2025-01-04)

### ✅ 已完成的核心修复
1. **mask逻辑修正** - 解决了最致命的mask composite逻辑颠倒问题
2. **图像处理优化** - 统一resize方法，保持mask精确性
3. **验证机制** - 添加mask有效性检查，防止空mask
4. **模型路径修正** - 更新为正确的PowerPaint v2模型路径
5. **参数传递优化** - 确保UI参数正确传递

### 🎯 重要发现
**标准SD1.5已经足够满足水印去除需求！**
- 当前实现使用标准SD1.5 inpainting pipeline
- 包含针对object removal优化的prompt
- 修复后应该已经有很好的效果

### 📋 PowerPaint v2 集成计划（可选）
1. **安装PowerPaint依赖**：如果需要更精确控制
2. **BrushNet集成**：获得专门针对inpainting优化的架构
3. **Task Token支持**：使用P_obj、P_ctxt等专用token

## IOPaint重构完成 (2025-07-04)

### ✅ 重大架构重构
1. **PowerPaint到IOPaint迁移** - 完全移除自定义PowerPaint实现，集成IOPaint统一框架
2. **多模型支持** - 支持ZITS、MAT、FCF、LaMA四种先进模型
3. **智能模型选择** - 根据图像复杂度和mask覆盖率自动选择最佳模型
4. **代码简化** - 代码量减少约70%，架构更清晰

### 🎯 核心技术方案
**IOPaint统一处理器：**
- **架构**: `IOPaintProcessor` (统一接口，支持多种模型)
- **模型**: ZITS(结构保持)、MAT(最佳质量)、FCF(快速修复)、LaMA(最快速度)
- **关键特性**: 智能模型选择，自动切换，统一配置

### 📊 性能表现
| 功能 | 处理时间 | 技术方案 | 效果评价 |
|------|----------|----------|----------|
| **ZITS Inpainting** | ~3.8秒 | 最佳结构保持 | ✅ 复杂图像专业修复 |
| **MAT Inpainting** | ~1.7秒 | 最佳质量 | ✅ 大水印高质量去除 |
| **FCF Inpainting** | ~1.5秒 | 快速修复 | ✅ 平衡性能质量 |
| **LaMA Inpainting** | ~1.0秒 | 最快速度 | ✅ 小水印快速修复 |
| **自定义Mask生成** | <0.1秒 | FPN + MIT-B5 | ✅ 精准水印检测 |

### 🚀 使用方法
```bash
# 启动应用
streamlit run interfaces/web/main.py --server.port 8501

# Web界面使用IOPaint:
# 1. 选择 iopaint 模型
# 2. 选择具体模型: auto/zits/mat/fcf/lama
# 3. 系统将自动选择或使用指定模型处理
```

### 🔧 重构成果
- **删除文件**: PowerPaint处理器、模块目录、模型文件(释放4-6GB空间)
- **新增文件**: IOPaint处理器、配置文件、测试脚本
- **修改文件**: Web UI、推理模块、配置管理
- **测试验证**: 所有模型切换功能正常，结果差异明显

## 技术改进详情

### 1. 模块解耦优化
- **InferenceManager** 现在接受可选的配置文件路径，避免硬编码
- **WatermarkProcessor** 支持自定义配置路径，提高灵活性
- 移除循环依赖，增强代码可测试性

### 2. 错误处理增强
- **模型加载失败**：提供降级方案，避免整个系统崩溃
- **自定义模型缺失**：自动切换到fallback mask生成器
- **LaMA初始化失败**：提供明确的错误信息和诊断建议

### 3. 资源管理优化
- **BaseModel类**：实现上下文管理器和自动资源清理
- **CUDA内存管理**：自动清理GPU缓存，避免内存泄漏
- **模型注册机制**：统一管理加载的模型，确保正确释放

### 4. 输入验证完善
- **上传mask验证**：检查文件完整性、格式正确性、尺寸有效性
- **空值检查**：UI层面增加null检查，避免运行时错误
- **参数验证**：完善配置参数的范围和类型检查

### 5. 配置管理统一
- **消除重复配置**：inference.py不再重复定义默认值
- **ConfigManager集成**：统一使用ConfigManager获取默认配置
- **文件清理**：移除废弃的app.py文件

### 6. PowerPaint核心问题修复 (2025-01-04)
- **Mask逻辑修正**：修复PowerPaint V2中致命的mask composite逻辑颠倒问题
- **图像预处理优化**：统一mask resize方法为NEAREST，避免边缘模糊
- **验证机制完善**：添加mask有效性检查，防止空mask导致处理失效
- **路径配置修正**：更新模型路径为正确的PowerPaint v2模型位置
- **参数传递改进**：确保Web UI设置的参数正确传递给处理器

---

如需开发、测试或集成新模型，请以`interfaces/web/main.py`为主入口，遵循上述模块结构进行扩展。现在的架构具备更好的错误恢复能力和资源管理机制。 