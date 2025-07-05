# PowerPaint到IOPaint重构方案

## 📋 重构目标

**主要目标：**
- 彻底移除所有自定义PowerPaint实现
- 替换为IOPaint集成方式
- 支持MAT、ZITS等先进模型
- 简化代码架构，提高维护性

**预期收益：**
- 代码量减少约70%
- 支持更多先进模型（MAT、ZITS、MIGAN等）
- 更简洁的架构
- 更好的水印去除效果

---

## 🔍 当前PowerPaint相关代码分析

### 📁 需要删除的文件和目录

#### 1. 核心处理器文件
```
core/models/powerpaint_processor.py
core/models/powerpaint_v2_processor.py  
core/models/powerpaint_v2_real_processor.py
```

#### 2. PowerPaint模块目录（整个目录）
```
powerpaint/
├── __init__.py
├── datasets/
│   ├── __init__.py
│   ├── laion.py
│   └── openimage.py
├── models/
│   ├── __init__.py
│   ├── brushnet.py
│   ├── unet_2d_blocks.py
│   └── unet_2d_condition.py
├── pipelines/
│   ├── __init__.py
│   ├── pipeline_powerpaint.py
│   ├── pipeline_powerpaint_brushnet.py
│   └── pipeline_powerpaint_controlnet.py
└── utils/
    ├── __init__.py
    └── loaders.py
```

#### 3. 模型文件目录
```
models/powerpaint_v2/
└── Realistic_Vision_V1.4-inpainting/
    ├── feature_extractor/
    ├── safety_checker/
    ├── scheduler/
    ├── text_encoder/
    ├── tokenizer/
    ├── unet/
    └── vae/
```
**估计空间：约4-6GB**

#### 4. 脚本文件
```
scripts/download_powerpaint_model.py
scripts/test_powerpaint_integration.py
```

#### 5. 测试文件中的PowerPaint相关部分
```
test_high_resolution_fix.py (PowerPaint测试部分)
test_ui_functionality.py (PowerPaint相关测试)
test_integration.py (PowerPaint集成测试)
test_full_pipeline.py (PowerPaint pipeline测试)
scripts/test_webui_flow.py (PowerPaint workflow测试)
```

### 🔧 需要修改的文件

#### 1. 核心推理模块
- `core/inference.py`
  - 移除PowerPaint处理器导入
  - 移除PowerPaint相关逻辑
  - 简化模型选择逻辑

#### 2. 配置文件
- `config/powerpaint_config.yaml`
  - 移除PowerPaint特定配置
  - 添加IOPaint模型配置
  - 简化配置结构

- `config/config.py`
  - 移除PowerPaint参数验证
  - 添加IOPaint模型选择支持

#### 3. Web界面
- `interfaces/web/ui.py`
  - 移除PowerPaint特定UI控件
  - 添加IOPaint模型选择器
  - 简化参数配置界面

#### 4. 模块初始化
- `core/models/__init__.py`
  - 移除PowerPaint处理器导入

---

## 🏗️ 新架构设计

### 📦 简化后的目录结构

```
WatermarkRemover-AI/
├── core/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   └── iopaint_processor.py  # 新：统一的IOPaint处理器
│   ├── utils/
│   │   └── image_utils.py
│   └── inference.py              # 简化版
├── config/
│   ├── config.py                 # 简化版
│   └── iopaint_config.yaml       # 新：IOPaint配置
├── interfaces/
│   └── web/
│       ├── main.py
│       └── ui.py                 # 简化版
└── data/                         # 保留自定义mask模型
    └── models/
        └── epoch=071-valid_iou=0.7267.ckpt
```

### 🔄 新的处理流程

```
用户上传图像
    ↓
自定义Mask生成 (保留现有FPN模型)
    ↓
IOPaint处理器 (统一接口)
    ↓
模型选择 (MAT/ZITS/LaMA)
    ↓
一键Inpainting
    ↓
返回结果
```

---

## 💻 具体实现方案

### 阶段1：IOPaint升级和测试

#### 1.1 升级IOPaint
```bash
# 在conda环境中执行
/home/duolaameng/miniconda/envs/py310aiwatermark/bin/pip install --upgrade iopaint
```

#### 1.2 测试新模型可用性
```bash
# 测试MAT模型
iopaint download --model mat

# 测试ZITS模型  
iopaint download --model zits

# 测试fcf模型
iopaint download --model fcf
```

### 阶段2：创建新的IOPaint处理器

#### 2.1 创建统一处理器
**文件：`core/models/iopaint_processor.py`**

```python
"""
IOPaint统一处理器
支持ZITS、MAT、FCF、LaMA等多种模型
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class IOPaintProcessor(BaseModel):
    \"\"\"IOPaint统一处理器，支持多种先进模型\"\"\"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_manager = None
        self.current_model = None
        self._load_model()
    
    def _load_model(self):
        \"\"\"加载IOPaint模型\"\"\"
        try:
            from iopaint.model_manager import ModelManager
            from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
            
            # 获取模型名称，默认使用MAT
            model_name = self.config.get('models', {}).get('inpaint_model', 'mat')
            
            self.model_manager = ModelManager(name=model_name, device=str(self.device))
            self.current_model = model_name
            self.register_model(self.model_manager)
            
            # 存储配置类
            self.HDStrategy = HDStrategy
            self.LDMSampler = LDMSampler
            self.Config = Config
            
            self.model_loaded = True
            logger.info(f\"✅ IOPaint模型加载成功: {model_name}\")
            logger.info(f\"   设备: {self.device}\")
            
        except Exception as e:
            logger.error(f\"❌ IOPaint模型加载失败: {e}\")
            self.model_loaded = False
            raise
    
    def switch_model(self, model_name: str):
        \"\"\"动态切换模型\"\"\"
        if model_name == self.current_model:
            return
            
        try:
            from iopaint.model_manager import ModelManager
            
            # 清理旧模型
            if self.model_manager:
                del self.model_manager
                
            # 加载新模型
            self.model_manager = ModelManager(name=model_name, device=str(self.device))
            self.current_model = model_name
            self.register_model(self.model_manager)
            
            logger.info(f\"🔄 模型切换成功: {model_name}\")
            
        except Exception as e:
            logger.error(f\"❌ 模型切换失败: {e}\")
            raise
    
    def predict(self, 
                image: Union[Image.Image, np.ndarray], 
                mask: Union[Image.Image, np.ndarray],
                custom_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        \"\"\"执行Inpainting预测\"\"\"
        
        if not self.model_loaded:
            raise RuntimeError(\"IOPaint模型未加载\")
        
        # 验证输入
        image, mask = self.validate_inputs(image, mask)
        
        # 获取处理参数
        params = self._get_processing_params(custom_config)
        
        # 智能模型选择
        optimal_model = self._choose_optimal_model(image, mask, params)
        if optimal_model != self.current_model:
            self.switch_model(optimal_model)
        
        try:
            # 处理图像格式
            if isinstance(image, Image.Image):
                image_rgb = np.array(image.convert(\"RGB\"))
            else:
                image_rgb = image
                
            if isinstance(mask, Image.Image):
                mask_gray = np.array(mask.convert(\"L\"))
            else:
                mask_gray = mask
            
            logger.info(f\"🎨 使用{self.current_model}模型处理: {image_rgb.shape}\")
            
            # 构建IOPaint配置
            config = self._build_iopaint_config(params)
            
            # 执行inpainting
            result = self.model_manager(image_rgb, mask_gray, config)
            
            logger.info(f\"✅ {self.current_model}处理完成\")
            return result
            
        except Exception as e:
            logger.error(f\"❌ {self.current_model}处理失败: {e}\")
            raise
    
    def _choose_optimal_model(self, image, mask, params) -> str:
        \"\"\"智能选择最优模型\"\"\"
        
        # 计算mask覆盖率
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask.convert(\"L\"))
        else:
            mask_array = mask
            
        mask_coverage = np.sum(mask_array > 128) / mask_array.size * 100
        
        # 获取图像复杂度（边缘密度）
        image_complexity = self._calculate_image_complexity(image)
        
        # 智能选择策略
        if mask_coverage > 25:
            return 'mat'      # 大水印用MAT
        elif image_complexity > 0.7:
            return 'zits'     # 复杂结构用ZITS  
        elif mask_coverage < 5:
            return 'lama'     # 小水印用LaMA（快速）
        else:
            return 'mat'      # 默认用MAT
    
    def _calculate_image_complexity(self, image) -> float:
        \"\"\"计算图像复杂度\"\"\"
        # 简单的边缘密度计算
        import cv2
        
        if isinstance(image, Image.Image):
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        edges = cv2.Canny(gray, 50, 150)
        complexity = np.sum(edges > 0) / edges.size
        return complexity
    
    def _build_iopaint_config(self, params):
        \"\"\"构建IOPaint配置\"\"\"
        
        strategy_map = {
            'CROP': self.HDStrategy.CROP,
            'RESIZE': self.HDStrategy.RESIZE,
            'ORIGINAL': self.HDStrategy.ORIGINAL
        }
        
        config = self.Config(
            ldm_steps=params.get('ldm_steps', 50),
            ldm_sampler=self.LDMSampler.ddim,
            hd_strategy=strategy_map.get(params.get('hd_strategy', 'CROP')),
            hd_strategy_crop_margin=params.get('hd_strategy_crop_margin', 64),
            hd_strategy_crop_trigger_size=params.get('hd_strategy_crop_trigger_size', 1024),
            hd_strategy_resize_limit=params.get('hd_strategy_resize_limit', 2048),
        )
        
        return config
    
    def _get_processing_params(self, custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        \"\"\"获取处理参数\"\"\"
        
        default_params = {
            'ldm_steps': 50,
            'hd_strategy': 'CROP',
            'hd_strategy_crop_margin': 64,
            'hd_strategy_crop_trigger_size': 1024,
            'hd_strategy_resize_limit': 2048,
        }
        
        if custom_config:
            default_params.update(custom_config)
            
        return default_params
    
    def get_model_info(self) -> Dict[str, Any]:
        \"\"\"获取模型信息\"\"\"
        info = super().get_model_info()
        info.update({
            \"model_type\": \"IOPaint_Unified\",
            \"current_model\": self.current_model,
            \"supported_models\": [\"mat\", \"zits\", \"lama\", \"migan\", \"fcf\"],
            \"intelligent_selection\": True,
            \"framework\": \"IOPaint\"
        })
        return info
```

#### 2.2 更新配置文件
**文件：`config/iopaint_config.yaml`**

```yaml
# IOPaint统一配置文件
app:
  title: \"AI Watermark Remover - IOPaint Edition\"
  host: \"0.0.0.0\"
  port: 8501
  debug: false

# 处理选项
processing:
  default_max_bbox_percent: 10.0
  default_transparent: false
  default_overwrite: false
  default_force_format: \"PNG\"
  supported_formats: [\"jpg\", \"jpeg\", \"png\", \"webp\"]

# Mask生成器配置 (保持不变)
mask_generator:
  model_type: \"custom\"
  mask_model_path: \"./data/models/epoch=071-valid_iou=0.7267.ckpt\"
  image_size: 768
  imagenet_mean: [0.485, 0.456, 0.406]
  imagenet_std: [0.229, 0.224, 0.225]
  mask_threshold: 0.5
  mask_dilate_kernel_size: 3
  mask_dilate_iterations: 1

# IOPaint模型配置
models:
  # 默认inpainting模型
  default_inpainting: \"mat\"
  
  # 支持的模型列表
  available_models:
    - \"zits\"     # 最佳结构保持，适合复杂图像
    - \"mat\"      # 最佳质量，适合大水印
    - \"fcf\"      # 快速修复，平衡性能
    - \"lama\"     # 最快速度，适合小水印
  
  # 智能模型选择
  auto_model_selection: true
  
  # 模型选择策略
  selection_strategy:
    large_watermark_threshold: 25    # 大于25%使用MAT
    complex_image_threshold: 0.7     # 复杂度大于0.7使用ZITS
    small_watermark_threshold: 5     # 小于5%使用LaMA

# IOPaint处理配置
iopaint_config:
  # 高分辨率处理策略
  hd_strategy: \"CROP\"                    # CROP, RESIZE, ORIGINAL
  hd_strategy_crop_margin: 64
  hd_strategy_crop_trigger_size: 1024     # 高清处理优化
  hd_strategy_resize_limit: 2048
  
  # LDM参数 (适用于某些模型)
  ldm_steps: 50
  ldm_sampler: \"ddim\"
  
  # 性能配置
  enable_gpu_optimization: true
  clear_cache_after_processing: true

# 文件处理
files:
  max_upload_size: 20  # MB
  temp_dir: \"./temp\"
  output_dir: \"./data/output\"
  allowed_extensions: [\".jpg\", \".jpeg\", \".png\", \".webp\"]

# UI配置
ui:
  show_model_selector: true
  show_advanced_options: true
  show_auto_selection: true
  default_download_format: \"png\"
  
  # 参数范围
  parameter_ranges:
    ldm_steps: [10, 100]
    crop_trigger_size: [512, 2048]
    crop_margin: [32, 128]

# 性能监控
performance:
  log_processing_time: true
  log_memory_usage: true
  log_model_switches: true

# 日志配置
logging:
  level: \"INFO\"
  format: \"%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s\"
```

### 阶段3：更新核心推理模块

#### 3.1 简化inference.py
**主要修改：**

```python
# 移除PowerPaint导入
# from core.models.powerpaint_processor import PowerPaintProcessor
# from core.models.powerpaint_v2_real_processor import PowerPaintV2RealProcessor

# 添加IOPaint导入
from core.models.iopaint_processor import IOPaintProcessor

class InferenceManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 简化初始化，只需要LaMA和IOPaint处理器
        self.lama_processor = None
        self.iopaint_processor = None  # 新：统一的IOPaint处理器
        
        # 移除PowerPaint相关处理器
        # self.powerpaint_processor = None
        # self.powerpaint_v2_real_processor = None
    
    def _initialize_processors(self):
        \"\"\"初始化处理器 - 简化版\"\"\"
        try:
            # LaMA处理器 (保留作为备选)
            self.lama_processor = LamaProcessor(self.config)
            
            # IOPaint统一处理器 (新主力)
            self.iopaint_processor = IOPaintProcessor(self.config)
            
            logger.info(\"✅ 所有处理器初始化完成\")
            
        except Exception as e:
            logger.error(f\"❌ 处理器初始化失败: {e}\")
            raise
    
    def process_image(self, image, mask=None, custom_config=None):
        \"\"\"处理图像 - 简化版\"\"\"
        
        # 简化的模型选择逻辑
        inpaint_model = custom_config.get('inpaint_model', 'iopaint')
        
        if inpaint_model == 'iopaint':
            # 使用IOPaint统一处理器 (支持MAT/ZITS/LaMA自动选择)
            result = self.iopaint_processor.predict(image, mask, custom_config)
        else:
            # 备选LaMA处理器
            result = self.lama_processor.predict(image, mask, custom_config)
            
        return result
```

### 阶段4：更新Web界面

#### 4.1 简化ui.py
**主要修改：**

```python
# 在模型选择部分
col1, col2 = st.columns(2)
with col1:
    inpaint_model = st.selectbox(
        \"🎨 Inpainting模型\",
        options=[\"iopaint\", \"lama\"],
        index=0,
        help=\"IOPaint支持MAT/ZITS/LaMA智能选择，LaMA适合快速处理\"
    )

# 如果选择IOPaint，显示具体模型选择
if inpaint_model == \"iopaint\":
    with col2:
        specific_model = st.selectbox(
            \"🔧 具体模型\",
            options=[\"auto\", \"mat\", \"zits\", \"lama\", \"migan\"],
            index=0,
            help=\"auto会根据图像自动选择最佳模型\"
        )
        
    # 添加到处理参数
    if specific_model != \"auto\":
        processing_params['inpaint_model'] = specific_model

# 移除所有PowerPaint特定的UI控件
# - PowerPaint参数面板
# - Task选择器  
# - Prompt输入框
# - 复杂的参数调节器
```

---

## 🗂️ 删除清单

### 完全删除的文件 (⚠️ 不可恢复)

```bash
# 核心处理器文件
rm core/models/powerpaint_processor.py
rm core/models/powerpaint_v2_processor.py  
rm core/models/powerpaint_v2_real_processor.py

# PowerPaint模块 (整个目录)
rm -rf powerpaint/

# 模型文件 (约4-6GB)
rm -rf models/powerpaint_v2/

# 脚本文件
rm scripts/download_powerpaint_model.py
rm scripts/test_powerpaint_integration.py

# 高清修复测试文件 (PowerPaint特定)
rm test_high_resolution_fix.py
```

### 需要清理的代码部分

#### 1. core/inference.py
- [ ] 删除PowerPaint导入 (第18-19行)
- [ ] 删除PowerPaint处理器初始化
- [ ] 删除PowerPaint相关处理逻辑
- [ ] 简化模型选择逻辑

#### 2. interfaces/web/ui.py  
- [ ] 删除PowerPaint模型选择选项
- [ ] 删除Task选择器 (object-removal相关)
- [ ] 删除Prompt输入界面
- [ ] 删除复杂参数调节面板
- [ ] 添加IOPaint模型选择器

#### 3. config/powerpaint_config.yaml
- [ ] 重命名为iopaint_config.yaml
- [ ] 删除PowerPaint特定配置
- [ ] 添加IOPaint模型配置

#### 4. config/config.py
- [ ] 删除PowerPaint参数验证逻辑
- [ ] 简化配置管理
- [ ] 添加IOPaint支持

#### 5. core/models/__init__.py
- [ ] 删除PowerPaint处理器导入

#### 6. 测试文件清理
- [ ] test_ui_functionality.py - 删除PowerPaint测试
- [ ] test_integration.py - 删除PowerPaint集成测试  
- [ ] test_full_pipeline.py - 删除PowerPaint pipeline测试
- [ ] scripts/test_webui_flow.py - 删除PowerPaint workflow测试

---

## 📋 执行步骤

### 🔄 Phase 1: 准备阶段 (备份前)
1. [ ] **创建GitHub备份**
   ```bash
   git add .
   git commit -m \"Backup before PowerPaint to IOPaint refactor\"
   git push origin feature/inpainting-model-replacement
   ```

2. [ ] **升级IOPaint**
   ```bash
   /home/duolaameng/miniconda/envs/py310aiwatermark/bin/pip install --upgrade iopaint
   ```

3. [ ] **验证新模型可用性**
   ```bash
   iopaint download --model mat
   iopaint download --model zits  
   iopaint download --model migan
   ```

### 🏗️ Phase 2: 创建新架构
1. [ ] 创建 `core/models/iopaint_processor.py`
2. [ ] 创建 `config/iopaint_config.yaml` 
3. [ ] 测试IOPaint处理器基本功能

### 🔧 Phase 3: 修改现有文件
1. [ ] 更新 `core/inference.py`
2. [ ] 更新 `interfaces/web/ui.py`
3. [ ] 更新 `config/config.py`
4. [ ] 更新 `core/models/__init__.py`

### 🗑️ Phase 4: 删除旧代码  
1. [ ] 删除PowerPaint处理器文件
2. [ ] 删除powerpaint模块目录
3. [ ] 删除模型文件目录 (释放4-6GB空间)
4. [ ] 删除相关脚本文件
5. [ ] 清理测试文件中的PowerPaint代码

### ✅ Phase 5: 测试验证
1. [ ] 功能测试：基本水印去除流程
2. [ ] 模型测试：MAT、ZITS、LaMA切换
3. [ ] 性能测试：处理时间和内存使用
4. [ ] UI测试：界面功能完整性
5. [ ] 集成测试：完整workflow验证

### 📝 Phase 6: 文档更新
1. [ ] 更新README.md
2. [ ] 更新CLAUDE.md  
3. [ ] 创建迁移说明文档
4. [ ] 更新issues_log.md

---

## ⚠️ 风险评估与备份策略

### 🔒 高风险操作
1. **删除模型文件** - 4-6GB数据不可恢复
2. **删除powerpaint模块** - 大量自定义代码  
3. **修改核心推理逻辑** - 影响核心功能

### 🛡️ 风险缓解
1. **完整Git备份** - 代码层面保护
2. **模型文件备份** - 考虑是否需要单独备份大模型文件
3. **分阶段执行** - 每个阶段都验证功能
4. **回滚方案** - 准备快速回滚计划

### 📊 预期收益
- **代码简化**: 约70%代码量减少
- **性能提升**: MAT模型水印去除效果提升30-50%
- **维护性**: 大幅简化架构，便于后续开发
- **扩展性**: 轻松支持更多IOPaint模型

---

## 🎯 成功标准

### ✅ 功能标准
- [ ] 基本水印去除功能正常
- [ ] 自定义mask生成保持不变
- [ ] MAT/ZITS/LaMA模型可正常切换  
- [ ] Web UI界面功能完整
- [ ] 处理性能不降低

### 📈 质量标准  
- [ ] 水印去除效果优于现有PowerPaint
- [ ] 代码结构更简洁清晰
- [ ] 无明显性能回归
- [ ] 所有测试用例通过

### 🔧 技术标准
- [ ] 无导入错误
- [ ] 无运行时异常
- [ ] 日志输出正常
- [ ] 配置文件有效

---

## 🔄 进一步重构建议

### 📁 拆分core/inference.py

当前`core/inference.py`文件仍然较大（633行），建议进一步拆分：

#### 建议的新结构：
```
core/
├── inference/
│   ├── __init__.py
│   ├── mask_generators/
│   │   ├── __init__.py
│   │   ├── custom_mask_generator.py    # CustomMaskGenerator
│   │   ├── florence_mask_generator.py  # FlorenceMaskGenerator
│   │   └── base_mask_generator.py      # 基础mask生成器接口
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── watermark_processor.py      # WatermarkProcessor
│   │   ├── enhanced_processor.py       # EnhancedWatermarkProcessor
│   │   └── base_processor.py           # 基础处理器接口
│   ├── managers/
│   │   ├── __init__.py
│   │   └── inference_manager.py        # InferenceManager
│   └── results/
│       ├── __init__.py
│       └── processing_result.py        # ProcessingResult
```

#### 拆分优先级：
1. **高优先级** - 拆分mask生成器（CustomMaskGenerator, FlorenceMaskGenerator）
2. **中优先级** - 拆分处理器（WatermarkProcessor, EnhancedWatermarkProcessor）
3. **低优先级** - 拆分管理器和结果类

#### 拆分好处：
- 更好的代码组织和维护性
- 更容易进行单元测试
- 更清晰的职责分离
- 更容易扩展新功能

---

## 📞 执行确认

**在开始执行前，请确认：**

1. ✅ 已仔细阅读完整方案
2. ✅ 已完成GitHub代码备份  
3. ✅ 理解删除操作的不可逆性
4. ✅ 准备好足够的时间完成重构
5. ✅ 确认服务器环境稳定

**准备就绪后，按Phase顺序执行即可！**

---

*本方案预计执行时间：2-3小时*  
*预计释放磁盘空间：4-6GB*  
*预计代码量减少：约70%*