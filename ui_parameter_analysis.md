# UI参数分析和简化建议

## 当前UI参数完整清单

### 📋 **1. Mask Generation 参数组**
- **mask_model**: "custom" / "upload" (mask生成方式选择)

#### Custom模式参数：
- **mask_threshold**: 0.0-1.0 (默认0.5) - 二值化阈值
- **mask_dilate_kernel_size**: 1-50 (默认3) - 膨胀核大小  
- **mask_dilate_iterations**: 1-20 (默认1) - 膨胀迭代次数
- **detection_prompt**: 选择框 ["watermark", "logo", "text overlay", "signature", "copyright mark"]
- **max_bbox_percent**: 1.0-50.0 (默认10.0) - 最大检测区域百分比
- **confidence_threshold**: 0.1-0.9 (默认0.3) - 检测置信度阈值

#### Upload模式参数：
- **uploaded_mask**: 文件上传组件
- **mask_dilate_kernel_size**: 0-20 (默认5) - 额外膨胀
- **mask_dilate_iterations**: 1-5 (默认2) - 膨胀迭代次数

### 🎨 **2. Inpainting Parameters 参数组**
- **inpaint_model**: "iopaint" / "lama" (主模型选择)

#### IOPaint模式参数：
- **specific_model**: ["auto", "zits", "mat", "fcf", "lama"] - 具体模型选择
- **ldm_steps**: 10-100 (默认50) - 扩散模型步数
- **hd_strategy**: ["CROP", "RESIZE", "ORIGINAL"] - 高分辨率策略

#### HD Strategy相关参数：
- **hd_strategy_crop_margin**: 32-128 (默认64) - 分块边距 (仅CROP)
- **hd_strategy_crop_trigger_size**: 512-2048 (默认1024) - 触发分块的尺寸 (仅CROP)
- **hd_strategy_resize_limit**: 512-2048 (默认2048) - 缩放上限 (仅RESIZE)

#### LaMA模式参数：
- **ldm_steps**: 10-200 (默认50) - 扩散步数
- **ldm_sampler**: ["ddim", "plms"] - 采样器
- **hd_strategy**: ["CROP", "RESIZE", "ORIGINAL"] - 处理策略
- **hd_strategy_crop_margin**: 32-256 (默认64) - 分块边距
- **hd_strategy_crop_trigger_size**: 512-2048 (默认800) - 触发尺寸
- **hd_strategy_resize_limit**: 512-2048 (默认1600) - 缩放上限

#### 通用参数：
- **seed**: -1 to 999999 (默认-1) - 随机种子

### ⚡ **3. Performance Options 参数组**
- **mixed_precision**: bool (默认True) - 混合精度计算
- **log_processing_time**: bool (默认True) - 记录处理时间

### 🔧 **4. Processing Mode 参数组**  
- **transparent**: bool (默认False) - 透明模式 (已移除)

---

## 🎯 **简化建议（符合SIMP-LAMA原则）**

### ✅ **建议保留的核心参数（5-6个）**

#### **必保参数（用户必须控制）**
1. **模型选择**: "mat" / "zits" / "fcf" / "lama" 
   - 这是核心功能，用户需要选择不同模型
   - 直接暴露4个具体模型，移除"auto"和复杂的嵌套选择

2. **Mask方式**: "custom" / "upload"
   - 核心功能，用户需要选择mask生成方式

#### **重要参数（影响质量）**
3. **推理步数**: 20-100 (默认50)
   - 直接影响质量，用户可能需要调整
   - 统一所有模型的步数参数

4. **HD策略**: ["CROP", "ORIGINAL"] 
   - 简化为2个选项，移除RESIZE（较少使用）
   - 大图片选CROP，小图片选ORIGINAL

#### **可选保留参数**
5. **Mask阈值**: 0.3-0.8 (默认0.5) - 仅Custom模式
   - 用户经常需要调整的mask质量参数

6. **随机种子**: -1 to 999999 (默认-1)  
   - 用于结果重现，调试时有用

### ❌ **建议隐藏的技术参数（8-10个）**

#### **自动化/默认值处理**
- **所有膨胀参数**: 设为智能默认值（根据图片大小自适应）
- **所有HD策略详细参数**: 使用模型优化的默认值
- **采样器选择**: 每个模型使用最佳默认采样器
- **性能选项**: 默认开启，用户无需调整
- **置信度阈值**: 设为经验最佳值0.3
- **检测提示词**: 默认"watermark"
- **最大检测区域**: 默认10%

#### **技术细节参数**
- **crop_margin, crop_trigger_size, resize_limit**: 每个模型使用优化默认值
- **mixed_precision, log_processing_time**: 默认开启
- **detection_prompt选择**: 默认"watermark"
- **max_bbox_percent, confidence_threshold**: 使用经验最佳值

---

## 🚀 **简化后的UI设计**

### **核心参数面板（6个参数）**
```
🎯 Watermark Removal Settings
├── Model: [MAT] [ZITS] [FCF] [LaMA]  
├── Mask Source: [Smart Detection] [Upload File]
├── Quality: [Fast(20)] ──────●──── [Best(80)] (Steps: 50)
├── Large Image: [Original] [Smart Crop] 
└── Advanced:
    ├── Mask Sensitivity: 0.3 ──●── 0.8 (0.5) [仅Smart Detection时显示]
    └── Seed: [-1] (Random)
```

### **隐藏的智能默认值**
- 每个模型使用优化的预设配置
- 根据图片尺寸自动调整处理策略  
- 根据mask覆盖率自动调整膨胀参数
- 性能优化自动开启

### **预期效果**
- **参数数量**: 从15+个减少到5-6个核心参数
- **用户体验**: 简单易用，专注核心功能
- **专业用户**: 可通过配置文件调整高级参数
- **初学者**: 开箱即用，智能默认值

---

## 📝 **实施建议**

1. **保留当前完整UI作为"Advanced Mode"**
2. **创建新的"Simple Mode"作为默认界面** 
3. **在设置中提供模式切换选项**
4. **为每个模型预设最佳配置文件**
5. **实现参数的智能默认值算法**

你觉得这个简化方案如何？需要调整哪些参数的保留或隐藏决策？