# 🔧 Advanced Parameter Guide
## AI Watermark Remover Pro - 高级参数详解

本文档详细说明AI Watermark Remover Pro版本中所有可自定义参数的用途、取值范围和使用建议。

---

## 📋 目录

1. [🎯 Mask Generation Parameters](#mask-generation-parameters)
2. [🎨 LaMA Inpainting Parameters](#lama-inpainting-parameters)
3. [🖼️ Image Processing Parameters](#image-processing-parameters)
4. [🔧 Post Processing Parameters](#post-processing-parameters)
5. [⚡ Performance Presets](#performance-presets)
6. [💡 Usage Tips](#usage-tips)

---

## 🎯 Mask Generation Parameters

### Custom Model Parameters

#### `mask_threshold` (0.0 - 1.0)
**用途**: 控制自定义分割模型的二值化阈值
- **默认值**: 0.5
- **低值 (0.1-0.4)**: 更敏感，检测更多区域（可能包含假阳性）
- **中值 (0.4-0.6)**: 平衡准确性和覆盖率
- **高值 (0.6-0.9)**: 更保守，只检测高置信度区域（可能遗漏部分水印）

**使用建议**:
- 清晰水印: 0.5-0.7
- 模糊水印: 0.3-0.5
- 复杂背景: 0.6-0.8

#### `mask_dilate_kernel_size` (1 - 15, 奇数)
**用途**: 控制形态学膨胀操作的核大小，扩展检测区域
- **默认值**: 3
- **小值 (1-3)**: 精确边界，适合清晰边界的水印
- **中值 (5-7)**: 适合一般情况
- **大值 (9-15)**: 大幅扩展，适合模糊或需要完全覆盖的情况

**使用建议**:
- 文字水印: 3-5
- 图形水印: 5-7
- 复杂形状: 7-11

#### `mask_dilate_iterations` (1 - 5)
**用途**: 膨胀操作的迭代次数
- **默认值**: 1
- **1次**: 轻微扩展
- **2-3次**: 适度扩展
- **4-5次**: 显著扩展覆盖范围

### Florence-2 Parameters

#### `detection_prompt` (字符串)
**用途**: 指定Florence-2检测的目标类型
- **默认值**: "watermark"
- **预设选项**:
  - `"watermark"`: 通用水印检测
  - `"logo"`: 标志/徽标检测
  - `"text overlay"`: 文字叠加检测
  - `"signature"`: 签名检测
  - `"copyright mark"`: 版权标识
  - `"brand mark"`: 品牌标识

**自定义Prompt格式**:
- 简单描述: `"company logo"`
- 具体描述: `"red circular watermark"`
- 多目标: `"watermark or logo"`

**使用建议**:
- 根据实际水印类型选择最匹配的描述
- 可以使用更具体的描述来提高准确性
- 避免过于复杂的描述

#### `max_bbox_percent` (1.0 - 50.0)
**用途**: 限制检测边界框占图像的最大百分比
- **默认值**: 10.0%
- **低值 (1-5%)**: 只检测小型水印
- **中值 (10-20%)**: 适合一般水印
- **高值 (30-50%)**: 允许检测大型水印或多个水印

#### `confidence_threshold` (0.1 - 0.9)
**用途**: Florence-2检测的置信度阈值
- **默认值**: 0.3
- **低值 (0.1-0.3)**: 更激进，可能检测到更多目标
- **高值 (0.5-0.9)**: 更保守，只检测高置信度目标

---

## 🎨 LaMA Inpainting Parameters

### Core Processing Parameters

#### `ldm_steps` (10 - 200)
**用途**: 扩散模型的去噪步数，直接影响修复质量和处理时间
- **默认值**: 50
- **快速 (10-30)**: 适合预览或批量处理
- **平衡 (40-80)**: 推荐日常使用
- **高质量 (80-150)**: 适合重要图片
- **极致 (150-200)**: 专业需求，处理时间较长

**质量vs速度对比**:
- 20 steps: ~10秒，基础质量
- 50 steps: ~25秒，良好质量
- 100 steps: ~50秒，高质量
- 200 steps: ~100秒，极致质量

#### `ldm_sampler` (字符串)
**用途**: 选择扩散模型的采样算法
- **默认值**: "ddim"
- **选项**:
  - `"ddim"`: 确定性采样，结果稳定一致
  - `"plms"`: 更快的收敛速度
  - `"dpm_solver++"`: 高质量采样，适合专业需求

**使用建议**:
- 稳定性优先: ddim
- 速度优先: plms
- 质量优先: dpm_solver++

### High Resolution Strategy

#### `hd_strategy` (字符串)
**用途**: 处理高分辨率图像的策略
- **默认值**: "CROP"
- **选项**:
  - `"CROP"`: 分块处理，保持最高细节
  - `"RESIZE"`: 缩放处理，速度更快
  - `"ORIGINAL"`: 原尺寸处理（需要大量内存）

**使用建议**:
- 高质量需求: CROP
- 速度优先: RESIZE
- 小图像(<1024px): ORIGINAL

#### `hd_strategy_crop_margin` (32 - 256)
**用途**: 分块处理时的重叠边距
- **默认值**: 64
- **小值 (32-48)**: 较少重叠，可能有边界痕迹
- **大值 (128-256)**: 更多重叠，边界更自然

#### `hd_strategy_crop_trigger_size` (512 - 2048)
**用途**: 触发分块处理的图像尺寸阈值
- **默认值**: 800
- **较小值**: 更早触发分块，适合内存受限环境
- **较大值**: 更晚触发分块，适合高内存环境

#### `hd_strategy_resize_limit` (1024 - 4096)
**用途**: RESIZE策略下的最大处理尺寸
- **默认值**: 1600
- **1024**: 快速处理
- **2048**: 平衡质量和速度
- **4096**: 最高质量

---

## 🖼️ Image Processing Parameters

### Preprocessing Parameters

#### `max_input_size` (512 - 4096)
**用途**: 输入图像的最大尺寸限制
- **默认值**: 2048
- **用途**: 控制处理复杂度和内存使用

#### `gamma_correction` (0.5 - 2.0)
**用途**: Gamma校正，调整图像亮度曲线
- **默认值**: 1.0 (无校正)
- **<1.0**: 增强暗部细节
- **>1.0**: 增强亮部细节

#### `contrast_enhancement` (0.5 - 2.0)
**用途**: 对比度增强
- **默认值**: 1.0 (无增强)
- **>1.0**: 增强对比度
- **<1.0**: 降低对比度

---

## 🔧 Post Processing Parameters

### Mask Refinement

#### `mask_blur_radius` (0 - 10)
**用途**: 对生成的mask进行高斯模糊
- **默认值**: 0 (无模糊)
- **用途**: 软化mask边界，创建更自然的过渡

#### `mask_feather_size` (0 - 20)
**用途**: mask羽化大小，创建渐变边界
- **默认值**: 0 (无羽化)
- **用途**: 使修复区域边界更自然

#### `mask_erosion_size` (-10 到 10)
**用途**: 形态学腐蚀/膨胀操作
- **默认值**: 0
- **负值**: 腐蚀，收缩mask区域
- **正值**: 膨胀，扩展mask区域

### Result Enhancement

#### `output_sharpening` (0.0 - 2.0)
**用途**: 输出结果锐化
- **默认值**: 0.0 (无锐化)
- **适度 (0.1-0.5)**: 增强细节
- **过度 (>1.0)**: 可能产生伪影

#### `output_denoising` (0.0 - 1.0)
**用途**: 输出结果降噪
- **默认值**: 0.0 (无降噪)
- **轻微 (0.1-0.3)**: 减少细微噪点
- **强烈 (>0.5)**: 可能损失细节

---

## ⚡ Performance Presets

### Fast Preset - 快速处理
```yaml
ldm_steps: 20
hd_strategy_resize_limit: 1024
mask_dilate_kernel_size: 1
```
- **用途**: 快速预览、批量处理
- **处理时间**: ~10-15秒
- **质量**: 基础

### Balanced Preset - 平衡处理
```yaml
ldm_steps: 50
hd_strategy_resize_limit: 1600
mask_dilate_kernel_size: 3
```
- **用途**: 日常使用推荐设置
- **处理时间**: ~25-35秒
- **质量**: 良好

### Quality Preset - 高质量处理
```yaml
ldm_steps: 100
hd_strategy_resize_limit: 2048
mask_dilate_kernel_size: 5
```
- **用途**: 重要图片、专业需求
- **处理时间**: ~50-70秒
- **质量**: 高

### Ultra Preset - 极致质量
```yaml
ldm_steps: 200
hd_strategy_resize_limit: 4096
mask_dilate_kernel_size: 7
```
- **用途**: 专业需求、极致质量
- **处理时间**: ~100-150秒
- **质量**: 极致

---

## 💡 Usage Tips

### 🎯 针对不同水印类型的建议设置

#### 文字水印
```yaml
mask_threshold: 0.6
mask_dilate_kernel_size: 3
detection_prompt: "text overlay"
ldm_steps: 50
```

#### 透明LOGO
```yaml
mask_threshold: 0.4
mask_dilate_kernel_size: 5
detection_prompt: "logo"
mask_blur_radius: 1
```

#### 复杂图形水印
```yaml
mask_threshold: 0.5
mask_dilate_kernel_size: 7
detection_prompt: "watermark"
ldm_steps: 100
hd_strategy: "CROP"
```

#### 多个小水印
```yaml
max_bbox_percent: 5.0
confidence_threshold: 0.2
mask_dilate_kernel_size: 3
```

### 🚀 性能优化建议

#### 内存受限环境
```yaml
max_input_size: 1024
hd_strategy: "RESIZE"
hd_strategy_resize_limit: 1024
ldm_steps: 30
```

#### 高性能环境
```yaml
max_input_size: 4096
hd_strategy: "CROP"
hd_strategy_resize_limit: 4096
ldm_steps: 100
```

### 🎨 质量优化建议

#### 边界优化
```yaml
mask_blur_radius: 1
mask_feather_size: 2
output_sharpening: 0.2
```

#### 细节保护
```yaml
hd_strategy: "CROP"
hd_strategy_crop_margin: 128
output_sharpening: 0.1
output_denoising: 0.1
```

---

## 🔧 故障排除

### 常见问题及解决方案

1. **检测不到水印**
   - 降低 `mask_threshold`
   - 调整 `detection_prompt`
   - 增加 `max_bbox_percent`

2. **检测范围过大**
   - 提高 `mask_threshold`
   - 减小 `mask_dilate_kernel_size`
   - 降低 `max_bbox_percent`

3. **修复效果不自然**
   - 增加 `ldm_steps`
   - 使用 `mask_blur_radius`
   - 调整 `hd_strategy`

4. **处理速度太慢**
   - 使用Fast预设
   - 降低 `max_input_size`
   - 选择 "RESIZE" 策略

5. **内存不足**
   - 降低 `max_input_size`
   - 使用 "RESIZE" 策略
   - 减少 `ldm_steps`

---

**提示**: 建议从预设开始，然后根据具体需求微调参数。使用Debug模式可以查看中间结果，帮助理解参数效果。