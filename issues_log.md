# Issues Log

## 2025-01-04 PowerPaint水印去除问题诊断与修复

### 🚨 问题描述
用户反馈Web UI选择PowerPaint模型后，图片根本没有做remove操作，疑似：
1. Mask没有被正确加载到模型中
2. 载入到模型的mask或者原图尺寸不对
3. 调用模型失败回滚到LaMA模型

### 🔍 问题排查过程

#### 发现的根本原因
1. **PowerPaint依赖缺失** - 导致静默回滚到LaMA
2. **模型路径错误** - 硬编码错误路径
3. **致命的Mask逻辑颠倒** - PowerPaint V2 processor中mask composite逻辑错误
4. **Mask处理问题** - resize方法导致边缘模糊
5. **缺少mask验证** - 可能传入空mask

#### 具体问题细节

##### 1. Mask逻辑颠倒 (powerpaint_v2_processor.py:237-241)
```python
# 错误的实现
masked_image = Image.composite(
    Image.new("RGB", (w, h), hole_value),  # 黑色背景
    image,                                 # 原图
    mask.convert("L")                      # mask作为alpha
)
```
**结果**: 白色mask区域保持原图，黑色区域变黑！完全相反！

##### 2. Mask resize方法错误 (powerpaint_processor.py:375)
```python
resized_mask = mask.resize((new_width, new_height), Image.LANCZOS)
```
**问题**: LANCZOS会模糊二值mask边缘

##### 3. 模型路径配置错误
- **错误路径**: `./models/powerpaint_v2/Realistic_Vision_V1.4-inpainting`
- **实际路径**: `./models/powerpaint_v2_real/realisticVisionV60B1_v51VAE`

##### 4. 静默回滚机制 (core/inference.py:486-493)
```python
if inpaint_model == 'powerpaint' and self.powerpaint_processor and self.powerpaint_processor.model_loaded:
    # PowerPaint处理
else:
    # 这里是回滚到LaMA
```

##### 5. PowerPaint依赖缺失
```python
from powerpaint.models import BrushNetModel, UNet2DConditionModel
from powerpaint.pipelines import StableDiffusionPowerPaintBrushNetPipeline
```
这些包在系统中不存在

### ✅ 修复方案

#### 修复1: 修正Mask逻辑颠倒
- **文件**: `core/models/powerpaint_v2_processor.py:235-255`
- **修复**: 反转mask逻辑，确保白色区域被正确标记为需要inpaint
- **影响**: 解决图片"看起来没有处理"的根本问题

#### 修复2: 统一Mask Resize方法
- **文件**: `core/models/powerpaint_processor.py`, `powerpaint_v2_processor.py`
- **修复**: 图像用LANCZOS，mask统一用NEAREST保持锐利边缘
- **影响**: 保持mask的精确性

#### 修复3: 添加Mask有效性验证
- **文件**: 两个processor文件
- **修复**: 检查mask是否有白色像素，覆盖率警告
- **影响**: 防止空mask导致无效果

#### 修复4: 修正模型路径
- **文件**: `core/inference.py`, 两个processor文件
- **修复**: 更新为正确的模型路径
- **影响**: 确保能找到模型文件

#### 修复5: 改进参数传递
- **文件**: 两个processor文件
- **修复**: 确保UI参数优先级最高，不被默认值覆盖
- **影响**: UI设置的prompt正确生效

### 🎯 重要发现

**不一定需要PowerPaint依赖！**
- 当前实现已使用标准SD1.5 inpainting
- 包含针对object removal的优化prompt
- 修复后应该已经有很好效果

### 📊 解决方案优先级

1. **立即生效** - 已修复的mask逻辑和参数问题
2. **可选** - 安装PowerPaint依赖获得更精确控制
3. **备选** - 继续使用标准SD1.5，效果可能已经足够

### ⏰ 时间记录
- **问题发现**: 2025-01-04 下午
- **问题分析**: 2025-01-04 下午 2-3小时
- **修复完成**: 2025-01-04 下午
- **状态**: 待测试

### 🔮 后续步骤
1. 测试修复后的效果
2. 如效果满意，无需额外依赖
3. 如需进一步优化，考虑安装PowerPaint