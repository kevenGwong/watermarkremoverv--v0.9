# 🔬 AI Watermark Remover - Debug UI 使用指南

## 🎯 界面布局

### 📊 左侧参数面板
完整的参数控制，分为以下几个部分：

#### 🎯 Mask Generation (Mask生成)
**模型选择下拉菜单：**
- **Custom Watermark**: 专门训练的FPN+MIT-B5模型
- **Florence-2**: Microsoft多模态检测模型

**Custom模型参数：**
- `mask_threshold` (0.0-1.0): 二值化阈值，控制检测敏感度
- `mask_dilate_kernel_size` (1-50): 膨胀核大小，扩展检测区域  
- `mask_dilate_iterations` (1-20): 膨胀迭代次数

**Florence-2参数：**
- `detection_prompt`: 检测目标描述词
- `max_bbox_percent` (1.0-50.0): 最大检测区域百分比
- `confidence_threshold` (0.1-0.9): 检测置信度阈值

#### 🎨 Inpainting Parameters (修复参数)
- `prompt`: 文本提示词 (当前LaMA不支持，保留供未来使用)
- `ldm_steps` (10-200): 扩散模型步数
- `ldm_sampler`: 采样器选择 (ddim/plms)
- `hd_strategy`: 高分辨率处理策略
- `hd_strategy_crop_margin`: 分块处理边距
- `seed`: 随机种子 (-1为随机)

#### ⚡ Performance Options (性能选项)
- `mixed_precision`: 混合精度计算
- `log_processing_time`: 记录处理时间

### 🔄 右侧对比区域
- **Interactive Slider**: 左右滑动对比处理前后
- **Real-time Comparison**: 使用streamlit-image-comparison组件
- **Background Options**: 透明图像的背景选择
- **Mask Visualization**: 显示生成的水印mask

## 🚀 启动方式

### 环境准备
```bash
conda activate py310aiwatermark
```

### 启动Debug UI
```bash
# 直接启动
./run_debug_app.sh

# 或测试后端
./run_debug_app.sh test

# 访问地址
http://localhost:8506
```

## 🧪 参数调试建议

### 🎯 Mask参数调试

#### Custom模型调试顺序：
1. **起始设置**: threshold=0.5, kernel_size=3, iterations=1
2. **检测不足**: 降低threshold到0.3-0.4
3. **误检过多**: 提高threshold到0.6-0.7
4. **边界不清**: 调整kernel_size (3-15)
5. **覆盖不足**: 增加iterations (2-5)
6. **大面积扩展**: 使用更大的kernel_size (15-50) 和 iterations (5-20)

#### Florence-2调试顺序：
1. **选择prompt**: 根据水印类型选择合适描述词
2. **调整bbox**: 从10%开始，根据水印大小调整
3. **置信度**: 从0.3开始，检测不足时降低

### 🎨 Inpainting参数调试

#### 质量优先：
- ldm_steps: 100-200
- hd_strategy: CROP
- sampler: ddim

#### 速度优先：
- ldm_steps: 20-50  
- hd_strategy: RESIZE
- sampler: plms

## 📊 实际测试结果

基于参数效果测试，我们得到以下性能数据：

### ⏱️ 处理时间
- **256x256图片**: ~0.45秒
- **512x512图片**: ~0.93秒
- **透明模式**: ~0.96秒
- **修复模式**: ~1.31秒

### 🎯 检测效果
- **低阈值 (0.1-0.3)**: 敏感检测，可能误检
- **中阈值 (0.5)**: 平衡检测，推荐起始值
- **高阈值 (0.7-0.9)**: 保守检测，可能遗漏

### 📦 区域控制
- **1-5%**: 只检测小型水印
- **10%**: 标准设置，适合多数情况
- **25-50%**: 允许大型水印检测

## 💡 调试工作流程

### 1. 🔍 分析原图
- 观察水印类型（文字/图形/透明）
- 估计水印大小占比
- 确定水印边界清晰度

### 2. 🎯 选择模型
- **规则水印**: Custom模型
- **文字水印**: Florence-2模型
- **复杂水印**: 两个模型都试试

### 3. 📊 调整参数
- 从默认参数开始
- 观察mask生成效果
- 根据实际需要微调

### 4. 🔄 对比结果
- 使用交互式slider对比
- 检查透明区域准确性
- 验证修复质量

### 5. 💾 保存设置
- 找到最佳参数组合
- 下载处理结果
- 记录参数设置供后续使用

## 📁 输出文件

Debug模式会生成以下文件：
- `{filename}_debug.png/webp/jpg`: 处理结果
- 实时mask可视化
- 参数设置摘要

## ⚠️ 注意事项

1. **Prompt参数**: 当前LaMA模型不支持文本提示，参数保留供未来使用
2. **性能影响**: 高质量设置会显著增加处理时间
3. **内存使用**: 大图片和高质量设置需要更多GPU内存
4. **种子控制**: 固定种子可以获得可重复的结果

## 🔧 故障排除

### 常见问题：
1. **检测不到水印**: 降低threshold或调整prompt
2. **误检过多**: 提高threshold或调整bbox_percent  
3. **边界不清**: 调整dilate参数
4. **处理太慢**: 降低ldm_steps或使用RESIZE策略
5. **透明效果不对**: 检查mask生成是否准确

### 解决方案：
- 使用参数效果测试脚本验证
- 对比不同参数设置的结果
- 参考测试报告中的建议值

---

**🎉 现在你可以精确调试水印去除效果了！**

使用这个Debug UI，你可以：
- 📊 实时调整所有参数
- 🔄 即时查看处理效果对比
- 🎯 精确控制检测和修复质量
- 💾 找到最佳参数组合