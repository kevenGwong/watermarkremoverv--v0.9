# 📊 Parameter Ranges Update

## 🔄 Updated Ranges

### ⬆️ Increased Upper Limits

| Parameter | Old Range | New Range | Change |
|-----------|-----------|-----------|---------|
| `mask_dilate_kernel_size` | 1-15 | **1-50** | +35 (233% increase) |
| `mask_dilate_iterations` | 1-5 | **1-20** | +15 (400% increase) |

## 🎯 Use Cases for Extended Ranges

### 🔍 Dilate Kernel Size (1-50)

**Small Values (1-15)**: 标准使用
- 1-3: 精确边界，适合清晰水印
- 5-7: 适合一般情况
- 9-15: 中等扩展

**Medium Values (15-25)**: 中等扩展
- 15-20: 模糊边界水印
- 20-25: 需要较大覆盖的水印

**Large Values (25-50)**: 大面积扩展
- 25-35: 大型复杂水印
- 35-50: 极大扩展，用于特殊情况

### 🔄 Dilate Iterations (1-20)

**Low Iterations (1-5)**: 标准使用
- 1-2: 轻微扩展
- 3-5: 适度扩展

**Medium Iterations (5-10)**: 中等扩展
- 5-7: 较大扩展
- 8-10: 显著扩展

**High Iterations (10-20)**: 强烈扩展
- 10-15: 大幅扩展
- 15-20: 极大扩展，用于特殊复杂情况

## ⚠️ 注意事项

### 性能影响
- **Kernel Size > 25**: 处理时间显著增加
- **Iterations > 10**: 内存使用量增加
- **组合使用**: kernel_size=50 + iterations=20 可能导致极长处理时间

### 质量影响
- **过度膨胀**: 可能影响周围正常内容
- **边界模糊**: 大参数值可能导致不自然的边界
- **细节丢失**: 极大扩展可能覆盖重要细节

## 💡 建议使用策略

### 🎯 渐进调整
```
起始值: kernel_size=3, iterations=1
↓
中等调整: kernel_size=7-15, iterations=2-5  
↓
大幅调整: kernel_size=15-25, iterations=5-10
↓
极值调整: kernel_size=25-50, iterations=10-20
```

### 🔍 特殊情况使用

**复杂透明水印:**
- kernel_size: 15-25
- iterations: 3-7

**大面积模糊水印:**
- kernel_size: 25-35  
- iterations: 5-10

**极端边界修复:**
- kernel_size: 35-50
- iterations: 10-20

## 🧪 测试建议

1. **从默认值开始**: kernel_size=3, iterations=1
2. **逐步增加**: 观察mask覆盖效果
3. **找到平衡点**: 覆盖完整但不过度
4. **验证边界**: 检查是否影响正常内容
5. **性能检查**: 确保处理时间可接受

## 📊 预期效果

### 扩展能力
- **覆盖范围**: 可以处理更大、更复杂的水印
- **边界处理**: 更好地处理模糊或渐变边界
- **复杂形状**: 适应不规则水印形状

### 应用场景
- **艺术水印**: 复杂设计的装饰性水印
- **企业LOGO**: 大型品牌标识
- **全图水印**: 覆盖大面积的保护性水印
- **多层水印**: 重叠或嵌套的复合水印

---

**🎯 现在你可以处理更复杂的水印了！**

使用这些扩展的参数范围，可以应对各种复杂的水印清除挑战。