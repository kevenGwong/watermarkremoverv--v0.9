
core/models/lama_processor_unified.py (335行) - LaMA处理器过于复杂
请帮我创建一个简化的LaMA处理器，要求如下：
LaMA模块改造方案
目标
将LaMA处理器从335行复杂代码简化为20行左右，与其他模型保持一致，强制使用IOPaint内置方法。
改造策略
继承IOPaintBaseProcessor - 获得所有IOPaint功能
删除重复代码 - 移除手动实现的高分辨率处理
统一接口 - 使用标准化的predict方法
简化配置 - 移除复杂的依赖检测逻辑.

1. 继承IOPaintBaseProcessor基类
2. 使用IOPaint内置的LaMA实现
3. 代码行数控制在20行以内
4. 与其他模型（MAT/ZITS/FCF）保持一致的接口
5. 移除所有手动实现的高分辨率处理逻辑
6. 移除复杂的依赖检测和双重架构
7. 使用标准的predict方法，不需要重写

文件路径：core/models/lama_processor_simplified.py

参考其他模型的实现：
- core/models/mat_processor.py
- core/models/zits_processor.py  
- core/models/fcf_processor.py

确保新实现能够：
- 正确处理高分辨率图像（通过IOPaint内置方法）
- 支持所有HD策略（CROP/RESIZE/ORIGINAL）
- 自动处理颜色空间转换
- 与现有架构完全兼容
请帮我更新模型注册，将新的简化LaMA处理器集成到系统中：

1. 更新core/models/__init__.py文件
2. 导入新的SimplifiedLamaProcessor
3. 注册到ModelRegistry
4. 确保向后兼容性
5. 更新相关的导入语句

同时需要：
- 检查是否有其他文件引用了旧的lama_processor_unified.py
- 更新这些引用到新的简化版本
- 确保测试文件能够正常工作

请帮我清理core/utils/image_utils.py文件，移除LaMA特定的处理逻辑：

1. 删除prepare_arrays_for_lama方法
2. 删除postprocess_lama_result方法
3. 更新prepare_arrays_for_iopaint方法，使其适用于所有IOPaint模型（包括LaMA）
4. 确保颜色空间处理统一化
5. 移除不必要的BGR/RGB转换逻辑

目标：
- 简化代码结构
- 统一处理流程
- 减少维护复杂度
- 保持功能完整性

请确保清理后的代码能够：
- 正确处理所有IOPaint模型（MAT/ZITS/FCF/LaMA）
- 自动处理颜色空间转换
- 保持与现有接口的兼容性
请帮我创建测试脚本来验证简化后的LaMA处理器：

1. 测试基本功能：
   - 模型加载
   - 图像处理
   - 高分辨率处理
   - 不同HD策略

2. 测试兼容性：
   - 与现有接口的兼容性
   - 与其他模型的接口一致性
   - 配置参数的兼容性

3. 测试错误处理：
   - 无效输入
   - 模型加载失败
   - 资源清理

请创建完整的测试脚本，包括：
- 单元测试
- 集成测试
- 性能测试
- 错误处理测试

测试文件路径：tests/test_simplified_lama.py
 实施检查清单
代码改造
[ ] 创建SimplifiedLamaProcessor类
[ ] 继承IOPaintBaseProcessor
[ ] 移除手动高分辨率处理
[ ] 移除颜色空间转换逻辑
[ ] 移除依赖检测代码
[ ] 注册到ModelRegistry
文件更新
[ ] 更新core/models/__init__.py
[ ] 清理core/utils/image_utils.py
[ ] 备份旧实现到backup_complex_features/
[ ] 更新相关导入语句
测试验证
[ ] 创建测试脚本
[ ] 验证基本功能
[ ] 验证高分辨率处理
[ ] 验证HD策略
[ ] 验证错误处理
文档更新
[ ] 更新README文档
[ ] 更新架构说明
[ ] 更新API文档
[ ] 记录改造过程
�� 预期效果
代码简化
行数减少: 335行 → 20行 (减少94%)
复杂度降低: 移除双重架构和手动实现
维护成本: 大幅降低
功能保持
高分辨率处理: 通过IOPaint内置方法
HD策略支持: CROP/RESIZE/ORIGINAL
颜色空间处理: 自动处理
性能表现: 与IOPaint其他模型一致
架构统一
接口一致性: 与其他模型完全一致
基类继承: 使用统一的IOPaintBaseProcessor
配置统一: 使用相同的配置参数
错误处理: 统一的错误处理机制
⚠️ 注意事项
备份重要: 在改造前备份现有实现
逐步测试: 每个步骤都要进行充分测试
向后兼容: 确保现有代码不会受到影响
性能验证: 确保简化后性能没有明显下降
文档同步: 及时更新相关文档
这个改造方案将让LaMA处理器变得与其他模型一样简洁，同时保持完整的功能和性能！

*** 关键点：
IOPaint库内部实现：所有高分辨率处理逻辑都在IOPaint库内部
自动处理：model_manager(image_array, mask_array, iopaint_config) 自动根据配置处理
无需手动实现：MAT/ZITS/FCF处理器只需要传递配置参数

**总结对比
模型类型	高分辨率处理	实现复杂度	原因
MAT/ZITS/FCF	IOPaint库内置	简单（20行）	继承IOPaintBaseProcessor，所有复杂逻辑由IOPaint库处理
LaMA	手动实现	复杂（335行）	原生saicinpainting，需要自己实现所有高分辨率处理逻辑