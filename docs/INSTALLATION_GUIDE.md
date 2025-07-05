# WatermarkRemover-AI 安装指南

## 快速安装

### 1. 基础环境
```bash
# 克隆项目
git clone <repository_url>
cd WatermarkRemover-AI

# 创建conda环境
conda create -n py310aiwatermark python=3.10
conda activate py310aiwatermark

# 安装基础依赖
pip install -r requirements.txt
```

### 2. 模型文件
项目已包含预训练模型文件，无需额外下载。

### 3. 可选：LaMA支持
如果需要使用LaMA模型（快速处理），安装额外依赖：
```bash
# 方法1: pip安装
pip install saicinpainting

# 方法2: conda安装
conda install -c conda-forge saicinpainting
```

## 启动应用

### 使用优化脚本（推荐）
```bash
# 使用CUDA内存优化脚本
./scripts/set_cuda_env.sh
```

### 手动启动
```bash
# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# 启动应用
streamlit run interfaces/web/main.py --server.port 8501
```

## 故障排除

### CUDA内存不足
1. 降低图像分辨率
2. 使用更小的模型（ZITS → FCF → LaMA）
3. 重启应用清理显存

### LaMA模型不可用
- 检查saicinpainting是否正确安装
- 使用其他可用模型（ZITS/MAT/FCF）

### 颜色异常
- 已修复颜色空间问题
- 如仍有问题，请报告issue

## 性能优化

### 显存管理
- 使用提供的CUDA环境脚本
- 避免同时加载多个大模型
- 定期重启应用清理显存

### 处理速度
- LaMA: 最快（2-5秒）
- FCF: 快速（8-20秒）
- ZITS: 中等（10-30秒）
- MAT: 最慢但质量最好（15-45秒）

## 支持

如遇问题，请：
1. 查看日志输出
2. 检查显存使用情况
3. 提交issue并附上错误信息
