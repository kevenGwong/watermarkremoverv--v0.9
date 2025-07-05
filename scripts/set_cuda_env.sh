#!/bin/bash
# CUDA内存管理优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

# 启动应用
cd "$(dirname "$0")/.."
streamlit run interfaces/web/main.py --server.port 8501
