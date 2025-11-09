#!/bin/bash

# 确保脚本在出错时停止执行
set -e

echo "Starting small text modeling project..."

# 设置环境变量
export PYTHONPATH="$PYTHONPATH:/data/wzy/zy/small_text_modeling"

# 检查Python环境
echo "Checking Python version..."
python3 --version

# 安装依赖（如果需要）
echo "Installing dependencies..."
pip install -r /data/wzy/zy/small_text_modeling/requirements.txt

# 运行训练脚本
echo "Starting training..."
python3 /data/wzy/zy/small_text_modeling/src/train.py

echo "Training completed successfully!"