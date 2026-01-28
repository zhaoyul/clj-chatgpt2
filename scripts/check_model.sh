#!/bin/bash
# 检查模型文件是否存在

MODEL_FILE="resources/onnx/model.onnx"

if [ -f "$MODEL_FILE" ]; then
    SIZE=$(ls -lh "$MODEL_FILE" | awk '{print $5}')
    echo "✅ 模型文件已存在: $MODEL_FILE ($SIZE)"
    exit 0
else
    echo "❌ 模型文件不存在: $MODEL_FILE"
    echo ""
    echo "请运行以下命令下载模型:"
    echo "  make model"
    echo ""
    echo "或:"
    echo "  python scripts/setup_model.py"
    exit 1
fi
