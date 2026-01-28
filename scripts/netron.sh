#!/bin/bash
# Netron 模型可视化启动脚本

PORT=${1:-8080}
MODEL_FILE="resources/onnx/model.onnx"

echo "🌐 启动 Netron 模型可视化服务器..."
echo "   Model: $MODEL_FILE"
echo "   Port: $PORT"
echo ""

# 检查模型文件是否存在
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_FILE"
    echo "   请先运行: python scripts/export_model.py"
    exit 1
fi

# 检查 netron 是否安装
if ! command -v netron &> /dev/null; then
    echo "⚠️  Netron 未安装，正在安装..."
    pip install netron
fi

echo "✅ 启动服务器..."
echo "   打开浏览器访问: http://localhost:$PORT"
echo ""

python3 -m netron "$MODEL_FILE" --port "$PORT"
