#!/bin/bash
# Clerk Notebook 启动脚本

echo "🚀 启动 Clerk Notebook 服务器..."
echo ""
echo "Notebook 地址:"
echo "  - 首页: http://localhost:7777"
echo "  - 模型架构: http://localhost:7777/notebooks/model_architecture"
echo "  - 注意力机制: http://localhost:7777/notebooks/attention_mechanism"
echo "  - 神经网络层: http://localhost:7777/notebooks/layer_visualization"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

clojure -M:clerk
