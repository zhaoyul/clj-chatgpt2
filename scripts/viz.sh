#!/bin/bash
# 注意力可视化服务器启动脚本

PORT=${1:-8888}

echo "📊 启动注意力可视化服务器..."
echo "   Port: $PORT"
echo ""

clojure -M -m gpt2.viz-server "$PORT"
