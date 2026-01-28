#!/bin/bash
# 启动完整的可视化环境（Python API + Clojure 静态服务器）

PYTHON_PORT=5000
CLOJURE_PORT=8888

echo "🚀 启动完整可视化环境..."
echo ""

# 检查 Python API 依赖
echo "📦 检查 Python 依赖..."
pip show flask flask-cors transformers torch >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  缺少依赖，正在安装..."
    pip install flask flask-cors transformers torch
fi

# 启动 Python API 服务（后台）
echo "🐍 启动 Python API 服务..."
echo "   API: http://localhost:$PYTHON_PORT"
python3 scripts/attention_api.py --port $PYTHON_PORT &
PYTHON_PID=$!
sleep 3

# 检查 Python 服务是否启动成功
if ! curl -s http://localhost:$PYTHON_PORT/health >/dev/null; then
    echo "❌ Python API 启动失败"
    exit 1
fi
echo "✅ Python API 已启动"
echo ""

# 启动 Clojure 静态服务器
echo "☕ 启动 Clojure 静态服务器..."
echo "   URL: http://localhost:$CLOJURE_PORT"
echo ""

# 确保在脚本退出时停止 Python 服务
cleanup() {
    echo ""
    echo "🛑 正在停止服务..."
    kill $PYTHON_PID 2>/dev/null
    exit 0
}
trap cleanup INT TERM

clojure -M -m gpt2.viz-server $CLOJURE_PORT &
CLOJURE_PID=$!

# 等待用户中断
wait
