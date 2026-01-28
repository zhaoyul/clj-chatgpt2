#!/bin/bash
# GPT-2 Clojure 服务启动脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  GPT-2 Clojure 服务启动工具   ${NC}"
echo -e "${GREEN}================================${NC}"

# 检查模型文件
MODEL_FILE="resources/onnx/model.onnx"
if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${YELLOW}[WARN] 模型文件不存在: $MODEL_FILE${NC}"
    echo -e "${YELLOW}[INFO] 请先导出 ONNX 模型:${NC}"
    echo -e "${YELLOW}       python scripts/export_model.py${NC}"
    echo ""
    read -p "是否现在导出模型? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/export_model.py
    else
        echo -e "${RED}[ERROR] 缺少模型文件，无法启动服务${NC}"
        exit 1
    fi
fi

# 检查依赖
echo -e "${GREEN}[INFO] 检查依赖...${NC}"
clojure -P

# 启动服务
PORT=${1:-3000}
echo -e "${GREEN}[INFO] 启动服务，端口: $PORT${NC}"
echo -e "${GREEN}[INFO] API 端点: http://localhost:$PORT/api/generate${NC}"
echo -e "${GREEN}[INFO] 按 Ctrl+C 停止服务${NC}"
echo ""

clojure -M -m gpt2.server "$PORT"
