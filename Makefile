# GPT-2 Clojure 项目 Makefile

.PHONY: help install model test run server clerk clean

# 默认目标
help:
	@echo "GPT-2 Clojure 项目命令:"
	@echo ""
	@echo "  make install    - 安装 Python 依赖并下载模型"
	@echo "  make model      - 仅下载并导出 ONNX 模型"
	@echo "  make test       - 运行测试"
	@echo "  make server     - 启动 API 服务 (端口 3000)"
	@echo "  make clerk      - 启动 Clerk Notebook (端口 7788)"
	@echo "  make run        - 同时启动服务和 Clerk"
	@echo "  make clean      - 清理缓存文件"
	@echo ""

# 安装所有依赖
install: model
	@echo "✅ 安装完成"

# 下载并设置模型
model:
	@echo "📥 正在下载 GPT-2 模型..."
	python3 scripts/setup_model.py --model gpt2

# 下载更大的模型
model-medium:
	python3 scripts/setup_model.py --model gpt2-medium

model-large:
	python3 scripts/setup_model.py --model gpt2-large

# 仅下载分词器（如果模型已存在）
tokenizer:
	python3 scripts/setup_model.py --skip-onnx

# 运行测试
test:
	clojure -M:dev -e "(require '[clojure.test :refer :all])(require 'gpt2.token-test 'gpt2.generate-test)(run-tests 'gpt2.token-test 'gpt2.generate-test)"

# 启动 API 服务
server:
	@echo "🚀 启动 API 服务 http://localhost:3000"
	clojure -M -m gpt2.server 3000

# 启动 Clerk
clerk:
	@echo "🚀 启动 Clerk http://localhost:7788"
	./scripts/clerk.sh 7788

# 同时启动服务和 Clerk（后台运行）
run:
	@echo "🚀 启动所有服务..."
	@make -j2 server clerk

# 清理缓存
clean:
	rm -rf .cpcache/
	rm -f .nrepl-port
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ 清理完成"

# 清理模型文件（谨慎使用）
clean-model:
	@echo "⚠️  即将删除模型文件..."
	@read -p "确认删除? [y/N] " confirm && [ "$${confirm}" = "y" ] && rm -f resources/onnx/model.onnx && echo "✅ 模型已删除" || echo "❌ 取消"
