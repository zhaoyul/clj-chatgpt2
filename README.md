# clj-chatgpt2

> 基于 Clojure + DJL + ONNX Runtime 的 GPT-2 推理引擎

## 项目状态

✅ **项目已完成** - 包含完整的前后端实现、模型导出脚本和测试套件

## 目录

- [快速开始](#快速开始)
- [Clerk 可视化分析](#clerk-可视化分析)
- [1. 项目概述](#1-项目概述)
- [2. 技术架构](#2-技术架构)
- [3. 项目结构](#3-项目结构)
- [4. 使用指南](#4-使用指南)
- [5. API 文档](#5-api-文档)
- [6. 开发指南](#6-开发指南)
- [7. 性能优化](#7-性能优化)

---

## 快速开始

### 1. 克隆项目并下载模型

```bash
# 模型文件已包含在项目中
ls resources/onnx/model.onnx  # 623 MB ONNX 模型
```

### 2. 运行测试

```bash
clojure -M:test -e "
  (require '[clojure.test :refer :all])
  (require 'gpt2.token-test 'gpt2.generate-test)
  (run-tests 'gpt2.token-test 'gpt2.generate-test)
"
```

### 3. 启动服务

```bash
clojure -M -m gpt2.server 3000
```

### 4. 测试 API

```bash
curl -X POST http://localhost:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 20, "strategy": "greedy"}'
```

**响应：**
```json
{
  "generated_text": "Hello, world!\n\nI'm sorry, but I'm not sure what to do.",
  "prompt": "Hello, world!",
  "params": {
    "max_tokens": 20,
    "strategy": "greedy",
    "k": 50,
    "temperature": 1.0
  }
}
```

---

## Clerk 可视化分析

项目包含基于 **Clerk** 的交互式 Notebook，用于可视化展示 GPT-2 模型的内部结构：

### 启动 Notebook 服务器

```bash
./scripts/clerk.sh
# 或
clojure -M:clerk
```

### Notebook 列表

| Notebook | 内容 | URL |
|----------|------|-----|
| **🏠 首页** | Notebook 索引和导航 | http://localhost:7777 |
| **🏗️ 模型架构** | 整体架构、参数分布、ONNX 结构 | http://localhost:7777/notebooks/model_architecture |
| **🎯 注意力机制** | 自注意力、多头注意力、因果掩码 | http://localhost:7777/notebooks/attention_mechanism |
| **🔬 神经网络层** | 权重矩阵、激活函数、信息流动 | http://localhost:7777/notebooks/layer_visualization |

### 可视化示例

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT-2 Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  Input Tokens (batch_size × seq_len)                        │
│                      ↓                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Token Embeddings (50257 × 768)                     │   │
│  │ Position Embeddings (1024 × 768)                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                      ↓                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Transformer Block × 12                      │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │ LayerNorm + Multi-Head Attention + Residual │   │   │
│  │  │ LayerNorm + Feed Forward (MLP) + Residual   │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                      ↓                                      │
│  Output Logits (batch_size × seq_len × 50257)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 项目概述

本项目基于 Clojure 构建 GPT-2 推理引擎，采用 DJL (Deep Java Library) + ONNX Runtime 的技术栈。该方案平衡了开发效率与运行性能，利用 Clojure 的函数式编程特性处理复杂的解码逻辑，同时借助 ONNX Runtime 获得接近原生 C++ 的推理性能。

**核心优势**：
- 利用 Clojure 的不可变数据结构安全管理解码状态
- 基于 JVM 线程模型构建高并发推理服务
- REPL 驱动开发支持实时调试张量操作
- ONNX 模型格式支持跨平台部署

---

## 2. 技术架构

### 2.1 核心技术栈

| 层级             | 技术选型           | 说明                            |
|------------------|--------------------|---------------------------------|
| **编程语言**     | Clojure 1.12       | JVM 上的函数式 Lisp 方言        |
| **深度学习框架** | DJL 0.29           | 亚马逊开源的 Java 深度学习库    |
| **推理引擎**     | ONNX Runtime 1.18  | 高性能跨平台推理引擎            |
| **分词器**       | JTokkit 1.1        | 针对 GPT-2 优化的 Java BPE 实现 |
| **Web 框架**     | Reitit 0.7 + Ring  | 高性能路由 + HTTP 服务          |
| **模型格式**     | ONNX               | 跨语言模型交换标准              |
| **模型来源**     | Hugging Face GPT-2 | 124M 参数版本                   |

### 2.2 核心组件映射

| 组件     | Python/PyTorch    | Clojure/JVM                 |
|----------|-------------------|-----------------------------|
| 张量计算 | `torch.Tensor`    | DJL `NDArray`               |
| 模型加载 | `torch.nn.Module` | DJL `Criteria` + `ZooModel` |
| 分词器   | `tiktoken`        | JTokkit `Encoding`          |
| 执行引擎 | PyTorch Runtime   | ONNX Runtime via DJL        |
| Web 服务 | FastAPI           | Ring + Reitit + Jetty       |

---

## 3. 项目结构

```text
clj-chatgpt2/
├── deps.edn                    # 依赖配置
├── README.md                   # 项目文档
├── .gitignore                  # Git 忽略配置
├── scripts/
│   ├── export_model.py         # Python 模型导出脚本
│   ├── run.sh                  # 服务启动脚本
│   └── clerk.sh                # Clerk notebook 启动脚本
├── notebooks/
│   ├── index.clj               # Notebook 首页
│   ├── model_architecture.clj  # 模型架构分析
│   ├── attention_mechanism.clj # 注意力机制解析
│   └── layer_visualization.clj # 神经网络分层
├── src/gpt2/
│   ├── token.clj               # JTokkit 分词器封装
│   ├── model.clj               # DJL 模型加载与推理
│   ├── generate.clj            # 贪婪/Top-K 解码算法
│   └── server.clj              # Ring/Reitit Web API
├── test/gpt2/
│   ├── token_test.clj          # 分词器测试
│   └── generate_test.clj       # 生成算法测试
└── resources/onnx/
    ├── model.onnx              # GPT-2 ONNX 模型 (623 MB)
    ├── vocab.json              # 词表
    ├── merges.txt              # BPE 合并规则
    ├── tokenizer_config.json   # 分词器配置
    └── special_tokens_map.json # 特殊标记映射
```

---

## 4. 使用指南

### 4.1 模型导出（如需要更新模型）

```bash
# 安装 Python 依赖
pip install transformers==4.39.3 torch==2.2.2 numpy==1.26.4 onnx

# 导出 ONNX 模型
python scripts/export_model.py --model gpt2 --output resources/onnx/

# 可选：导出更大的模型
# python scripts/export_model.py --model gpt2-medium --output resources/onnx/
```

### 4.2 REPL 交互式开发

```bash
clojure -M
```

```clojure
;; 加载命名空间
(require '[gpt2.token :as token])
(require '[gpt2.generate :as gen])

;; 测试分词器
(token/encode "Hello, world!")
;; => [15496 11 995 0]

(token/decode [15496 11 995 0])
;; => "Hello, world!"

;; 生成文本
(gen/generate-text "Once upon a time" :max-tokens 30)
;; => "Once upon a time, there was a little girl named Alice."

;; Top-K 采样生成
(gen/generate-text "Hello" 
                   :max-tokens 20 
                   :strategy :top-k 
                   :k 40 
                   :temperature 0.8)
```

### 4.3 启动 Web 服务

```bash
# 默认端口 3000
clojure -M -m gpt2.server

# 指定端口
clojure -M -m gpt2.server 8080

# 或使用脚本
./scripts/run.sh 3000
```

---

## 5. API 文档

### 5.1 文本生成接口

**POST /api/generate**

生成文本（非流式）。

**请求体：**
```json
{
  "prompt": "Hello, world!",      // 输入提示（必需）
  "max_tokens": 50,               // 最大生成 token 数（默认 50）
  "strategy": "greedy",           // 解码策略：greedy 或 top-k（默认 greedy）
  "k": 50,                        // Top-K 值（默认 50）
  "temperature": 1.0              // 温度参数（默认 1.0）
}
```

**响应：**
```json
{
  "generated_text": "Hello, world! I'm a language model...",
  "prompt": "Hello, world!",
  "params": {
    "max_tokens": 50,
    "strategy": "greedy",
    "k": 50,
    "temperature": 1.0
  }
}
```

**示例：**
```bash
curl -X POST http://localhost:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?", "max_tokens": 30, "strategy": "top-k", "k": 40}'
```

### 5.2 流式生成接口

**POST /api/stream**

SSE 流式返回生成的 token。

**请求体：** 同 `/api/generate`

**响应：** Server-Sent Events 流

```bash
curl -X POST http://localhost:3000/api/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

### 5.3 健康检查

**GET /health**

```bash
curl http://localhost:3000/health
```

**响应：**
```json
{"status": "ok", "model_loaded": true}
```

---

## 6. 开发指南

### 6.1 运行测试

```bash
# 运行所有测试
clojure -M:test -e "
  (require '[clojure.test :refer :all])
  (require 'gpt2.token-test 'gpt2.generate-test)
  (run-tests 'gpt2.token-test 'gpt2.generate-test)
"
```

**预期输出：**
```
Testing gpt2.token-test
Testing gpt2.generate-test

Ran 7 tests containing 26 assertions.
0 failures, 0 errors.
```

### 6.2 代码结构说明

**token.clj** - 分词器封装
- `encode` - 文本编码为 token ID 序列
- `decode` - token ID 序列解码为文本
- `eos-token` - 结束标记常量 (50256)

**model.clj** - 模型推理
- `get-model` - 获取/加载 ONNX 模型
- `create-predictor` - 创建推理实例
- `forward-pass` - 执行前向传播

**generate.clj** - 文本生成
- `generate-text` - 生成完整文本
- `generate-stream` - 流式生成
- `argmax` - 贪婪解码
- `top-k-sample` - Top-K 采样解码

**server.clj** - Web 服务
- `generate-handler` - 生成接口处理函数
- `stream-handler` - 流式接口处理函数
- `start-server` / `stop-server` - 服务生命周期管理

---

## 7. 性能优化

### 7.1 KV Cache 实现

生产环境建议使用 KV Cache 避免重复计算历史序列的 Attention，将复杂度从 $O(N^2)$ 降至 $O(N)$。

**实现要点：**
- 在 `loop/recur` 中传递 `past-states`
- 每次推理返回更新后的 KV tensors
- 下次迭代将 KV tensors 作为输入传回

### 7.2 内存管理

- 使用 `NDManager` 管理堆外内存
- 使用 `try-finally` 确保 `Predictor` 关闭
- 中间产生的 `NDArray` 需要及时释放

**示例：**
```clojure
(let [predictor (model/create-predictor)]
  (try
    ;; 使用 predictor 进行推理
    (model/forward-pass predictor input-ids)
    (finally
      (.close predictor))))
```

### 7.3 并发处理

- `ZooModel` 线程安全，可全局共享
- `Predictor` **非线程安全**，每个请求需要独立实例
- 生产环境建议使用 `Predictor` 对象池或 `ThreadLocal`

---

## 8. 总结

本方案采用 **DJL + ONNX Runtime + JTokkit** 技术栈，在 JVM 上实现 GPT-2 推理引擎：

1. **模型层**：ONNX Runtime 提供接近原生 C++ 的高性能推理
2. **逻辑层**：Clojure 函数式编程简化解码算法实现
3. **服务层**：Ring + Reitit 构建高并发 Web 服务

该方案适用于需要将 AI 能力集成到现有 JVM 基础设施，或对系统稳定性有极高要求的生产环境。

---

## 许可证

MIT License
