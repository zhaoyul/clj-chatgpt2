# clj-chatgpt2

> 基于 Clojure + DJL + ONNX Runtime 的 GPT-2 推理引擎

## 项目状态

✅ **项目已完成** - 包含完整的前后端实现、模型导出脚本和测试套件

## 目录

- [快速开始](#快速开始)
- [模型管理](#模型管理)
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

### 1. 克隆项目

```bash
git clone <repository-url>
cd clj-chatgpt2
```

**注意**: 模型文件 (~623MB) 不包含在 Git 仓库中，需要单独下载。

### 2. 下载模型

#### 方式一: 使用 Make (推荐)

```bash
# 安装 Python 依赖并下载模型
make install
```

#### 方式二: 使用 Python 脚本

```bash
# 安装依赖
pip install transformers torch onnx

# 下载并导出 ONNX 模型
python scripts/setup_model.py --model gpt2
```

#### 方式三: 手动下载

如果你有现成的 GPT-2 ONNX 模型，直接复制到:
```
resources/onnx/model.onnx
```

**支持的模型:**
- `gpt2` (124M) - 默认，速度快
- `gpt2-medium` (345M) - 更好的质量
- `gpt2-large` (774M) - 更大，需要更多内存
- `gpt2-xl` (1.5B) - 最大，需要 GPU

**验证模型:**
```bash
ls -lh resources/onnx/model.onnx  # 应该显示 ~623 MB
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

## 模型管理

### 模型文件 (.gitignore)

模型文件默认被排除在版本控制外：

```gitignore
resources/onnx/*.onnx      # ONNX 模型 (~623MB)
resources/onnx/*.bin       # 二进制权重
resources/onnx/*.safetensors
resources/weights/         # 提取的权重
```

### Makefile 命令

```bash
# 查看所有可用命令
make help

# 下载默认模型 (gpt2, 124M)
make model

# 下载更大的模型
make model-medium   # 345M
make model-large    # 774M

# 仅下载分词器
make tokenizer

# 清理缓存
make clean

# 删除模型文件
make clean-model
```

### 手动导出模型

如果你需要自定义导出参数：

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 导出 ONNX
torch.onnx.export(
    model,
    (dummy_input, dummy_mask),
    "resources/onnx/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={...},
    opset_version=14
)
```

---

## Clerk 可视化分析

项目包含基于 **Clerk** 的交互式 Notebook，用于可视化展示 GPT-2 模型的内部结构：

### 启动 Notebook 服务器

```bash
# 使用脚本启动（默认端口 7788）
./scripts/clerk.sh

# 指定端口
./scripts/clerk.sh 8080

# 或直接运行
clojure -M -e "
(require '[nextjournal.clerk :as clerk])
(clerk/serve! {:browse? true :watch-paths ['notebooks'] :port 7788})
@(promise)
"
```

### Notebook 列表

| Notebook | 内容 | URL |
|----------|------|-----|
| **🏠 首页** | Notebook 索引和导航 | http://localhost:7788/notebooks/index |
| **🏗️ 模型架构** | 整体架构、参数分布、ONNX 结构 | http://localhost:7788/notebooks/model_architecture |
| **🎯 注意力机制** | 自注意力、多头注意力、因果掩码 | http://localhost:7788/notebooks/attention_mechanism |
| **🔬 神经网络层** | 权重矩阵、激活函数、信息流动 | http://localhost:7788/notebooks/layer_visualization |
| **🎯 真实权重** | 从 ONNX 提取的真实 GPT-2 权重 | http://localhost:7788/notebooks/real_weights |
| **🤖 问答演示** | GPT-2 问答功能展示 | http://localhost:7788/notebooks/qa_demo |

### 2. Netron 模型可视化 (推荐)

**Netron** 是一个专业的神经网络模型可视化工具，支持交互式查看 ONNX 模型结构。

```bash
# 启动 Netron 可视化服务器
./scripts/netron.sh        # 默认端口 8080
./scripts/netron.sh 9000   # 自定义端口
```

然后打开 http://localhost:8080 查看：
- **交互式网络图** - 可缩放、拖拽查看模型结构
- **层属性查看** - 点击任意层查看输入输出形状、参数数量
- **数据流追踪** - 理解数据在模型中的流动

或者使用在线版本：https://netron.app/ (直接拖拽 model.onnx 文件)

### 3. 注意力可视化服务器

#### 方案 A：纯静态页面（模拟数据）

```bash
# 启动可视化服务器（默认端口 8888）
clojure -M -m gpt2.viz-server
# 访问 http://localhost:8888
```

页面使用模拟的注意力数据展示效果。

#### 方案 B：动态页面（真实 GPT-2 注意力权重）

需要同时运行 Python API 服务和 Clojure 静态服务器：

**方式 1：分别启动**

```bash
# 终端 1：启动 Python API（提供真实注意力权重）
python3 scripts/attention_api.py --port 5000

# 终端 2：启动静态页面服务器
clojure -M -m gpt2.viz-server 8888
```

**方式 2：一键启动（推荐）**

```bash
# 安装 Python 依赖
pip install flask flask-cors transformers torch

# 启动完整环境
./scripts/start-viz-full.sh
```

**功能：**
- **真实注意力权重** - 从 GPT-2 模型提取
- **注意力热力图** - 矩阵形式展示
- **注意力连接图** - 网络图形式展示
- **交互式控制** - 选择层（1-12）和注意力头（1-12）
- **实时计算** - 输入任意文本查看其注意力模式

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
│   ├── clerk.sh                # Clerk notebook 启动脚本
│   ├── netron.sh               # Netron 模型可视化
│   ├── viz.sh                  # 注意力可视化服务器
│   ├── attention_api.py        # Python API（真实注意力权重）
│   └── start-viz-full.sh       # 启动完整可视化环境
├── notebooks/
│   ├── index.clj               # Notebook 首页
│   ├── model_architecture.clj  # 模型架构分析
│   ├── attention_mechanism.clj # 注意力机制解析
│   └── layer_visualization.clj # 神经网络分层
├── src/gpt2/
│   ├── token.clj               # JTokkit 分词器封装
│   ├── model.clj               # DJL 模型加载与推理
│   ├── generate.clj            # 贪婪/Top-K 解码算法
│   ├── server.clj              # Ring/Reitit Web API
│   └── viz_server.clj          # 可视化服务器
├── test/gpt2/
│   ├── token_test.clj          # 分词器测试
│   └── generate_test.clj       # 生成算法测试
├── resources/onnx/
│   ├── model.onnx              # GPT-2 ONNX 模型 (623 MB)
│   ├── vocab.json              # 词表
│   ├── merges.txt              # BPE 合并规则
│   ├── tokenizer_config.json   # 分词器配置
│   └── special_tokens_map.json # 特殊标记映射
└── resources/public/
    ├── attention-viz.html      # 静态注意力可视化页面
    └── attention-viz-dynamic.html # 动态注意力可视化页面（需 API）
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

## 8. 可视化工具对比

| 工具 | 类型 | 启动命令 | 数据 | 特点 |
|------|------|---------|------|------|
| **Clerk** | Notebook | `./scripts/clerk.sh` | 静态/真实 | 交互式文档、代码与图表混排 |
| **Netron** | 模型查看器 | `./scripts/netron.sh` | 模型文件 | 专业 ONNX 可视化 |
| **Viz Static** | Web | `./scripts/viz.sh` | 模拟 | 注意力可视化（演示效果） |
| **Viz Dynamic** | Web | `./scripts/start-viz-full.sh` | 真实 GPT-2 | 真实注意力权重 |

### Clerk Notebook 说明

| Notebook | 内容 | 数据来源 |
|----------|------|----------|
| `index` | 项目导航和概览 | 静态 |
| `model_architecture` | 架构分析、参数统计 | 静态 |
| `attention_mechanism` | 注意力原理讲解 | 静态 |
| `layer_visualization` | 层次结构可视化 | 静态 |
| **`real_weights`** | **真实权重可视化** | **ONNX 模型提取** |

### 推荐使用流程

```bash
# 1. 查看模型结构
./scripts/netron.sh

# 2. 学习架构原理
./scripts/clerk.sh

# 3. 探索注意力模式
./scripts/viz.sh
```

---

## 9. 总结

本方案采用 **DJL + ONNX Runtime + JTokkit** 技术栈，在 JVM 上实现 GPT-2 推理引擎：

1. **模型层**：ONNX Runtime 提供接近原生 C++ 的高性能推理
2. **逻辑层**：Clojure 函数式编程简化解码算法实现
3. **服务层**：Ring + Reitit 构建高并发 Web 服务
4. **可视化层**：多种工具支持模型理解和调试

该方案适用于需要将 AI 能力集成到现有 JVM 基础设施，或对系统稳定性有极高要求的生产环境。

---

## 许可证

MIT License
