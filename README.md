# clj-chatgpt2

> 基于 Clojure + DJL + ONNX Runtime 的 GPT-2 推理引擎

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 技术架构](#2-技术架构)
  - [2.1 核心技术栈](#21-核心技术栈)
  - [2.2 核心组件映射](#22-核心组件映射)
- [3. 开发环境准备](#3-开发环境准备)
  - [3.1 项目结构](#31-项目结构)
  - [3.2 依赖配置](#32-依赖配置)
- [4. 实施方案](#4-实施方案)
  - [4.1 模型导出](#41-模型导出)
  - [4.2 分词器实现](#42-分词器实现)
  - [4.3 模型加载与推理](#43-模型加载与推理)
  - [4.4 解码循环与生成](#44-解码循环与生成)
  - [4.5 Web API 服务](#45-web-api-服务)
- [5. 性能优化](#5-性能优化)
  - [5.1 KV Cache 实现](#51-kv-cache-实现)
  - [5.2 内存管理](#52-内存管理)
  - [5.3 并发处理](#53-并发处理)
- [6. 总结](#6-总结)

---

## 1. 项目概述

本项目基于 Clojure 构建 GPT-2 推理引擎，采用 DJL (Deep Java Library) + ONNX Runtime 的技术栈。该方案平衡了开发效率与运行性能，利用 Clojure 的函数式编程特性处理复杂的解码逻辑，同时借助 ONNX Runtime 获得接近原生 C++ 的推理性能。

**核心优势**：
- 利用 Clojure 的不可变数据结构安全管理解码状态
- 基于 JVM 线程模型构建高并发推理服务
- REPL 驱动开发支持实时调试张量操作

---

## 2. 技术架构

### 2.1 核心技术栈

| 层级             | 技术选型      | 说明                            |
|------------------|---------------|---------------------------------|
| **编程语言**     | Clojure 1.12  | JVM 上的函数式 Lisp 方言        |
| **深度学习框架** | DJL 0.29      | 亚马逊开源的 Java 深度学习库    |
| **推理引擎**     | ONNX Runtime  | 高性能跨平台推理引擎            |
| **分词器**       | JTokkit 1.1   | 针对 GPT-2 优化的 Java BPE 实现 |
| **Web 框架**     | Reitit + Ring | 高性能路由 + HTTP 服务          |
| **模型格式**     | ONNX          | 跨语言模型交换标准              |

### 2.2 核心组件映射

| 组件     | Python/PyTorch    | Clojure/JVM                 |
|----------|-------------------|-----------------------------|
| 张量计算 | `torch.Tensor`    | DJL `NDArray`               |
| 模型加载 | `torch.nn.Module` | DJL `Criteria` + `ZooModel` |
| 分词器   | `tiktoken`        | JTokkit `Encoding`          |
| 执行引擎 | PyTorch Runtime   | ONNX Runtime via DJL        |
| Web 服务 | FastAPI           | Ring + Reitit + Jetty       |

---

## 3. 开发环境准备

### 3.1 项目结构

```text
clj-chatgpt2/
├── deps.edn              # 依赖配置
├── src/
│   └── gpt2/
│       ├── token.clj     # 分词器封装
│       ├── model.clj     # 模型加载与推理
│       ├── generate.clj  # 解码算法实现
│       └── server.clj    # Web服务入口
├── resources/
│   └── onnx/
│       └── model.onnx    # 导出的GPT-2 ONNX模型
└── test/
    └── gpt2/
        └── model_test.clj
```

### 3.2 依赖配置

```clojure
;; deps.edn
{:deps {org.clojure/clojure {:mvn/version "1.12.0"}
        ;; DJL 核心组件
        ai.djl/api {:mvn/version "0.29.0"}
        ai.djl/onnxruntime-engine {:mvn/version "0.29.0"}
        ;; 分词器
        com.knuddels/jtokkit {:mvn/version "1.1.0"}
        ;; Web 服务栈
        metosin/reitit {:mvn/version "0.7.0-alpha7"}
        ring/ring-jetty-adapter {:mvn/version "1.12.1"}
        org.clojure/data.json {:mvn/version "2.5.0"}}
 :aliases {:dev {:extra-paths ["test"]}}}
```

---

## 4. 实施方案

### 4.1 模型导出

使用 Python 将 Hugging Face 的 GPT-2 模型导出为 ONNX 格式：

```bash
# 安装依赖
pip install transformers torch onnx onnxruntime

# 导出 ONNX 模型
python -m transformers.onnx --model=gpt2 --feature=causal-lm resources/onnx/
```

导出的 `model.onnx` 接受 `input_ids` 和 `attention_mask`，输出 `logits`。

### 4.2 分词器实现

`src/gpt2/token.clj`：

```clojure
(ns gpt2.token
  (:import [com.knuddels.jtokkit Encodings]
           [com.knuddels.jtokkit.api EncodingType]
           [it.unimi.dsi.fastutil.ints IntArrayList]))

(def registry (Encodings/newDefaultEncodingRegistry))

(def encoder (.getEncoding registry EncodingType/R50K_BASE))

(defn encode
  "将文本转换为 token ID 向量"
  [text]
  (let [tokens (.encode encoder text)]
    (vec (.toArray tokens))))

(defn decode
  "将 token ID 序列解码为文本"
  [token-ids]
  (let [int-array (int-array token-ids)
        int-list (IntArrayList. int-array)]
    (.decode encoder int-list)))
```

### 4.3 模型加载与推理

`src/gpt2/model.clj`：

```clojure
(ns gpt2.model
  (:require [gpt2.token :as token])
  (:import [ai.djl.repository.zoo Criteria ZooModel]
           [ai.djl.inference Predictor]
           [ai.djl.ndarray NDList NDManager]
           [ai.djl.translate NoopTranslator]
           [java.nio.file Paths]))

(defn build-criteria
  []
  (-> (Criteria/builder)
      (.setTypes NDList NDList)
      (.optModelPath (Paths/get "resources/onnx" (into-array String)))
      (.optModelName "model.onnx")
      (.optEngine "OnnxRuntime")
      (.optTranslator (NoopTranslator.))
      (.build)))

(def gpt-model (.loadModel (build-criteria)))

(defn create-predictor
  []
  (.newPredictor gpt-model))

(defn forward-pass
  "执行一次前向传播，返回最后一个 token 的 logits"
  [predictor input-ids]
  (let [manager (NDManager/newBaseManager)
        seq-len (count input-ids)
        input-array (.create manager (long-array input-ids)
                            (ai.djl.ndarray.types.Shape. 1 seq-len))
        mask-array (.ones manager (ai.djl.ndarray.types.Shape. 1 seq-len))
        inputs (NDList. input-array mask-array)]
    (let [outputs (.predict predictor inputs)
          logits-tensor (.get outputs 0)
          last-token-logits (.get logits-tensor (long-array [0 (dec seq-len)]))]
      (.toFloatArray last-token-logits))))
```

### 4.4 解码循环与生成

`src/gpt2/generate.clj`：

```clojure
(ns gpt2.generate
  (:require [gpt2.model :as model]
            [gpt2.token :as token]))

(defn argmax
  "返回数组中最大值的索引"
  [float-array]
  (let [len (alength float-array)]
    (loop [i 1
           max-idx 0
           max-val (aget float-array 0)]
      (if (>= i len)
        max-idx
        (let [val (aget float-array i)]
          (if (> val max-val)
            (recur (inc i) i val)
            (recur (inc i) max-idx max-val)))))))

(defn generate-text
  "使用贪婪搜索生成文本"
  [prompt max-tokens]
  (let [predictor (model/create-predictor)
        start-ids (token/encode prompt)
        eos-token 50256]  ; <|endoftext|>
    (try
      (loop [current-ids start-ids
             steps 0]
        (if (>= steps max-tokens)
          (token/decode current-ids)
          (let [logits (model/forward-pass predictor current-ids)
                next-token (argmax logits)]
            (if (= next-token eos-token)
              (token/decode current-ids)
              (recur (conj current-ids next-token) (inc steps))))))
      (finally
        (.close predictor)))))
```

### 4.5 Web API 服务

`src/gpt2/server.clj`：

```clojure
(ns gpt2.server
  (:require [reitit.ring :as ring]
            [ring.adapter.jetty :refer [run-jetty]]
            [muuntaja.core :as m]
            [reitit.ring.middleware.muuntaja :as muuntaja]
            [gpt2.generate :as generate]))

(defn chat-handler
  [request]
  (let [prompt (get-in request [:body-params :prompt])
        max-tokens (get-in request [:body-params :max_tokens] 50)
        response (generate/generate-text prompt max-tokens)]
    {:status 200
     :body {:generated_text response}}))

(def app
  (ring/ring-handler
   (ring/router
    [["/api"
      ["/chat" {:post chat-handler}]]]
    {:data {:muuntaja m/instance
            :middleware [muuntaja/format-middleware]}})))

(defn -main
  [& _args]
  (run-jetty app {:port 3000 :join? false}))
```

---

## 5. 性能优化

### 5.1 KV Cache 实现

生产环境必须使用 KV Cache 避免重复计算历史序列的 Attention，将复杂度从 $O(N^2)$ 降至 $O(N)$。

**实现要点**：
- 在 `loop/recur` 中传递 `past-states`
- 每次推理返回更新后的 KV tensors
- 下次迭代将 KV tensors 作为输入传回

### 5.2 内存管理

- 使用 `NDManager` 管理堆外内存
- 使用 `with-open` 或 `try-finally` 确保 `Predictor` 关闭
- 中间产生的 `NDArray` 需要及时释放

### 5.3 并发处理

- `ZooModel` 线程安全，可全局共享
- `Predictor` 非线程安全，每个请求需要独立的实例
- 使用 `Predictor` 对象池或 `ThreadLocal` 管理实例

---

## 6. 总结

本方案采用 **DJL + ONNX Runtime + JTokkit** 技术栈，在 JVM 上实现 GPT-2 推理引擎：

1. **模型层**：ONNX Runtime 提供接近原生 C++ 的高性能推理
2. **逻辑层**：Clojure 函数式编程简化解码算法实现
3. **服务层**：Ring + Reitit 构建高并发 Web 服务

该方案适用于需要将 AI 能力集成到现有 JVM 基础设施，或对系统稳定性有极高要求的生产环境。
