# 神经网络可视化工具推荐

## 1. 专业模型可视化工具

### 1.1 Netron (强烈推荐)

**特点：**
- 支持 ONNX、PyTorch、TensorFlow 等 20+ 格式
- 交互式网络图，可缩放、拖拽
- 显示每层输入输出形状、参数数量
- 点击节点查看详细属性

**使用方式：**
```bash
# 安装
pip install netron

# 启动本地服务器
netron resources/onnx/model.onnx --port 8080

# 或在浏览器中使用
# https://netron.app/ (拖拽模型文件)
```

**集成到项目：**
```bash
# 添加启动脚本
echo '#!/bin/bash
echo "启动 Netron 可视化服务器..."
python3 -m netron resources/onnx/model.onnx --port 8080' > scripts/netron.sh
chmod +x scripts/netron.sh
```

### 1.2 TensorBoard

**特点：**
- 功能全面（图、指标、直方图）
- 与 PyTorch/TensorFlow 深度集成
- 支持 Embedding 投影可视化

**缺点：**
- 较重，需要额外依赖
- 主要针对训练过程

---

## 2. 注意力机制专用可视化

### 2.1 BertViz (最佳选择)

**特点：**
- 专为 Transformer 设计
- 交互式注意力头选择
- 神经元级别可视化
- 支持 GPT-2、BERT 等

**安装：**
```bash
pip install bertviz
```

### 2.2 自定义 D3.js 注意力可视化

创建轻量级 Web 可视化：
- 使用 D3.js 绘制注意力热力图
- 交互式选择层和头
- 实时显示注意力权重

---

## 3. 交互式 3D 可视化

### 3.1 TensorSpace
- 3D 神经网络可视化
- 层与层之间的数据流动画
- https://github.com/tensorspace-team/tensorspace

### 3.2 ONNX.js + Three.js
自定义 3D 可视化方案

---

## 4. 推荐的集成方案

### 方案 A：Netron + 自定义 Web UI (推荐)

**架构：**
```
Clojure API Server
    ↓
Web Dashboard (Reagent/Re-frame)
    ├── Netron Iframe (模型结构)
    ├── Custom Attention Viz (D3.js)
    └── Real-time Inference Monitor
```

**优点：**
- 快速实现，Netron 处理模型可视化
- 自定义部分专注于注意力/激活值

### 方案 B：全自定义 D3.js 可视化

**组件：**
1. **网络拓扑图** - 展示层间连接
2. **注意力热力图** - 动态交互
3. **神经元激活图** - 实时显示
4. **参数分布图** - 权重直方图

---

## 5. 实现示例：可视化服务器

创建一个新的可视化服务：

```clojure
;; src/gpt2/viz_server.clj
(ns gpt2.viz-server
  (:require [ring.adapter.jetty :refer [run-jetty]]
            [ring.middleware.resource :refer [wrap-resource]]
            [ring.util.response :refer [response content-type]]
            [clojure.data.json :as json]))

(defn viz-handler [request]
  (case (:uri request)
    "/" 
    (-> (response viz-html-page)
        (content-type "text/html"))
    
    "/api/model-info"
    {:status 200
     :headers {"Content-Type" "application/json"}
     :body (json/write-str 
             {:layers 12
              :attention-heads 12
              :parameters 117000000
              :hidden-size 768})}
    
    {:status 404}))
```

---

## 6. 快速开始建议

**最快路径（5分钟）：**

1. **安装 Netron：**
   ```bash
   pip install netron
   netron resources/onnx/model.onnx
   ```

2. **使用在线工具：**
   - 打开 https://netron.app/
   - 拖拽 model.onnx 文件

**进阶路径（1小时）：**

1. 创建基于 D3.js 的注意力可视化
2. 集成到现有的 Web 服务
3. 添加实时推理监控

需要我帮你实现其中任何一个方案吗？
