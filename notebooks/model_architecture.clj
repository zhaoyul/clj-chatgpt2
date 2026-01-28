;; # GPT-2 模型架构分析
;; 
;; 本 Notebook 深入解析 GPT-2 模型的内部结构，包括神经网络分层、Transformer 架构和注意力机制。

(ns model-architecture
  (:require [nextjournal.clerk :as clerk]
            [gpt2.token :as token]
            [gpt2.model :as model]
            [clojure.string :as str]
            [clojure.data.json :as json])
  (:import [java.nio.file Paths]))

;; ## 1. 模型概览

;; GPT-2 (Generative Pre-trained Transformer 2) 是一个基于 Transformer 解码器架构的语言模型。
;; 本分析基于 124M 参数版本的 ONNX 格式模型。

;; ONNX 模型信息（静态定义，避免运行时加载问题）
(def model-info
  {:input-info [{:name "input_ids"
                 :shape ["batch_size" "sequence_length"]
                 :type "tensor(int64)"}
                {:name "attention_mask"
                 :shape ["batch_size" "sequence_length"]
                 :type "tensor(int64)"}]
   :output-info [{:name "logits"
                  :shape ["batch_size" "sequence_length" 50257]
                  :type "tensor(float)"}]
   :num-ops "500+"
   :model-size-mb 623.44})

;; ### 1.1 输入输出结构

(clerk/table
  {:headers ["类型" "名称" "形状" "数据类型"]
   :rows (concat
           (mapv #(vector "输入" (:name %) (str/join " × " (:shape %)) (:type %))
                 (:input-info model-info))
           (mapv #(vector "输出" (:name %) (str/join " × " (:shape %)) (:type %))
                 (:output-info model-info)))})

;; **关键维度说明：**
;; - **batch_size**: 批处理大小（动态）
;; - **sequence_length**: 序列长度（动态，最大 1024）
;; - **vocab_size**: 词表大小 50257

;; ## 2. GPT-2 架构总览

;; ```
;; ┌─────────────────────────────────────────────────────────────┐
;; │                    GPT-2 Architecture                        │
;; ├─────────────────────────────────────────────────────────────┤
;; │  Input Tokens (batch_size × seq_len)                        │
;; │                      ↓                                      │
;; │  ┌─────────────────────────────────────────────────────┐   │
;; │  │ Token Embeddings (50257 × 768)                     │   │
;; │  │ Position Embeddings (1024 × 768)                   │   │
;; │  └─────────────────────────────────────────────────────┘   │
;; │                      ↓                                      │
;; │  ┌─────────────────────────────────────────────────────┐   │
;; │  │         Transformer Block × 12                      │   │
;; │  │  ┌─────────────────────────────────────────────┐   │   │
;; │  │  │ LayerNorm + Multi-Head Attention + Residual │   │   │
;; │  │  │ LayerNorm + Feed Forward (MLP) + Residual   │   │   │
;; │  │  └─────────────────────────────────────────────┘   │   │
;; │  └─────────────────────────────────────────────────────┘   │
;; │                      ↓                                      │
;; │  ┌─────────────────────────────────────────────────────┐   │
;; │  │ Final LayerNorm                                     │   │
;; │  │ Linear Projection to Vocab (768 × 50257)           │   │
;; │  └─────────────────────────────────────────────────────┘   │
;; │                      ↓                                      │
;; │  Output Logits (batch_size × seq_len × 50257)             │
;; └─────────────────────────────────────────────────────────────┘
;; ```

;; ## 3. 词嵌入层 (Embeddings)

;; GPT-2 使用两个嵌入层的组合：

^{::clerk/visibility :folded}
(defn embedding-viz
  "展示嵌入层结构"
  []
  {:token-embedding {:vocab-size 50257
                     :hidden-size 768
                     :parameters (* 50257 768)}
   :position-embedding {:max-positions 1024
                        :hidden-size 768
                        :parameters (* 1024 768)}
   :total-embedding-params (+ (* 50257 768) (* 1024 768))})

(def embedding-info (embedding-viz))

(clerk/html
  [:div.grid.grid-cols-2.gap-4
   [:div.bg-blue-50.p-4.rounded
    [:h3.font-bold.text-blue-800 "Token 嵌入"]
    [:ul.mt-2.space-y-1
     [:li "词表大小: " [:span.font-mono "50,257"]]
     [:li "隐藏维度: " [:span.font-mono "768"]]
     [:li "参数量: " [:span.font-mono (format "%,d" (:parameters (:token-embedding embedding-info)))]]]]
   [:div.bg-green-50.p-4.rounded
    [:h3.font-bold.text-green-800 "位置嵌入"]
    [:ul.mt-2.space-y-1
     [:li "最大位置: " [:span.font-mono "1,024"]]
     [:li "隐藏维度: " [:span.font-mono "768"]]
     [:li "参数量: " [:span.font-mono (format "%,d" (:parameters (:position-embedding embedding-info)))]]]]])

;; ## 4. Transformer 解码器块

;; GPT-2 包含 12 个相同的 Transformer 解码器层，每层包含两个子层：

;; ### 4.1 多头注意力机制 (Multi-Head Attention)

;; ```
;; ┌──────────────────────────────────────────────┐
;; │          Multi-Head Self-Attention           │
;; │              (12 heads × 64 dims)            │
;; ├──────────────────────────────────────────────┤
;; │                                              │
;; │  Input: X (batch × seq × 768)               │
;; │         ↓                                    │
;; │  ┌──────────────────────────────────────┐   │
;; │  │  Q = X × W_q  (768 × 768)            │   │
;; │  │  K = X × W_k  (768 × 768)            │   │
;; │  │  V = X × W_v  (768 × 768)            │   │
;; │  └──────────────────────────────────────┘   │
;; │         ↓                                    │
;; │  Split into 12 heads:                       │
;; │  Q_h, K_h, V_h (batch × 12 × seq × 64)      │
;; │         ↓                                    │
;; │  Attention(Q_h, K_h, V_h) =                 │
;; │    softmax(Q_h × K_h^T / √64) × V_h         │
;; │         ↓                                    │
;; │  Concatenate heads                          │
;; │         ↓                                    │
;; │  Output Projection (768 × 768)              │
;; │         ↓                                    │
;; │  Output: (batch × seq × 768)                │
;; │                                              │
;; └──────────────────────────────────────────────┘
;; ```

(clerk/table
  {:headers ["组件" "输入形状" "输出形状" "参数量"]
   :rows [["Q/K/V 线性投影" "B × S × 768" "B × S × 768" "3 × 768 × 768 = 1,769,472"]
          ["多头注意力" "B × 12 × S × 64" "B × 12 × S × 64" "计算操作"]
          ["输出投影" "B × S × 768" "B × S × 768" "768 × 768 = 589,824"]
          ["注意力层总计" "-" "-" "~2.36M"]]})

;; ### 4.2 注意力计算详解

^{::clerk/visibility :folded}
(defn attention-diagram
  "展示注意力计算过程"
  [seq-len]
  (let [head-dim 64
        scale (Math/sqrt head-dim)]
    {:scaled-dot-product-attention
     {:step1 "Q × K^T → 注意力分数矩阵 (seq × seq)"
      :step2 (format "/ %.1f → 缩放 (防止梯度消失)" scale)
      :step3 "Softmax → 注意力权重 (每行和为1)"
      :step4 "× V → 加权求和输出 (seq × head_dim)"}
     :causal-mask
     {:description "上三角矩阵（防止看到未来信息）"
      :example (for [i (range (min seq-len 5))]
                 (for [j (range (min seq-len 5))]
                   (if (<= j i) 1 0)))}}))

(def attention-calc (attention-diagram 10))

(clerk/html
  [:div.bg-gray-50.p-4.rounded.font-mono.text-sm
   [:h4.font-bold.mb-2 "缩放点积注意力计算:"]
   [:pre "Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V\n\nd_k = 64 (每个头的维度)\n√d_k = 8.0 (缩放因子)"]
   [:h4.font-bold.mb-2.mt-4 "因果掩码矩阵 (5×5 示例):"]
   [:pre (str/join "\n" (map #(str/join " " %) (:example (:causal-mask attention-calc))))]])

;; ### 4.3 前馈神经网络 (Feed Forward)

;; 每个 Transformer 层包含一个位置前馈网络（Position-wise Feed-Forward Network）：

;; ```
;; ┌──────────────────────────────────────────────┐
;; │            Feed-Forward Network              │
;; ├──────────────────────────────────────────────┤
;; │                                              │
;; │  Input: (batch × seq × 768)                 │
;; │         ↓                                    │
;; │  Linear 1: 768 → 3072  (×4 扩展)            │
;; │  [W_1: 768 × 3072, b_1: 3072]               │
;; │         ↓                                    │
;; │  GELU Activation                             │
;; │  GELU(x) = x × Φ(x) ≈ x × 0.5(1 + tanh(    │
;; │            √(2/π) × (x + 0.044715x³)))      │
;; │         ↓                                    │
;; │  Linear 2: 3072 → 768  (投影回原维度)       │
;; │  [W_2: 3072 × 768, b_2: 768]                │
;; │         ↓                                    │
;; │  Output: (batch × seq × 768)                │
;; │                                              │
;; │  参数量: 768×3072 + 3072 + 3072×768 + 768   │
;; │        = 2,362,368 + 3,072 + 2,359,296 + 768│
;; │        ≈ 4.72M                              │
;; │                                              │
;; └──────────────────────────────────────────────┘
;; ```

;; ## 5. 模型参数统计

^{::clerk/visibility :folded}
(defn parameter-count
  "计算各组件参数量"
  []
  (let [vocab-size 50257
        hidden-size 768
        num-layers 12
        intermediate-size (* 4 hidden-size)
        max-position 1024]
    {:embeddings
     {:token (* vocab-size hidden-size)
      :position (* max-position hidden-size)}
     :transformer-layer
     {:attention
      {:qkv (* 3 hidden-size hidden-size)
       :projection (* hidden-size hidden-size)
       :total (* 4 hidden-size hidden-size)}
      :feed-forward
      {:linear1 (+ (* hidden-size intermediate-size) intermediate-size)
       :linear2 (+ (* intermediate-size hidden-size) hidden-size)
       :total (+ (* hidden-size intermediate-size) intermediate-size
                 (* intermediate-size hidden-size) hidden-size)}
      :layer-norm (* 2 2 hidden-size) ; 两个 LayerNorm，每个有 weight 和 bias
      :total-per-layer (+ (* 4 hidden-size hidden-size)
                          (* hidden-size intermediate-size) intermediate-size
                          (* intermediate-size hidden-size) hidden-size
                          (* 4 hidden-size))}
     :total
     (+ (* vocab-size hidden-size) ; token embedding
        (* max-position hidden-size) ; position embedding
        (* num-layers (+ (* 4 hidden-size hidden-size)
                         (* hidden-size intermediate-size) intermediate-size
                         (* intermediate-size hidden-size) hidden-size
                         (* 4 hidden-size))) ; 12 层 transformer
        (* 2 hidden-size))})) ; final layer norm

(def params (parameter-count))

;; ### 5.1 参数分布可视化

(clerk/plotly
  {:data [{:x ["Token Embeddings" "Position Embeddings" "Attention (12层)" 
               "Feed-Forward (12层)" "LayerNorm" "Final LayerNorm"]
          :y [(/ (:token (:embeddings params)) 1e6)
             (/ (:position (:embeddings params)) 1e6)
             (/ (* 12 (:qkv (:attention (:transformer-layer params)))) 1e6)
             (/ (* 12 (:total (:feed-forward (:transformer-layer params)))) 1e6)
             (/ (* 12 (:layer-norm (:transformer-layer params))) 1e6)
             0.0015]
          :type "bar"
          :marker {:color ["#3B82F6" "#10B981" "#F59E0B" "#EF4444" "#8B5CF6" "#6B7280"]}}]
   :layout {:title "GPT-2 参数分布 (百万)"
            :yaxis {:title "参数量 (M)"}
            :xaxis {:title "组件"}}})

;; ### 5.2 详细参数表

(clerk/table
  {:headers ["组件" "层数" "每层的参数量" "总参数量"]
   :rows [["Token Embeddings" "1" "-" (format "%,d" (:token (:embeddings params)))]
          ["Position Embeddings" "1" "-" (format "%,d" (:position (:embeddings params)))]
          ["Attention Q/K/V" "12" (format "%,d" (:qkv (:attention (:transformer-layer params))))
           (format "%,d" (* 12 (:qkv (:attention (:transformer-layer params)))))]
          ["Attention 输出投影" "12" (format "%,d" (:projection (:attention (:transformer-layer params))))
           (format "%,d" (* 12 (:projection (:attention (:transformer-layer params)))))]
          ["Feed-Forward" "12" (format "%,d" (:total (:feed-forward (:transformer-layer params))))
           (format "%,d" (* 12 (:total (:feed-forward (:transformer-layer params)))))]
          ["LayerNorm" "24" (format "%,d" (:layer-norm (:transformer-layer params)))
           (format "%,d" (* 12 (:layer-norm (:transformer-layer params))))]
          ["**总计**" "-" "-" "**~117M**"]]})

;; ## 6. 推理流程可视化

;; 展示一个 token 的生成过程：

^{::clerk/visibility :folded}
(defn inference-flow
  "展示推理流程"
  [input-text]
  (let [tokens (token/encode input-text)
        token-ids tokens
        seq-len (count tokens)]
    {:input input-text
     :tokenized tokens
     :embedding-step {:shape [1 seq-len 768]
                      :operation "Lookup + Position Encoding"}
     :transformer-steps (for [layer (range 12)]
                          {:layer (inc layer)
                           :attention {:type "Masked Self-Attention"
                                       :heads 12
                                       :head-dim 64}
                           :ffn {:hidden (* 4 768)
                                 :activation "GELU"}})
     :output {:logits-shape [1 seq-len 50257]
              :next-token "argmax or sampling"}}))

(def flow (inference-flow "Hello, world!"))

;; ### 6.1 输入处理流程

(clerk/html
  [:div.space-y-4
   [:div.bg-blue-50.p-4.rounded
    [:h4.font-bold.text-blue-800 "1. 文本输入"]
    [:p (:input flow)]]
   [:div.bg-green-50.p-4.rounded
    [:h4.font-bold.text-green-800 "2. Tokenization"]
    [:p "Token IDs: " [:span.font-mono (str (:tokenized flow))]]
    [:p.text-sm.text-gray-600 "词汇映射: "
     (for [id (:tokenized flow)]
       [:span.font-mono.mr-2 (str id "→" (token/decode [id]))])]]
   [:div.bg-purple-50.p-4.rounded
    [:h4.font-bold.text-purple-800 "3. 嵌入层"]
    [:p "输出形状: " [:span.font-mono (str/join " × " (:shape (:embedding-step flow)))]]
    [:p.text-sm (:operation (:embedding-step flow))]]
   [:div.bg-orange-50.p-4.rounded
    [:h4.font-bold.text-orange-800 "4. Transformer 层 × 12"]
    [:p "每层包含:"]
    [:ul.list-disc.ml-5
     [:li "Masked Multi-Head Self-Attention (12 heads, d_k=64)"]
     [:li "Residual Connection + LayerNorm"]
     [:li "Position-wise Feed-Forward (768→3072→768)"]
     [:li "Residual Connection + LayerNorm"]]]
   [:div.bg-red-50.p-4.rounded
    [:h4.font-bold.text-red-800 "5. 输出生成"]
    [:p "Logits 形状: " [:span.font-mono (str/join " × " (:logits-shape (:output flow)))]]
    [:p "下一步: 取最后一个位置的 logits，应用 softmax 得到概率分布"]
    [:p "采样策略: Greedy (argmax) 或 Top-K / Temperature sampling"]]])

;; ## 7. 注意力可视化示例

;; 展示注意力权重矩阵的可视化概念：

(clerk/plotly
  {:data [{:z (for [i (range 10)]
                (for [j (range 10)]
                  (if (<= j i)
                    (let [score (/ 1.0 (inc (- i j)))]
                      (* score (rand)))
                    0)))
          :x (map #(str "t" %) (range 10))
          :y (map #(str "t" %) (range 10))
          :type "heatmap"
          :colorscale "Blues"}]
   :layout {:title "因果注意力权重矩阵示例 (Causal Masked Attention)"
            :xaxis {:title "Key Position"}
            :yaxis {:title "Query Position"}
            :width 500
            :height 500}})

;; ## 8. 总结

;; GPT-2 124M 模型的关键特征：

(clerk/html
  [:div.grid.grid-cols-2.gap-4.mt-4
   [:div.bg-gradient-to-br.from-blue-50.to-blue-100.p-4.rounded.shadow
    [:h3.font-bold.text-blue-800 "架构特点"]
    [:ul.list-disc.ml-5.space-y-1
     [:li "纯解码器 Transformer 架构"]
     [:li "12 层 Transformer blocks"]
     [:li "12 头注意力机制 (d_k=64)"]
     [:li "隐藏层维度: 768"]
     [:li "FFN 中间层: 3072 (4×)"]]]
   [:div.bg-gradient-to-br.from-green-50.to-green-100.p-4.rounded.shadow
    [:h3.font-bold.text-green-800 "关键机制"]
    [:ul.list-disc.ml-5.space-y-1
     [:li "因果/自回归注意力掩码"]
     [:li "可学习的位置编码"]
     [:li "Pre-LayerNorm 结构"]
     [:li "GELU 激活函数"]
     [:li "Dropout 正则化"]]]
   [:div.bg-gradient-to-br.from-purple-50.to-purple-100.p-4.rounded.shadow
    [:h3.font-bold.text-purple-800 "参数量"]
    [:ul.list-disc.ml-5.space-y-1
     [:li "总参数量: ~117M"]
     [:li "嵌入层: ~39M (33%)"]
     [:li "注意力: ~21M (18%)"]
     [:li "前馈网络: ~42M (36%)"]]]
   [:div.bg-gradient-to-br.from-orange-50.to-orange-100.p-4.rounded.shadow
    [:h3.font-bold.text-orange-800 "性能指标"]
    [:ul.list-disc.ml-5.space-y-1
     [:li "上下文长度: 1024 tokens"]
     [:li "词表大小: 50257 (BPE)"]
     [:li "推理速度: ~100 tokens/s (CPU)"]
     [:li "模型大小: 623 MB (ONNX)"]]]])

;; ## 参考

;; 1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
;; 2. [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 论文
;; 3. [ Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化指南
