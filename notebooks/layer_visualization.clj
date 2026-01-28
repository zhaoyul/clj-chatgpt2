;; # 神经网络分层可视化
;; 
;; 本 Notebook 深入展示 GPT-2 的神经网络层次结构，从宏观架构到微观神经元。

(ns layer-visualization
  (:require [nextjournal.clerk :as clerk]
            [gpt2.token :as token]
            [clojure.string :as str]))

;; ## 1. GPT-2 整体架构层次

;; GPT-2 的层次结构可以用以下方式表示：

(clerk/html
  [:div.bg-gray-50.p-4.rounded.overflow-x-auto
   [:pre.font-mono.text-sm
    "┌─────────────────────────────────────────────────────────────────────┐\n"
    "│                        GPT-2 Architecture                           │\n"
    "│                        (12 Layers, 117M Params)                     │\n"
    "├─────────────────────────────────────────────────────────────────────┤\n"
    "│ Layer 0: Input & Embeddings                                         │\n"
    "│   ├─ Token Embedding    [50257 × 768]     = 38.6M params           │\n"
    "│   └─ Position Embedding [1024 × 768]      = 0.8M params            │\n"
    "├─────────────────────────────────────────────────────────────────────┤\n"
    "│ Layer 1-12: Transformer Blocks (×12)                                │\n"
    "│   ├─ Sub-layer 1: Multi-Head Self-Attention                         │\n"
    "│   │   ├─ Q/K/V Linear    [768 × 768] × 3  = 1.77M params           │\n"
    "│   │   ├─ Attention Compute  [12 heads × 64 dim]                     │\n"
    "│   │   └─ Output Linear   [768 × 768]      = 0.59M params           │\n"
    "│   ├─ Sub-layer 2: Feed-Forward Network                             │\n"
    "│   │   ├─ Linear 1        [768 × 3072]     = 2.36M params           │\n"
    "│   │   ├─ GELU Activation                                          │\n"
    "│   │   └─ Linear 2        [3072 × 768]     = 2.36M params           │\n"
    "│   └─ Layer Normalization (×2)        = 0.002M params               │\n"
    "├─────────────────────────────────────────────────────────────────────┤\n"
    "│ Layer 13: Output                                                    │\n"
    "│   ├─ Final LayerNorm                                              │\n"
    "│   └─ LM Head Linear    [768 × 50257]      = 38.6M params           │\n"
    "└─────────────────────────────────────────────────────────────────────┘"]])

;; ## 2. 层次化参数分布

;; 使用树状图展示参数分布：

^{::clerk/visibility :folded}
(defn parameter-hierarchy
  "参数层次结构"
  []
  {:embeddings {:token 38639104
                :position 786432
                :total 39425536}
   :transformer-layers (for [layer (range 1 13)]
                        {:layer layer
                         :attention 2359296
                         :feed-forward 4718592
                         :layernorm 1536
                         :total 7079424})
   :output {:final-layernorm 1536
            :lm-head 38639104
            :total 38640640}})

(def params (parameter-hierarchy))

;; ### 2.1 各层参数量对比

(clerk/plotly
  {:data [{:x (mapv #(str "Layer " (:layer %)) (:transformer-layers params))
          :y (mapv #(/ (:attention %) 1e6) (:transformer-layers params))
          :name "Attention"
          :type "bar"
          :marker {:color "#3B82F6"}}
         {:x (mapv #(str "Layer " (:layer %)) (:transformer-layers params))
          :y (mapv #(/ (:feed-forward %) 1e6) (:transformer-layers params))
          :name "Feed-Forward"
          :type "bar"
          :marker {:color "#EF4444"}}
         {:x (mapv #(str "Layer " (:layer %)) (:transformer-layers params))
          :y (mapv #(/ (:layernorm %) 1e6) (:transformer-layers params))
          :name "LayerNorm"
          :type "bar"
          :marker {:color "#10B981"}}]
   :layout {:title "每层 Transformer 的参数分布"
            :barmode "stack"
            :yaxis {:title "参数量 (M)"}
            :xaxis {:title "层数"}
            :legend {:orientation "h" :y 1.1}}})

;; ### 2.2 整体参数饼图

(clerk/plotly
  {:data [{:values [(/ (:token (:embeddings params)) 1e6)
                    (/ (:position (:embeddings params)) 1e6)
                    (* 12 (/ (:attention (first (:transformer-layers params))) 1e6))
                    (* 12 (/ (:feed-forward (first (:transformer-layers params))) 1e6))
                    (* 12 (/ (:layernorm (first (:transformer-layers params))) 1e6))
                    (/ (:lm-head (:output params)) 1e6)]
          :labels ["Token Embeddings" "Position Embeddings" "Attention (×12)" 
                   "Feed-Forward (×12)" "LayerNorm (×12)" "LM Head"]
          :type "pie"
          :hole 0.4
          :marker {:colors ["#3B82F6" "#10B981" "#F59E0B" "#EF4444" "#8B5CF6" "#6B7280"]}}]
   :layout {:title "GPT-2 参数分布 (单位: 百万)"}})

;; ## 3. Transformer 层内部结构

;; ### 3.1 单层的详细结构

(clerk/html
  [:div.space-y-4
   [:h3.font-bold "单层 Transformer 的详细结构"]
   
   ;; 输入
   [:div.bg-blue-50.p-3.rounded
    [:h4.font-semibold.text-blue-800 "输入"]
    [:p.text-sm "X: [batch_size, seq_len, 768]"]]
   
   ;; LayerNorm 1
   [:div.flex.items-center.gap-2
    [:div.flex-1.bg-gray-100.p-3.rounded
     [:h4.font-semibold "Layer Normalization 1"]
     [:p.text-sm "对每个 token 的 768 维向量进行归一化"]
     [:code.block.mt-1.text-xs.bg-white.p-1.rounded
      "LN(x) = (x - mean) / sqrt(var + ε) * γ + β"]]]
   
   ;; Multi-Head Attention
   [:div.bg-purple-50.p-4.rounded.border-2.border-purple-200
    [:h4.font-semibold.text-purple-800 "Multi-Head Self-Attention"]
    [:div.grid.grid-cols-3.gap-2.mt-2
     [:div.bg-white.p-2.rounded
      [:h5.font-medium.text-sm "Linear Q"]
      [:p.text-xs "[768 × 768]"]
      [:p.text-xs "589,824 params"]]
     [:div.bg-white.p-2.rounded
      [:h5.font-medium.text-sm "Linear K"]
      [:p.text-xs "[768 × 768]"]
      [:p.text-xs "589,824 params"]]
     [:div.bg-white.p-2.rounded
      [:h5.font-medium.text-sm "Linear V"]
      [:p.text-xs "[768 × 768]"]
      [:p.text-xs "589,824 params"]]]
    [:div.mt-2.bg-white.p-2.rounded
     [:h5.font-medium.text-sm "Scaled Dot-Product Attention"]
     [:p.text-xs "12 heads × (Q_h × K_h^T / √64 × V_h)"]]
    [:div.mt-2.bg-white.p-2.rounded
     [:h5.font-medium.text-sm "Output Projection"]
     [:p.text-xs "[768 × 768] = 589,824 params"]]]
   
   ;; Residual Connection 1
   [:div.flex.items-center.gap-2
    [:div.bg-yellow-50.p-2.rounded.text-center.flex-1
     [:p.text-sm.font-medium "Residual Connection"]
     [:code.text-xs "Output = X + Attention(LN(X))"]]]
   
   ;; LayerNorm 2
   [:div.flex.items-center.gap-2
    [:div.flex-1.bg-gray-100.p-3.rounded
     [:h4.font-semibold "Layer Normalization 2"]
     [:p.text-sm "相同的归一化操作"]]]
   
   ;; Feed-Forward
   [:div.bg-green-50.p-4.rounded.border-2.border-green-200
    [:h4.font-semibold.text-green-800 "Position-wise Feed-Forward"]
    [:div.grid.grid-cols-2.gap-2.mt-2
     [:div.bg-white.p-2.rounded
      [:h5.font-medium.text-sm "Linear 1"]
      [:p.text-xs "[768 → 3072]"]
      [:p.text-xs "2,362,368 params"]
      [:p.text-xs.mt-1.font-semibold "GELU Activation"]]
     [:div.bg-white.p-2.rounded
      [:h5.font-medium.text-sm "Linear 2"]
      [:p.text-xs "[3072 → 768]"]
      [:p.text-xs "2,359,296 params"]]]]
   
   ;; Residual Connection 2
   [:div.flex.items-center.gap-2
    [:div.bg-yellow-50.p-2.rounded.text-center.flex-1
     [:p.text-sm.font-medium "Residual Connection"]
     [:code.text-xs "Output = X + FFN(LN(X))"]]]
   
   ;; 输出
   [:div.bg-blue-50.p-3.rounded
    [:h4.font-semibold.text-blue-800 "输出"]
    [:p.text-sm "Y: [batch_size, seq_len, 768]"]]])

;; ## 4. 激活函数与归一化

;; ### 4.1 GELU 激活函数

;; GPT-2 使用 GELU (Gaussian Error Linear Unit) 作为激活函数：

^{::clerk/visibility :folded}
(defn gelu-function
  "GELU 激活函数"
  [x]
  (* x 0.5 (+ 1 (Math/tanh (* (Math/sqrt (/ 2 Math/PI))
                              (+ x (* 0.044715 (Math/pow x 3)))) ))))

(def gelu-data
  (for [x (range -4 4.1 0.2)]
    {:x x :gelu (gelu-function x) :relu (max 0 x)}))

(clerk/plotly
  {:data [{:x (mapv :x gelu-data)
          :y (mapv :gelu gelu-data)
          :name "GELU"
          :type "scatter"
          :mode "lines"
          :line {:color "#3B82F6" :width 3}}
         {:x (mapv :x gelu-data)
          :y (mapv :relu gelu-data)
          :name "ReLU"
          :type "scatter"
          :mode "lines"
          :line {:color "#EF4444" :width 2 :dash "dash"}}]
   :layout {:title "GELU vs ReLU 激活函数"
            :xaxis {:title "x" :range [-4 4]}
            :yaxis {:title "activation(x)" :range [-1 4]}}})

;; **GELU 特点：**
;; - 平滑的非线性（不像 ReLU 有尖锐的拐点）
;; - 在负数区域有小的负值（允许负激活）
;; - 更符合自然神经元的激活模式

;; ### 4.2 Layer Normalization

;; LayerNorm 对每个样本的特征维度进行归一化：

(clerk/html
  [:div.bg-gray-50.p-4.rounded
   [:h4.font-bold "Layer Normalization 计算过程:"]
   [:pre.font-mono.text-sm.mt-2
    "Input: x = [x₁, x₂, ..., x₇₆₈]  (一个 token 的 768 维向量)\n\n"
    "μ = mean(x) = (x₁ + x₂ + ... + x₇₆₈) / 768\n"
    "σ² = var(x) = Σ(xᵢ - μ)² / 768\n\n"
    "x̂ = (x - μ) / √(σ² + ε)    ← 归一化\n"
    "y = γ × x̂ + β              ← 缩放和平移\n\n"
    "参数: γ (gain), β (bias) - 每个维度一对，共 768 × 2 = 1536 参数"]
   [:p.text-sm.mt-2.text-gray-700
    "作用: 稳定训练过程，减少内部协变量偏移"]])

;; ## 5. 权重矩阵可视化

;; 展示权重矩阵的结构：

^{::clerk/visibility :folded}
(defn generate-weight-matrix
  "生成模拟的权重矩阵"
  [rows cols]
  (for [_ (range rows)]
    (for [_ (range cols)]
      (-> (rand) (- 0.5) (* 2) (* 0.1)))))

;; 缩小版本用于可视化
(def q-weight (generate-weight-matrix 32 32))

(clerk/plotly
  {:data [{:z q-weight
          :type "heatmap"
          :colorscale "RdBu"
          :zmid 0
          :showscale true}]
   :layout {:title "Query 权重矩阵示例 (32×32 子集)"
            :xaxis {:title "Input Dimension" :showticklabels false}
            :yaxis {:title "Output Dimension" :showticklabels false}
            :width 500
            :height 500}})

;; ## 6. 信息流动可视化

;; 展示数据在网络中的流动：

(clerk/html
  [:div.bg-gray-50.p-4.rounded.overflow-x-auto
   [:h3.font-bold.mb-4 "数据流维度变化"]
   [:table.w-full.text-sm
    [:thead
     [:tr.bg-gray-200
      [:th.p-2 "阶段"] [:th.p-2 "操作"] [:th.p-2 "输入形状"] [:th.p-2 "输出形状"]
      [:th.p-2 "参数量"] [:th.p-2 "计算类型"]]]
    [:tbody
     [:tr.border-b
      [:td.p-2.font-semibold "Input"] [:td.p-2 "Tokenization"]
      [:td.p-2.font-mono "'Hello world'"] [:td.p-2.font-mono "[2]"]
      [:td.p-2 "-"] [:td.p-2 "查找"]]
     [:tr.border-b.bg-blue-50
      [:td.p-2.font-semibold "Embedding"] [:td.p-2 "Lookup + Pos Encode"
      [:td.p-2.font-mono "[2]"] [:td.p-2.font-mono "[1, 2, 768]"]
      [:td.p-2 "39.4M"] [:td.p-2 "嵌入"]]
     (for [layer (range 1 4)]
       [:tr.border-b {:class (if (odd? layer) "bg-gray-50" "")}
        [:td.p-2.font-semibold (str "Layer " layer)]
        [:td.p-2 "Self-Attention + FFN"]
        [:td.p-2.font-mono "[1, 2, 768]"] [:td.p-2.font-mono "[1, 2, 768]"]
        [:td.p-2 "7.1M"] [:td.p-2 "线性 + 注意力"]])
     [:tr.border-b.bg-purple-50
      [:td.p-2.font-semibold "..."] [:td.p-2 "..."]
      [:td.p-2 "..."] [:td.p-2 "..."]
      [:td.p-2 "..."] [:td.p-2 "..."]]
     [:tr.border-b.bg-green-50
      [:td.p-2.font-semibold "Layer 12"] [:td.p-2 "Self-Attention + FFN"]
      [:td.p-2.font-mono "[1, 2, 768]"] [:td.p-2.font-mono "[1, 2, 768]"]
      [:td.p-2 "7.1M"] [:td.p-2 "线性 + 注意力"]]
     [:tr.border-b.bg-orange-50
      [:td.p-2.font-semibold "Output"] [:td.p-2 "LM Head"
      [:td.p-2.font-mono "[1, 2, 768]"] [:td.p-2.font-mono "[1, 2, 50257]"]
      [:td.p-2 "38.6M"] [:td.p-2 "线性投影"]]
     [:tr.bg-red-50
      [:td.p-2.font-semibold "Prediction"] [:td.p-2 "Argmax/Sampling"]
      [:td.p-2.font-mono "[1, 2, 50257]"] [:td.p-2.font-mono "token ID"
      [:td.p-2 "-"] [:td.p-2 "解码"]]]]])

;; ## 7. 层次化计算复杂度

;; 分析不同层次的计算复杂度：

^{::clerk/visibility :folded}
(defn complexity-by-layer
  "计算各层复杂度"
  [seq-len batch-size]
  (let [d 768
        d-ff (* 4 768)
        h 12]
    {:embedding {:operation "Lookup"
                 :flops (* batch-size seq-len d)
                 :memory (* batch-size seq-len d 4)} ; float32
     :attention {:qkv-matmul (* 3 batch-size seq-len d d)
                :attention-scores (* batch-size h seq-len seq-len (/ d h))
                :attention-apply (* batch-size h seq-len seq-len (/ d h))
                :output-matmul (* batch-size seq-len d d)}
     :feed-forward {:linear1 (* batch-size seq-len d d-ff)
                   :activation (* batch-size seq-len d-ff)
                   :linear2 (* batch-size seq-len d-ff d)}}))

;; 以 batch=1, seq=512 为例
(def complexity (complexity-by-layer 512 1))

(clerk/plotly
  {:data [{:x ["Q/K/V MatMul" "Attention Scores" "Attention Apply" "Output MatMul"
               "FFN Linear1" "FFN Activation" "FFN Linear2"]
          :y [(Math/log10 (:qkv-matmul (:attention complexity)))
             (Math/log10 (:attention-scores (:attention complexity)))
             (Math/log10 (:attention-apply (:attention complexity)))
             (Math/log10 (:output-matmul (:attention complexity)))
             (Math/log10 (:linear1 (:feed-forward complexity)))
             (Math/log10 (:activation (:feed-forward complexity)))
             (Math/log10 (:linear2 (:feed-forward complexity)))]
          :type "bar"
          :marker {:color ["#3B82F6" "#F59E0B" "#F59E0B" "#3B82F6"
                          "#10B981" "#8B5CF6" "#10B981"]}}]
   :layout {:title "各操作计算复杂度对比 (log₁₀ FLOPs, seq=512)"
            :yaxis {:title "log₁₀(FLOPs)"}
            :xaxis {:title "操作" :tickangle 45}}})

;; ## 8. 总结

;; GPT-2 的层次结构特点：

(clerk/table
  {:headers ["层次" "关键特征" "参数占比" "计算占比"]
   :rows [["Embeddings" "50257 × 768 词向量 + 位置编码" "33%" "低"]
          ["Attention" "12头自注意力，捕捉长距离依赖" "18%" "高 (O(n²))"]
          ["Feed-Forward" "768→3072→768 位置前馈" "36%" "中"]
          ["LayerNorm" "归一化稳定训练" "<1%" "低"]
          ["LM Head" "投影到词表空间" "33%" "低"]]})

;; ### 关键洞察

(clerk/html
  [:div.grid.grid-cols-2.gap-4.mt-4
   [:div.bg-blue-50.p-4.rounded
    [:h4.font-bold.text-blue-800 "浅而宽 vs 深而窄"]
    [:p.text-sm "GPT-2 (124M) 选择了中等深度 (12层) 和宽度 (768维)。"
     [:br]
     "更大的模型 (GPT-2 1.5B) 使用 48 层和 1600 维。"]]
   [:div.bg-green-50.p-4.rounded
    [:h4.font-bold.text-green-800 "计算瓶颈"]
    [:p.text-sm "注意力计算是主要瓶颈，复杂度为 O(n²)。"
     [:br]
     "FFN 虽然参数多，但计算是 O(n)。"]]
   [:div.bg-purple-50.p-4.rounded
    [:h4.font-bold.text-purple-800 "残差连接"]
    [:p.text-sm "每个子层都有残差连接，帮助梯度流动，"
     [:br]
     "使得训练深层网络成为可能。"]]
   [:div.bg-orange-50.p-4.rounded
    [:h4.font-bold.text-orange-800 "权重共享"]
    [:p.text-sm "Token embedding 和 LM head 可以共享权重，"
     [:br]
     "减少参数量并提高泛化。"]]])
