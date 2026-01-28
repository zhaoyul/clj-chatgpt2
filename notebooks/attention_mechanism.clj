;; # 注意力机制深度解析
;; 
;; 本 Notebook 深入分析 Transformer 中的注意力机制，包括数学原理和可视化。

(ns attention-mechanism
  (:require [nextjournal.clerk :as clerk]
            [gpt2.token :as token]
            [clojure.string :as str]))

;; ## 1. 注意力机制基础

;; 注意力机制的核心思想是：**对每个位置，计算与所有其他位置的关联程度，并据此加权聚合信息**。

;; ### 1.1 缩放点积注意力公式

(clerk/tex "\\mathrm{Attention}(Q, K, V) = \\operatorname{softmax}\\left(\\frac{QK^{T}}{\\sqrt{d_k}}\\right)V")

;; 其中：
;; - **Q (Query)**: 查询向量，表示"我要找什么信息"
;; - **K (Key)**: 键向量，表示"我有什么信息"
;; - **V (Value)**: 值向量，表示"信息的具体内容"
;; - **d_k**: Key 向量的维度（GPT-2 中是 64）

;; ### 1.2 为什么需要缩放？

;; 当 d_k 较大时，点积的数值会变得很大，导致 softmax 进入梯度极小的区域。
;; 除以 √d_k 可以保持数值在合理范围内。

^{::clerk/visibility :folded}
(defn scaling-visualization
  "展示缩放因子的作用"
  []
  (let [dk 64
        sqrt-dk (Math/sqrt dk)]
    {:d_k dk
     :sqrt_dk sqrt-dk
     :example (for [dot-product [1 5 10 20 50 100]]
                {:original dot-product
                 :scaled (/ dot-product sqrt-dk)})}))

(def scaling (scaling-visualization))

(clerk/table
  {:headers ["原始点积" "缩放后 (√64=8)"]
   :rows (mapv #(vector (:original %) (format "%.2f" (:scaled %)))
               (:example scaling))})

;; ## 2. 自注意力计算过程

;; 以序列 "Hello world" 为例，展示自注意力的计算过程：

^{::clerk/visibility :folded}
(defn self-attention-example
  "展示自注意力计算示例"
  []
  (let [tokens ["Hello" "," "world" "!"]
        embeddings (for [i (range 4)]
                    (vec (repeatedly 768 #(-> (rand) (- 0.5) (* 2)))))]
    {:tokens tokens
     :seq-len (count tokens)
     :hidden-size 768
     :heads 12
     :head-dim 64}))

(def example (self-attention-example))

(clerk/html
  [:div.space-y-4
   [:div.bg-blue-50.p-4.rounded
    [:h4.font-bold.text-blue-800 "输入序列"]
    [:div.flex.gap-2
     (for [token (:tokens example)]
       [:span.bg-white.px-3.py-1.rounded.shadow.text-sm.font-mono token])]]
   [:div.bg-green-50.p-4.rounded
    [:h4.font-bold.text-green-800 "Step 1: 生成 Q, K, V 矩阵"]
    [:p.text-sm "通过三个独立的线性投影 (768 × 768):"]
    [:ul.list-disc.ml-5.text-sm.mt-2
     [:li "Q = X × W_q → shape: [4 × 768]"]
     [:li "K = X × W_k → shape: [4 × 768]"]
     [:li "V = X × W_v → shape: [4 × 768]"]]]
   [:div.bg-purple-50.p-4.rounded
    [:h4.font-bold.text-purple-800 "Step 2: 分割成多个头"]
    [:p.text-sm "将 768 维分割成 12 个 64 维的头:"]
    [:ul.list-disc.ml-5.text-sm.mt-2
     [:li "Q_h: [4 × 12 × 64]"]
     [:li "K_h: [4 × 12 × 64]"]
     [:li "V_h: [4 × 12 × 64]"]]]
   [:div.bg-orange-50.p-4.rounded
    [:h4.font-bold.text-orange-800 "Step 3: 计算注意力分数"]
    [:p.text-sm "对每个头计算:"]
    [:pre.bg-white.p-2.rounded.mt-2.text-xs.font-mono
     "scores = Q_h × K_h^T / √64\nshape: [4 × 64] × [64 × 4] = [4 × 4]"]]])

;; ## 3. 注意力分数矩阵可视化

;; 展示一个 4×4 的注意力分数矩阵：

^{::clerk/visibility :folded}
(defn generate-attention-matrix
  "生成模拟的注意力权重矩阵"
  [tokens]
  (let [n (count tokens)]
    ;; 模拟注意力权重（上三角为 0，因为是因果掩码）
    (for [i (range n)]
      (for [j (range n)]
        (cond
          (> j i) 0.0                    ; 未来信息，掩码为 0
          (= j i) (+ 0.5 (rand 0.3))     ; 自身注意力
          :else (/ (+ 0.2 (rand 0.3))    ; 历史信息
                   (inc (- i j))))))))

(def tokens ["Hello" "," "world" "!"])
(def attention-weights (generate-attention-matrix tokens))

;; ### 3.1 注意力热力图

(clerk/plotly
  {:data [{:z attention-weights
          :x tokens
          :y tokens
          :type "heatmap"
          :colorscale "Viridis"
          :showscale true}]
   :layout {:title "自注意力权重矩阵 (Single Head)"
            :xaxis {:title "Key (被注意的位置)" :side "top"}
            :yaxis {:title "Query (当前位置)"}
            :width 500
            :height 500
            :annotations (for [i (range (count tokens))
                              j (range (count tokens))
                              :when (<= j i)]
                          {:x (nth tokens j)
                           :y (nth tokens i)
                           :text (format "%.2f" (nth (nth attention-weights i) j))
                           :font {:size 12 :color "white"}
                           :showarrow false})}})

;; **观察：**
;; - 对角线数值较高（关注自身）
;; - 上三角为 0（因果掩码，不能看未来）
;; - "world" 对 "Hello" 有较高的注意力权重

;; ### 3.2 多头注意力可视化

;; GPT-2 使用 12 个注意力头，每个头关注不同的特征：

^{::clerk/visibility :folded}
(defn multi-head-visualization
  "模拟不同头的注意力模式"
  []
  (let [heads ["Head 1: 相邻词"
               "Head 2: 主语-谓语"
               "Head 3: 修饰关系"
               "Head 4: 句法依赖"
               "Head 5-12: 其他模式"]]
    {:heads heads
     :patterns (for [h (range 5)]
                (generate-attention-matrix tokens))}))

(def multi-head (multi-head-visualization))

(clerk/plotly
  {:data (for [i (range 4)]
          {:z (nth (:patterns multi-head) i)
           :x tokens
           :y tokens
           :type "heatmap"
           :colorscale "Blues"
           :name (str "Head " (inc i))
           :visible (if (= i 0) true "legendonly")})
   :layout {:title "多头注意力模式对比"
            :xaxis {:title "Key Position"}
            :yaxis {:title "Query Position"}
            :updatemenus [{:type "buttons"
                          :direction "left"
                          :buttons (for [i (range 4)]
                                   {:method "update"
                                    :label (str "Head " (inc i))
                                    :args [{:visible (vec (for [j (range 4)] (= i j)))}]})}]}})

;; ## 4. 因果掩码机制

;; GPT-2 使用因果（Causal）掩码确保自回归生成：

^{::clerk/visibility :folded}
(defn causal-mask-demo
  "展示因果掩码"
  [n]
  {:mask (for [i (range n)]
           (for [j (range n)]
             (if (<= j i) 1 0)))
   :description "上三角矩阵，防止看到未来信息"})

(def mask-4 (causal-mask-demo 4))

(clerk/html
  [:div.space-y-4
   [:div.bg-gray-50.p-4.rounded
    [:h4.font-bold "因果掩码矩阵 (4×4):"]
    [:table.border-collapse.mt-2
     [:tbody
      (for [row (:mask mask-4)]
        [:tr
         (for [cell row]
           [:td.border.px-3.py-2.text-center.font-mono
            {:class (if (= cell 1) "bg-blue-200" "bg-gray-200")}
            cell])])]]]
   [:div.bg-yellow-50.p-4.rounded
    [:h4.font-bold.text-yellow-800 "掩码作用说明:"]
    [:ul.list-disc.ml-5
     [:li "第 1 行: 只能看位置 0（自己）"]
     [:li "第 2 行: 可以看位置 0, 1（自己和前一个）"]
     [:li "第 3 行: 可以看位置 0, 1, 2"]
     [:li "第 4 行: 可以看位置 0, 1, 2, 3（全部历史）"]
     [:li.mt-2.text-red-600 "上三角为 0: 确保不能看到未来的 token"]]]])

;; ## 5. 注意力计算细节

;; ### 5.1 矩阵维度变化

(clerk/table
  {:headers ["步骤" "操作" "输入形状" "输出形状"]
   :rows [["1" "Linear(Q)" "[batch, seq, 768]" "[batch, seq, 768]"]
          ["2" "Split Heads" "[batch, seq, 768]" "[batch, 12, seq, 64]"]
          ["3" "Q × K^T" "[batch, 12, seq, 64] × [batch, 12, 64, seq]" "[batch, 12, seq, seq]"]
          ["4" "Scale (/√64)" "[batch, 12, seq, seq]" "[batch, 12, seq, seq]"]
          ["5" "Apply Mask" "[batch, 12, seq, seq]" "[batch, 12, seq, seq]"]
          ["6" "Softmax" "[batch, 12, seq, seq]" "[batch, 12, seq, seq]"]
          ["7" "× V" "[batch, 12, seq, seq] × [batch, 12, seq, 64]" "[batch, 12, seq, 64]"]
          ["8" "Concat Heads" "[batch, 12, seq, 64]" "[batch, seq, 768]"]
          ["9" "Linear Out" "[batch, seq, 768]" "[batch, seq, 768]"]]})

;; ### 5.2 计算复杂度

^{::clerk/visibility :folded}
(defn complexity-analysis
  "分析注意力机制的复杂度"
  [seq-len hidden-size num-heads]
  (let [head-dim (/ hidden-size num-heads)]
    {:self-attention
     {:qkv-projection (* 3 seq-len hidden-size hidden-size)
      :attention-scores (* seq-len seq-len hidden-size)
      :attention-application (* seq-len seq-len hidden-size)
      :output-projection (* seq-len hidden-size hidden-size)}
     :feed-forward
     {:first-linear (* seq-len hidden-size (* 4 hidden-size))
      :second-linear (* seq-len (* 4 hidden-size) hidden-size)}
     :total-per-layer
     (+ (* 4 seq-len hidden-size hidden-size)
        (* 2 seq-len seq-len hidden-size)
        (* 5 seq-len hidden-size hidden-size))}))

;; 以序列长度 1024，隐藏维度 768 为例：
(def complexity (complexity-analysis 1024 768 12))

(clerk/html
  [:div.space-y-4
   [:div.bg-blue-50.p-4.rounded
    [:h4.font-bold.text-blue-800 "自注意力层计算量 (seq=1024, hidden=768)"]
    [:ul.list-disc.ml-5.text-sm
     [:li "Q/K/V 投影: " [:span.font-mono (format "%,d" (:qkv-projection (:self-attention complexity))) " 次乘法"]]
     [:li "注意力分数: " [:span.font-mono (format "%,d" (:attention-scores (:self-attention complexity))) " 次乘法"]]
     [:li "注意力应用: " [:span.font-mono (format "%,d" (:attention-application (:self-attention complexity))) " 次乘法"]]
     [:li "输出投影: " [:span.font-mono (format "%,d" (:output-projection (:self-attention complexity))) " 次乘法"]]]]
   [:div.bg-red-50.p-4.rounded
    [:h4.font-bold.text-red-800 "关键观察"]
    [:p.text-sm "注意力分数计算的复杂度是 O(n²) 关于序列长度。"
     [:br]
     "这意味着对于长序列，注意力计算会成为瓶颈。"
     [:br]
     "例如："
     [:ul.list-disc.ml-5.mt-1
      [:li "seq=512: 注意力分数计算需要 2 亿次乘法"]
      [:li "seq=1024: 需要 8 亿次乘法 (4×)"]
      [:li "seq=2048: 需要 32 亿次乘法 (16×)"]]]]])

;; ## 6. 注意力模式分析

;; 不同类型的注意力头学习不同的语言模式：

(clerk/html
  [:div.grid.grid-cols-2.gap-4
   [:div.bg-gradient-to-br.from-blue-50.to-blue-100.p-4.rounded
    [:h4.font-bold.text-blue-800 "位置注意力 (Positional)"]
    [:p.text-sm "关注相邻的词，捕捉局部语法结构"
     [:br]
     [:span.text-xs.text-gray-600 "例: 'New York' 中的紧密关联"]]
    [:div.mt-2.h-24.bg-white.rounded.flex.items-center.justify-center
     [:span.text-xs "[热力图: 对角线附近高亮]"]]]
   [:div.bg-gradient-to-br.from-green-50.to-green-100.p-4.rounded
    [:h4.font-bold.text-green-800 "句法注意力 (Syntactic)"]
    [:p.text-sm "关注句法相关的词，如主语-谓语"
     [:br]
     [:span.text-xs.text-gray-600 "例: 'She runs' 中 She→runs 的关联"]]
    [:div.mt-2.h-24.bg-white.rounded.flex.items-center.justify-center
     [:span.text-xs "[热力图: 长距离依赖高亮]"]]]
   [:div.bg-gradient-to-br.from-purple-50.to-purple-100.p-4.rounded
    [:h4.font-bold.text-purple-800 "指代注意力 (Coreference)"]
    [:p.text-sm "关注代词和其指代的实体"
     [:br]
     [:span.text-xs.text-gray-600 "例: 'Alice said she...' 中 she→Alice"]]
    [:div.mt-2.h-24.bg-white.rounded.flex.items-center.justify-center
     [:span.text-xs "[热力图: 指代关系高亮]"]]]
   [:div.bg-gradient-to-br.from-orange-50.to-orange-100.p-4.rounded
    [:h4.font-bold.text-orange-800 "罕见模式 (Rare)"]
    [:p.text-sm "一些头的注意力模式不明显"
     [:br]
     [:span.text-xs.text-gray-600 "可能编码了复杂的语义关系"]]
    [:div.mt-2.h-24.bg-white.rounded.flex.items-center.justify-center
     [:span.text-xs "[热力图: 分散或不规则模式]"]]]])

;; ## 7. 实际推理示例

;; 让我们跟踪 "Hello world" 生成过程中注意力如何工作：

^{::clerk/visibility :folded}
(defn generation-process
  "展示生成过程的注意力"
  []
  (let [prompt "Hello"
        tokens (token/encode prompt)]
    {:prompt prompt
     :token-ids tokens
     :generation-steps
     [{:step 1
       :input "Hello"
       :tokens tokens
       :attention-focus "全部在历史 token 上"
       :next-token-predicted "world"}
      {:step 2
       :input "Hello world"
       :tokens (conj tokens 995) ; world 的 token id
       :attention-focus "Hello → 40%, world → 50%, 自身 → 10%"
       :next-token-predicted "!"}]}))

(def gen-process (generation-process))

(clerk/html
  [:div.space-y-4
   [:h4.font-bold "逐步生成过程:"]
   (for [step (:generation-steps gen-process)]
     [:div.border.p-4.rounded
      [:h5.font-bold (str "Step " (:step step))]
      [:p [:span.font-semibold "输入: "] (:input step)]
      [:p [:span.font-semibold "Token IDs: "] [:span.font-mono (str (:tokens step))]]
      [:p [:span.font-semibold "注意力分布: "] (:attention-focus step)]
      [:p [:span.font-semibold "预测下一个: "] [:span.text-blue-600.font-bold (:next-token-predicted step)]]])])

;; ## 8. 总结

;; 注意力机制是 Transformer 架构的核心创新：

(clerk/table
  {:headers ["特性" "说明"]
   :rows [["并行计算" "不同于 RNN，所有位置的注意力可以同时计算"]
          ["长距离依赖" "直接连接任意两个位置，距离不影响计算复杂度"]
          ["可解释性" "注意力权重可以可视化，理解模型关注什么"]
          ["灵活性" "多头机制允许模型学习多种不同的注意力模式"]
          ["计算成本" "O(n²) 复杂度，长序列时需要优化（如 KV Cache）"]]})

;; ### 关键公式回顾

(clerk/html
  [:div.bg-gray-50.p-6.rounded.font-serif.text-lg.space-y-4
   [:div
    [:strong "1. 缩放点积注意力:"]
    [:div.ml-4 "Attention(Q, K, V) = softmax(QK" [:sup "T"] "/√d" [:sub "k"] ")V"]]
   [:div
    [:strong "2. 多头注意力:"]
    [:div.ml-4 "MultiHead(Q, K, V) = Concat(head" [:sub "1"] ", ..., head" [:sub "h"] ")W" [:sup "O"]]
    [:div.ml-4.text-sm "where head" [:sub "i"] " = Attention(QW" [:sub "i"] [:sup "Q"] ", KW" [:sub "i"] [:sup "K"] ", VW" [:sub "i"] [:sup "V"] ")"]]
   [:div
    [:strong "3. 因果掩码:"]
    [:div.ml-4 "M" [:sub "ij"] " = 0 if j > i, else 1"]
    [:div.ml-4.text-sm "Attention(Q, K, V) = softmax((QK" [:sup "T"] " + M)/√d" [:sub "k"] ")V"]]])
