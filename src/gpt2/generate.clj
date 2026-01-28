(ns gpt2.generate
  "文本生成与解码算法"
  (:require [gpt2.model :as model]
            [gpt2.token :as token]))

(def ^:private vocab-size
  "GPT-2 词表大小"
  50257)

(defn- extract-last-token-logits
  "从完整的 logits 数组中提取最后一个 token 的 logits

   参数:
     full-logits - 完整的 logits 数组 [batch_size, seq_len, vocab_size]
     seq-len - 序列长度

   返回:
     最后一个 token 的 logits 数组"
  [^floats full-logits seq-len]
  (let [start-idx (* (dec seq-len) vocab-size)]
    (java.util.Arrays/copyOfRange full-logits start-idx (+ start-idx vocab-size))))

(defn argmax
  "返回 float 数组中最大值的索引（贪婪选择）

   参数:
     float-array - 输入数组

   返回:
     最大值对应的索引"
  [^floats float-array]
  (let [len (alength float-array)]
    (if (zero? len)
      -1
      (loop [i 1
             max-idx 0
             max-val (aget float-array 0)]
        (if (>= i len)
          max-idx
          (let [val (aget float-array i)]
            (if (> val max-val)
              (recur (inc i) i val)
              (recur (inc i) max-idx max-val))))))))

(defn top-k-sample
  "Top-K 采样：从概率最高的 K 个 token 中随机选择

   参数:
     logits - 模型输出的 logits 数组
     k - 采样的候选数量
     temperature - 温度参数（可选，默认 1.0）

   返回:
     采样的 token ID"
  [^floats logits k & {:keys [temperature] :or {temperature 1.0}}]
  (let [len (alength logits)
        k (min k len)
        ;; 创建索引-值对并排序
        indexed (map-indexed vector logits)
        top-k (take k (sort-by (fn [[_ v]] (- v)) indexed))
        ;; 应用 temperature 并计算 softmax
        scaled (map (fn [[idx v]] [idx (/ v temperature)]) top-k)
        max-val (apply max (map second scaled))
        exp-sum (reduce + (map (fn [[_ v]] (Math/exp (- v max-val))) scaled))
        probs (map (fn [[idx v]]
                     [idx (/ (Math/exp (- v max-val)) exp-sum)])
                   scaled)
        ;; 累积概率采样
        rand-val (rand)
        cumsum (reductions + (map second probs))]
    (loop [idx 0]
      (if (or (>= idx (count cumsum))
              (< rand-val (nth cumsum idx)))
        (first (nth probs idx))
        (recur (inc idx))))))

(defn generate-text
  "使用贪婪搜索生成文本

   参数:
     prompt - 输入提示文本
     max-tokens - 最大生成 token 数量（默认 50）
     strategy - 解码策略 :greedy 或 :top-k（默认 :greedy）
     k - Top-K 采样的 K 值（仅在 :top-k 策略下使用，默认 50）
     temperature - 温度参数（默认 1.0）

   返回:
     生成的文本字符串"
  [prompt & {:keys [max-tokens strategy k temperature]
             :or {max-tokens 50
                  strategy :greedy
                  k 50
                  temperature 1.0}}]
  (when-not (model/model-loaded?)
    (model/get-model)) ;; 触发模型加载
  (let [predictor (model/create-predictor)]
    (try
      (let [start-ids (or (token/encode prompt) [])
            eos-token token/eos-token]
        (loop [current-ids start-ids
               steps 0]
          (if (>= steps max-tokens)
            (token/decode current-ids)
            (let [full-logits (model/forward-pass predictor current-ids)
                  seq-len (count current-ids)
                  logits (extract-last-token-logits full-logits seq-len)
                  next-token (case strategy
                               :greedy (argmax logits)
                               :top-k (top-k-sample logits k :temperature temperature)
                               (argmax logits))]
              (if (= next-token eos-token)
                (token/decode current-ids)
                (recur (conj current-ids next-token) (inc steps)))))))
      (finally
        (.close predictor)))))

(defn generate-stream
  "流式生成文本，返回 lazy sequence

   参数同 generate-text"
  [prompt & {:keys [max-tokens strategy k temperature]
             :or {max-tokens 50
                  strategy :greedy
                  k 50
                  temperature 1.0}}]
  (when-not (model/model-loaded?)
    (model/get-model))
  (let [predictor (model/create-predictor)]
    ((fn step [current-ids steps]
       (lazy-seq
        (if (>= steps max-tokens)
          (do (.close predictor) nil)
          (let [full-logits (model/forward-pass predictor current-ids)
                seq-len (count current-ids)
                logits (extract-last-token-logits full-logits seq-len)
                next-token (case strategy
                             :greedy (argmax logits)
                             :top-k (top-k-sample logits k :temperature temperature)
                             (argmax logits))]
            (if (= next-token token/eos-token)
              (do (.close predictor) nil)
              (let [next-ids (conj current-ids next-token)
                    text (token/decode [next-token])]
                (cons {:token next-token
                       :text text
                       :full-text (token/decode next-ids)}
                      (step next-ids (inc steps))))))))
       (or (token/encode prompt) []) 0))))

(comment
  ;; 贪婪搜索生成
  (generate-text "Hello" :max-tokens 10)

  ;; Top-K 采样生成
  (generate-text "Hello" :max-tokens 10 :strategy :top-k :k 40 :temperature 0.8)

  ;; 流式生成
  (doseq [chunk (take 5 (generate-stream "Hello" :max-tokens 10))]
    (println chunk)))
