(ns gpt2.token
  "GPT-2 分词器封装，基于 JTokkit"
  (:import [com.knuddels.jtokkit Encodings]
           [com.knuddels.jtokkit.api EncodingType IntArrayList]))

(def ^:private registry
  "JTokkit 编码器注册表"
  (Encodings/newDefaultEncodingRegistry))

(def ^:private encoder
  "GPT-2 编码器 (r50k_base)"
  (.getEncoding registry EncodingType/R50K_BASE))

(def eos-token
  "GPT-2 结束标记 <|endoftext|>"
  50256)

(defn encode
  "将文本字符串转换为 token ID 向量

   参数:
     text - 输入文本字符串

   返回:
     token ID 的 vector"
  [text]
  (when text
    (let [tokens (.encode encoder text)]
      (vec (.toArray tokens)))))

(defn decode
  "将 token ID 序列解码回文本字符串

   参数:
     token-ids - token ID 序列 (vector 或 seq)

   返回:
     解码后的文本字符串"
  [token-ids]
  (if (seq token-ids)
    (let [int-list (IntArrayList. (count token-ids))]
      (doseq [id token-ids]
        (.add int-list (int id)))
      (.decode encoder int-list))
    ""))

(defn count-tokens
  "计算文本的 token 数量"
  [text]
  (count (encode text)))

(comment
  ;; 测试编码
  (encode "Hello, Clojure AI!")
  ;; => [15496 11 25445 616 9552 0]

  ;; 测试解码
  (decode [15496 11 25445 616 9552 0])
  ;; => "Hello, Clojure AI!"

  ;; 测试结束标记
  eos-token
  ;; => 50256

  ;; 测试 token 计数
  (count-tokens "Hello, world!")
  ;; => 4
  )
