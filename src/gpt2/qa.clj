(ns gpt2.qa
  "GPT-2 问答功能模块
   
   使用 prompt engineering 技术实现问答功能。
   GPT-2 是生成模型，不是专门的问答模型，
   需要通过精心设计的 prompt 来引导生成答案。"
  (:require [gpt2.generate :as generate]
            [gpt2.token :as token]))

(def ^:private qa-templates
  "问答 prompt 模板集合"
  {:default "Q: %s\nA:"
   :detailed "Question: %s\n\nAnswer:"
   :conversation "Human: %s\n\nAssistant:"
   :contextual "Context: %s\n\nQuestion: %s\n\nAnswer:"
   :factual "Fact: %s\n\nQ: What can you tell me about this?\nA:"
   :creative "Topic: %s\n\nLet me explain this interesting topic:\n\n"})

(defn- build-prompt
  "构建问答 prompt"
  [question & {:keys [template context]
               :or {template :default}}]
  (let [template-str (get qa-templates template (:default qa-templates))]
    (if (= template :contextual)
      (format template-str context question)
      (format template-str question))))

(defn answer
  "生成问题的答案
   
   参数:
     question - 问题文本
     max-tokens - 最大生成长度（默认 100）
     strategy - 解码策略 :greedy 或 :top-k
     temperature - 温度参数（默认 0.7，较低使输出更确定）
     template - prompt 模板类型
     context - 上下文（仅用于 :contextual 模板）
   
   返回:
     {:question 原始问题
      :answer 生成的答案
      :full-text 完整生成文本
      :params 使用的参数}"
  [question & {:keys [max-tokens strategy temperature template context]
               :or {max-tokens 100
                    strategy :top-k
                    temperature 0.7
                    template :default}}]
  (let [prompt (build-prompt question :template template :context context)
        full-text (generate/generate-text
                    prompt
                    :max-tokens max-tokens
                    :strategy strategy
                    :k 50
                    :temperature temperature)
        ;; 提取答案部分（去掉 prompt）
        prompt-len (count prompt)
        answer-text (clojure.string/trim
                      (subs full-text prompt-len))]
    {:question question
     :answer answer-text
     :full-text full-text
     :prompt prompt
     :params {:max-tokens max-tokens
              :strategy strategy
              :temperature temperature
              :template template}}))

(defn answer-stream
  "流式生成答案
   
   参数同 answer，返回 lazy sequence"
  [question & {:keys [max-tokens strategy temperature template context]
               :or {max-tokens 100
                    strategy :top-k
                    temperature 0.7
                    template :default}}]
  (let [prompt (build-prompt question :template template :context context)
        prompt-len (count prompt)]
    (->> (generate/generate-stream
           prompt
           :max-tokens max-tokens
           :strategy strategy
           :k 50
           :temperature temperature)
         (drop-while #(<= (count (:full-text %)) prompt-len))
         (map #(assoc % :answer (subs (:full-text %) prompt-len))))))

(defn batch-qa
  "批量问答
   
   参数:
     questions - 问题列表
     其他参数同 answer
   
   返回:
     问答结果列表"
  [questions & args]
  (mapv #(apply answer % args) questions))

;; 预设的问答示例
(def demo-questions
  ["What is machine learning?"
   "How does photosynthesis work?"
   "What are the benefits of exercise?"
   "Explain quantum computing in simple terms."
   "What is the capital of France?"])

(defn run-demos
  "运行演示问答"
  [& {:keys [max-tokens] :or {max-tokens 50}}]
  (println "=== GPT-2 问答演示 ===\n")
  (doseq [q demo-questions]
    (println (str "Q: " q))
    (let [result (answer q :max-tokens max-tokens :temperature 0.7)]
      (println (str "A: " (:answer result)))
      (println ""))
    (Thread/sleep 100)))  ; 避免输出过快

(comment
  ;; 基础问答
  (answer "What is Clojure?" :max-tokens 30)

  ;; 使用不同模板
  (answer "What is AI?" :template :detailed :max-tokens 50)
  (answer "Tell me about space" :template :creative :max-tokens 80)

  ;; 带上下文的问答
  (answer "What is it used for?"
          :template :contextual
          :context "Clojure is a modern Lisp dialect that runs on the JVM."
          :max-tokens 40)

  ;; 流式问答
  (doseq [chunk (take 10 (answer-stream "What is functional programming?"))]
    (print (:text chunk))
    (flush))

  ;; 批量问答
  (batch-qa ["What is 2+2?" "Who invented the telephone?"] :max-tokens 20)

  ;; 运行演示
  (run-demos :max-tokens 30))
