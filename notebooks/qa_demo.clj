;; # 🤖 GPT-2 问答功能演示
;; 
;; 本 Notebook 演示如何使用 GPT-2 进行问答。
;; 注意：GPT-2 是通用文本生成模型，不是专门的问答模型，
;; 通过 Prompt Engineering 可以实现问答功能。

(ns qa-demo
  (:require [nextjournal.clerk :as clerk]
            [gpt2.qa :as qa]
            [clojure.string :as str]))

;; ## 1. 问答系统概述

(clerk/html
  [:div.bg-blue-50.p-6.rounded-lg
   [:h2.text-2xl.font-bold.text-blue-800.mb-4 "📋 GPT-2 问答功能"]
   [:ul.list-disc.ml-6.space-y-2.text-gray-700
    [:li [:strong "模型基础"] " - GPT-2 (124M 参数)"]
    [:li [:strong "实现方式"] " - Prompt Engineering（提示工程）"]
    [:li [:strong "解码策略"] " - Top-K 采样 + 温度调节"]
    [:li [:strong "适用场景"] " - 开放式问答、文本补全、创意生成"]]
   [:div.mt-4.p-3.bg-yellow-100.rounded.text-sm.text-yellow-800
    "⚠️ 注意：GPT-2 可能生成不准确的信息，不适合事实性要求严格的场景。"]])

;; ## 2. 基础问答示例

;; 定义示例问题
(def sample-questions
  ["What is machine learning?"
   "How does photosynthesis work?"
   "What are the benefits of reading?"])

;; 生成答案（小批量，避免加载时间过长）
^{::clerk/visibility :folded}
(def sample-answers
  (try
    (doall
      (for [q (take 2 sample-questions)]
        (assoc (qa/answer q :max-tokens 40 :temperature 0.7)
               :status :success)))
    (catch Exception e
      [{:status :error :message (ex-message e)}])))

;; 展示问答结果
(clerk/html
  [:div.space-y-6
   [:h3.text-xl.font-bold "基础问答示例"]
   (for [ans sample-answers]
     (if (= :success (:status ans))
       [:div.border-l-4.border-blue-500.pl-4.py-2
        [:p.font-semibold.text-gray-800 (str "Q: " (:question ans))]
        [:p.text-gray-600.mt-1 (str "A: " (:answer ans))]]
       [:div.bg-red-50.p-4.rounded.text-red-600
        "模型加载失败或运行错误"]))])

;; ## 3. Prompt 模板对比

;; 使用不同模板生成答案
^{::clerk/visibility :folded}
(def template-comparison
  (try
    (let [question "What is artificial intelligence?"]
      {:question question
       :results (for [template [:default :detailed :creative]]
                  {:template template
                   :response (qa/answer question 
                                        :template template 
                                        :max-tokens 40
                                        :temperature 0.8)})})
    (catch Exception e
      {:error (ex-message e)})))

;; 展示模板对比
(if (:error template-comparison)
  (clerk/html [:div.bg-red-50.p-4.rounded "加载失败"])
  (clerk/html
    [:div.space-y-4
     [:h3.text-xl.font-bold (str "问题: " (:question template-comparison))]
     (for [result (:results template-comparison)]
       [:div.bg-gray-50.p-4.rounded
        [:h4.font-semibold.text-blue-700 
         (str "模板: " (name (:template result)))]
        [:p.mt-2.text-gray-700 
         (:answer (:response result))]])]))

;; ## 4. 解码策略对比

;; 贪婪搜索 vs Top-K 采样
^{::clerk/visibility :folded}
(def strategy-comparison
  (try
    (let [question "The future of AI is"]
      {:question question
       :greedy (qa/answer question 
                          :strategy :greedy 
                          :max-tokens 30
                          :template :creative)
       :top-k (qa/answer question 
                         :strategy :top-k 
                         :temperature 0.9
                         :max-tokens 30
                         :template :creative)})
    (catch Exception e
      {:error (ex-message e)})))

;; 展示策略对比
(if (:error strategy-comparison)
  (clerk/html [:div.bg-red-50.p-4.rounded "加载失败"])
  (clerk/html
    [:div.space-y-4
     [:h3.text-xl.font-bold (str "提示: " (:question strategy-comparison))]
     [:div.grid.grid-cols-2.gap-4
      [:div.bg-green-50.p-4.rounded
       [:h4.font-semibold.text-green-700 "贪婪搜索 (Greedy)"]
       [:p.mt-2.text-sm.text-gray-700 
        (:answer (:greedy strategy-comparison))]]
      [:div.bg-purple-50.p-4.rounded
       [:h4.font-semibold.text-purple-700 "Top-K 采样 (Temp=0.9)"]
       [:p.mt-2.text-sm.text-gray-700 
        (:answer (:top-k strategy-comparison))]]]]))

;; ## 5. 温度参数影响

;; 展示不同温度参数的效果
^{::clerk/visibility :folded}
(def temperature-demo
  (try
    (let [question "Once upon a time"]
      {:question question
       :results (for [temp [0.3 0.7 1.2]]
                  {:temperature temp
                   :response (qa/answer question 
                                        :temperature temp
                                        :max-tokens 30
                                        :template :creative)})})
    (catch Exception e
      {:error (ex-message e)})))

;; 温度参数可视化
(if (:error temperature-demo)
  (clerk/html [:div.bg-red-50.p-4.rounded "加载失败"])
  (clerk/html
    [:div.space-y-4
     [:h3.text-xl.font-bold (str "创意生成: " (:question temperature-demo))]
     [:div.space-y-3
      (for [result (:results temperature-demo)]
        [:div.flex.gap-4.items-start
         [:div.w-20.shrink-0
          [:span.inline-block.px-2.py-1.bg-blue-100.text-blue-800.rounded.text-sm.font-mono
           (str "T=" (:temperature result))]]
         [:div.flex-1.bg-gray-50.p-3.rounded.text-gray-700
          (:answer (:response result))]])]]))

;; ## 6. 问答系统参数说明

(clerk/table
  {:headers ["参数" "说明" "推荐值" "影响"]
   :rows [["max-tokens" "最大生成 token 数" "30-100" "控制回答长度"]
          ["strategy" "解码策略" ":top-k" "贪婪更确定，采样更多样"]
          ["temperature" "温度参数" "0.5-0.9" "低值更确定，高值更创意"]
          ["template" "Prompt 模板" ":default/:creative" "影响回答风格"]
          ["k" "Top-K 采样 K 值" "40-50" "候选词数量"]]})

;; ## 7. 使用代码示例

(clerk/html
  [:div.bg-gray-900.text-gray-100.p-4.rounded-lg.font-mono.text-sm.overflow-x-auto
   [:pre
    ";; 基础问答\n"
    "(require '[gpt2.qa :as qa])\n\n"
    "(qa/answer \"What is Clojure?\"\n"
    "          :max-tokens 40\n"
    "          :temperature 0.7)\n\n"
    ";; 使用特定模板\n"
    "(qa/answer \"Explain recursion\"\n"
    "          :template :detailed\n"
    "          :max-tokens 60)\n\n"
    ";; 批量问答\n"
    "(qa/batch-qa [\"Q1?\" \"Q2?\" \"Q3?\"]\n"
    "             :max-tokens 20)"]])

;; ## 8. 限制与注意事项

(clerk/html
  [:div.grid.grid-cols-2.gap-4
   [:div.bg-red-50.p-4.rounded
    [:h4.font-bold.text-red-800 "⚠️ 局限性"]
    [:ul.list-disc.ml-5.text-sm.text-gray-700.space-y-1
     [:li "可能生成不准确的事实"]
     [:li "没有真正的理解能力"]
     [:li "知识截止于训练数据时间"]
     [:li "对数学和逻辑推理能力有限"]]]
   [:div.bg-green-50.p-4.rounded
    [:h4.font-bold.text-green-800 "✅ 适用场景"]
    [:ul.list-disc.ml-5.text-sm.text-gray-700.space-y-1
     [:li "开放式创意写作"]
     [:li "文本补全和扩展"]
     [:li "概念解释（需验证）"]
     [:li "对话和交互体验"]]]])

;; ---
;; 
;; **API 端点**: 也可以通过 REST API 访问问答功能
;; 
;; ```bash
;; curl -X POST http://localhost:3000/api/generate \
;;   -H "Content-Type: application/json" \
;;   -d '{"prompt": "Q: What is AI?\nA:", "max_tokens": 50}'
;; ```
