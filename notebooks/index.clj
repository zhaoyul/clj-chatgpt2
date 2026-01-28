;; # 🧠 GPT-2 模型可视化分析
;; 
;; 使用 Clerk Notebook 深入探索 GPT-2 语言模型的内部结构

(ns index
  (:require [nextjournal.clerk :as clerk]))

;; ## 📚 Notebook 目录

;; 本项目包含以下可视化分析：

(clerk/html
  [:div.grid.grid-cols-1.gap-6.mt-6
   
   ;; Notebook 1
   [:a.block.bg-gradient-to-r.from-blue-500.to-blue-600.text-white.p-6.rounded-lg.shadow-lg.hover:shadow-xl.transition-shadow
    {:href "./model_architecture"}
    [:div.flex.items-start.justify-between
     [:div
      [:h2.font-bold.text-2xl.mb-2 "🏗️ 模型架构分析"]
      [:p.text-blue-100 "深入解析 GPT-2 的整体架构，包括神经网络分层、Transformer 结构和参数统计"]
      [:div.mt-4.flex.gap-2
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "架构图"]
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "参数分析"]
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "ONNX 结构"]]]
     [:div.text-4xl "→"]]]
   
   ;; Notebook 2
   [:a.block.bg-gradient-to-r.from-purple-500.to-purple-600.text-white.p-6.rounded-lg.shadow-lg.hover:shadow-xl.transition-shadow
    {:href "./attention_mechanism"}
    [:div.flex.items-start.justify-between
     [:div
      [:h2.font-bold.text-2xl.mb-2 "🎯 注意力机制解析"]
      [:p.text-purple-100 "探索 Transformer 的核心：自注意力机制、多头注意力、因果掩码和计算细节"]
      [:div.mt-4.flex.gap-2
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "自注意力"]
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "多头机制"]
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "可视化"]]]
     [:div.text-4xl "→"]]]
   
   ;; Notebook 3
   [:a.block.bg-gradient-to-r.from-green-500.to-green-600.text-white.p-6.rounded-lg.shadow-lg.hover:shadow-xl.transition-shadow
    {:href "./layer_visualization"}
    [:div.flex.items-start.justify-between
     [:div
      [:h2.font-bold.text-2xl.mb-2 "🔬 神经网络分层"]
      [:p.text-green-100 "微观视角：权重矩阵、激活函数、层归一化和信息流动可视化"]
      [:div.mt-4.flex.gap-2
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "权重可视化"]
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "GELU"]
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "数据流"]]]
     [:div.text-4xl "→"]]]
   
   ;; Notebook 4 - 新增
   [:a.block.bg-gradient-to-r.from-red-500.to-red-600.text-white.p-6.rounded-lg.shadow-lg.hover:shadow-xl.transition-shadow
    {:href "./real_weights"}
    [:div.flex.items-start.justify-between
     [:div
      [:h2.font-bold.text-2xl.mb-2 "🎯 真实权重可视化"]
      [:p.text-red-100 "从 ONNX 模型提取的真实 GPT-2 权重参数：LayerNorm、Attention Bias、MLP 参数"]
      [:div.mt-4.flex.gap-2
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "真实数据"]
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "参数分布"]
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "统计分析"]]]
     [:div.text-4xl "→"]]]
   
   ;; Notebook 5 - 问答功能
   [:a.block.bg-gradient-to-r.from-orange-500.to-orange-600.text-white.p-6.rounded-lg.shadow-lg.hover:shadow-xl.transition-shadow
    {:href "./qa_demo"}
    [:div.flex.items-start.justify-between
     [:div
      [:h2.font-bold.text-2xl.mb-2 "🤖 问答功能演示"]
      [:p.text-orange-100 "GPT-2 问答功能展示：Prompt Engineering、解码策略对比、温度参数影响"]
      [:div.mt-4.flex.gap-2
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "Q&A"]
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "Prompt工程"]
       [:span.bg-white.bg-opacity-20.px-3.py-1.rounded-full.text-sm "交互演示"]]]
     [:div.text-4xl "→"]]]])

;; ## 🚀 快速开始

;; ### 启动 Clerk 笔记本服务器

;; ```bash
;; # 方式 1: 使用 Clojure CLI
;; clojure -M:clerk
;;
;; # 方式 2: 使用 REPL
;; clojure -M
;; user=> (require '[nextjournal.clerk :as clerk])
;; user=> (clerk/serve! {:browse? true :watch-paths ["notebooks"]})
;; ```

;; 然后在浏览器中访问：
;; - 本页面: http://localhost:7777
;; - 模型架构: http://localhost:7777/notebooks/model_architecture

;; ## 📊 GPT-2 关键指标

(clerk/html
  [:div.grid.grid-cols-4.gap-4.mt-6
   [:div.bg-gray-50.p-4.rounded.text-center
    [:div.text-3xl.font-bold.text-blue-600 "124M"]
    [:div.text-sm.text-gray-600 "参数量"]]
   [:div.bg-gray-50.p-4.rounded.text-center
    [:div.text-3xl.font-bold.text-green-600 "12"]
    [:div.text-sm.text-gray-600 "Transformer 层"]]
   [:div.bg-gray-50.p-4.rounded.text-center
    [:div.text-3xl.font-bold.text-purple-600 "768"]
    [:div.text-sm.text-gray-600 "隐藏层维度"]]
   [:div.bg-gray-50.p-4.rounded.text-center
    [:div.text-3xl.font-bold.text-orange-600 "12"]
    [:div.text-sm.text-gray-600 "注意力头数"]]])

;; ## 🔍 分析内容概览

;; | Notebook | 主要内容 | 可视化类型 |
;; |----------|---------|-----------|
;; | 模型架构 | 整体结构、参数分布、ONNX 图 | 树状图、饼图、柱状图 |
;; | 注意力机制 | 自注意力、多头注意力、因果掩码 | 热力图、矩阵图、流程图 |
;; | 神经网络层 | 权重矩阵、激活函数、信息流动 | 层次图、曲线图、表格 |

;; ## 📝 技术栈

;; - **Clerk**: 交互式 Clojure 笔记本
;; - **Plotly**: 数据可视化
;; - **DJL**: 深度学习模型加载
;; - **ONNX**: 模型格式解析

;; ---

;; 开始探索 → 点击上方的 Notebook 卡片
