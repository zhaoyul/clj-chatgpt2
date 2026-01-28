#!/bin/bash
# 启动 Clerk 展示单个 Notebook（不监视文件变化）

PORT=${1:-7788}
NOTEBOOK=${2:-"notebooks/real_weights.clj"}

echo "🚀 启动 Clerk 展示单个 Notebook..."
echo "   Port: $PORT"
echo "   Notebook: $NOTEBOOK"
echo ""

clojure -M -e "
(require '[nextjournal.clerk :as clerk])
(println \"🚀 正在启动 Clerk...\")
(def server
  (clerk/serve! {:browse? true 
                 :port $PORT}))
;; 显示指定文件
(clerk/show! \"$NOTEBOOK\")
(println \"\")
(println \"✅ Notebook 已加载!\")
(println \"   URL: http://localhost:$PORT\")
(println \"\")
@(promise)
"
