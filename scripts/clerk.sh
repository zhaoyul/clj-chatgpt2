#!/bin/bash
# Clerk Notebook 启动脚本

PORT=${1:-7788}

echo "🚀 启动 Clerk Notebook 服务器..."
echo "   Port: $PORT"
echo ""

clojure -M -e "
(require '[nextjournal.clerk :as clerk])
(println \"🚀 正在启动 Clerk Notebook 服务器...\")
(def server
  (clerk/serve! {:browse? true 
                 :watch-paths [\"notebooks\"]
                 :port $PORT}))
(println \"\")
(println \"✅ Clerk server started!\")
(println \"\")
(println \"📚 Notebook URLs:\")
(println \"   Homepage:     http://localhost:\$PORT/notebooks/index\")
(println \"   Architecture: http://localhost:\$PORT/notebooks/model_architecture\")
(println \"   Attention:    http://localhost:\$PORT/notebooks/attention_mechanism\")
(println \"   Layers:       http://localhost:\$PORT/notebooks/layer_visualization\")
(println \"   Real Weights: http://localhost:\$PORT/notebooks/real_weights\")
(println \"   QA Demo:      http://localhost:\$PORT/notebooks/qa_demo\")
(println \"\")
(println \"Press Ctrl+C to stop\")
@(promise)
"
