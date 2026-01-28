(require '[nextjournal.clerk :as clerk])

(println "🚀 正在启动 Clerk Notebook 服务器...")
(println "")

(def server
  (clerk/serve! {:browse? true 
                 :watch-paths ["notebooks"]
                 :port 7777}))

(println "✅ Clerk server started!")
(println "")
(println "📚 Notebook URLs:")
(println "   Homepage:     http://localhost:7777/notebooks/index")
(println "   Architecture: http://localhost:7777/notebooks/model_architecture")
(println "   Attention:    http://localhost:7777/notebooks/attention_mechanism")
(println "   Layers:       http://localhost:7777/notebooks/layer_visualization")
(println "")
(println "Press Ctrl+C to stop")

;; 保持进程运行
@(promise)
