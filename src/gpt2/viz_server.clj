(ns gpt2.viz-server
  "可视化服务器 - 提供静态页面（需要配合 Python API 使用）"
  (:require [ring.adapter.jetty :refer [run-jetty]]
            [ring.middleware.resource :refer [wrap-resource]]
            [ring.middleware.content-type :refer [wrap-content-type]]
            [ring.middleware.not-modified :refer [wrap-not-modified]]
            [ring.util.response :refer [response redirect content-type]]))

(defn handler [request]
  (case (:uri request)
    "/" 
    (redirect "/attention-viz-dynamic.html")
    
    "/static"
    (redirect "/attention-viz.html")
    
    "/api/health"
    {:status 200
     :headers {"Content-Type" "application/json"}
     :body "{\"status\": \"ok\", \"service\": \"viz-server\", \"note\": \"Use Python API for real attention data\"}"}
    
    {:status 404
     :body "Not Found"}))

(def app
  (-> handler
      (wrap-resource "public")
      wrap-content-type
      wrap-not-modified))

(defn start-server
  "启动可视化服务器"
  [& {:keys [port] :or {port 8888}}]
  (println (str "🌐 Visualization server starting on http://localhost:" port))
  (println "   📊 Dynamic viz (needs Python API): http://localhost:" port "/attention-viz-dynamic.html")
  (println "   📊 Static viz: http://localhost:" port "/attention-viz.html")
  (println "")
  (println "To get real attention weights:")
  (println "   1. Start Python API: python scripts/attention_api.py")
  (println "   2. Open the dynamic visualization page")
  (println "")
  (run-jetty app {:port port :join? false}))

(defn -main [& args]
  (let [port (if (seq args)
               (Integer/parseInt (first args))
               8888)]
    (start-server :port port)
    @(promise)))
