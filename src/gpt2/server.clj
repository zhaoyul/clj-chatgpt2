(ns gpt2.server
  "GPT-2 Web API 服务"
  (:require [reitit.ring :as ring]
            [reitit.ring.middleware.muuntaja :as muuntaja]
            [ring.adapter.jetty :refer [run-jetty]]
            [muuntaja.core :as m]
            [gpt2.generate :as generate]
            [gpt2.model :as model]
            [clojure.data.json :as json])
  (:gen-class))

(defn- response
  "构建标准 HTTP 响应"
  ([body] (response 200 body))
  ([status body]
   {:status status
    :headers {"Content-Type" "application/json"}
    :body (json/write-str body)}))

(defn- parse-int
  "安全地解析整数"
  [s default]
  (try
    (Integer/parseInt (str s))
    (catch Exception _ default)))

(defn- parse-float
  "安全地解析浮点数"
  [s default]
  (try
    (Float/parseFloat (str s))
    (catch Exception _ default)))

(defn health-handler
  "健康检查接口"
  [_request]
  (response {:status "ok"
             :model_loaded (model/model-loaded?)}))

(defn generate-handler
  "文本生成接口
   
   请求体:
     {
       :prompt string         ; 输入提示（必需）
       :max_tokens int        ; 最大生成 token 数（默认 50）
       :strategy string       ; 解码策略：greedy 或 top-k（默认 greedy）
       :k int                 ; Top-K 值（默认 50）
       :temperature float     ; 温度参数（默认 1.0）
     }"
  [request]
  (try
    (let [body (:body-params request {})
          prompt (:prompt body)
          max-tokens (parse-int (:max_tokens body) 50)
          strategy (keyword (or (:strategy body) "greedy"))
          k (parse-int (:k body) 50)
          temperature (parse-float (:temperature body) 1.0)]
      (if (empty? prompt)
        (response 400 {:error "Missing required parameter: prompt"})
        (let [result (generate/generate-text
                      prompt
                      :max-tokens max-tokens
                      :strategy strategy
                      :k k
                      :temperature temperature)]
          (response {:generated_text result
                     :prompt prompt
                     :params {:max_tokens max-tokens
                              :strategy strategy
                              :k k
                              :temperature temperature}}))))
    (catch Exception e
      (.printStackTrace e)
      (response 500 {:error "Generation failed"
                     :message (.getMessage e)}))))

(defn stream-handler
  "流式文本生成接口（SSE）"
  [request]
  (try
    (let [body (:body-params request {})
          prompt (:prompt body)
          max-tokens (parse-int (:max_tokens body) 50)
          strategy (keyword (or (:strategy body) "greedy"))
          k (parse-int (:k body) 50)
          temperature (parse-float (:temperature body) 1.0)]
      (if (empty? prompt)
        (response 400 {:error "Missing required parameter: prompt"})
        {:status 200
         :headers {"Content-Type" "text/event-stream"
                   "Cache-Control" "no-cache"
                   "Connection" "keep-alive"}
         :body (let [chunks (generate/generate-stream
                             prompt
                             :max-tokens max-tokens
                             :strategy strategy
                             :k k
                             :temperature temperature)]
                 (->> chunks
                      (map (fn [chunk]
                             (str "data: " (json/write-str chunk) "\n\n")))
                      (clojure.string/join)
                      (str "data: [DONE]\n\n")))}))
    (catch Exception e
      (.printStackTrace e)
      (response 500 {:error "Stream generation failed"
                     :message (.getMessage e)}))))

(def app
  "Ring 应用路由"
  (ring/ring-handler
   (ring/router
    [["/health" {:get health-handler}]
     ["/api"
      ["/generate" {:post generate-handler}]
      ["/stream" {:post stream-handler}]]]
    {:data {:muuntaja m/instance
            :middleware [muuntaja/format-middleware
                         ;; 添加 CORS 支持
                         (fn [handler]
                           (fn [request]
                             (let [response (handler request)]
                               (assoc-in response [:headers "Access-Control-Allow-Origin"] "*"))))]}})
   (ring/create-default-handler
    {:not-found (constantly (response 404 {:error "Not found"}))})))

(defonce ^:private server (atom nil))

(defn start-server
  "启动 Web 服务器"
  [& {:keys [port join?] :or {port 3000 join? false}}]
  (if @server
    (println "[WARN] 服务器已在运行")
    (do
      ;; 预加载模型
      (when-not (model/model-loaded?)
        (println "[INFO] 正在加载模型...")
        (try
          (model/get-model)
          (catch Exception e
            (println "[WARN] 模型加载失败，服务将在首次请求时重试")
            (.printStackTrace e))))
      ;; 启动服务器
      (reset! server (run-jetty app {:port port :join? join?}))
      (println (str "[INFO] 服务器已启动，端口: " port))
      (println (str "[INFO] API 端点: http://localhost:" port "/api/generate"))
      (println (str "[INFO] 健康检查: http://localhost:" port "/health")))))

(defn stop-server
  "停止 Web 服务器"
  []
  (when @server
    (.stop @server)
    (reset! server nil)
    (println "[INFO] 服务器已停止")))

(defn -main
  "程序入口"
  [& args]
  (let [port (if (seq args)
               (parse-int (first args) 3000)
               3000)]
    (start-server :port port :join? true)))

(comment
  ;; 启动服务器（REPL 调试）
  (start-server :port 3000)

  ;; 停止服务器
  (stop-server)

  ;; 测试生成接口
  (generate-handler
   {:body-params {:prompt "Hello, world!"
                  :max_tokens 20
                  :strategy "greedy"}}))
