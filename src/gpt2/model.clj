(ns gpt2.model
  "GPT-2 模型加载与推理"
  (:require [clojure.java.io :as io]
            [gpt2.token :as token])
  (:import [ai.djl.repository.zoo Criteria ZooModel]
           [ai.djl.inference Predictor]
           [ai.djl.ndarray NDList NDManager]
           [ai.djl.ndarray.types Shape]
           [ai.djl.translate NoopTranslator]
           [java.nio.file Paths Path]))

(def ^:private model-path
  "ONNX 模型路径"
  "resources/onnx/model.onnx")

(defn- get-model-file
  "获取模型文件路径"
  ^Path []
  (Paths/get model-path (into-array String [])))

(defn build-criteria
  "构建模型加载条件"
  []
  (-> (Criteria/builder)
      (.setTypes NDList NDList)
      (.optModelPath (get-model-file))
      (.optModelName "model.onnx")
      (.optEngine "OnnxRuntime")
      (.optTranslator (NoopTranslator.))
      (.build)))

(def ^:private gpt-model
  "延迟加载的 GPT-2 模型"
  (delay
    (try
      (let [criteria (build-criteria)
            model (.loadModel criteria)]
        (println "[INFO] GPT-2 模型加载成功")
        model)
      (catch Exception e
        (println "[ERROR] 模型加载失败:" (.getMessage e))
        (println "[INFO] 请确保已导出 ONNX 模型到" model-path)
        (throw e)))))

(defn get-model
  "获取加载好的模型实例"
  []
  @gpt-model)

(defn create-predictor
  "创建新的预测器实例

   注意：Predictor 不是线程安全的，每个线程应该有自己的实例"
  ^Predictor []
  (.newPredictor (get-model)))

(defn- create-input-tensors
  "创建输入张量"
  [^NDManager manager input-ids]
  (let [seq-len (count input-ids)
        input-array (.create manager (long-array input-ids)
                             (Shape. (long-array [1 seq-len])))
        mask-array (.create manager (long-array (repeat seq-len 1))
                             (Shape. (long-array [1 seq-len])))]
    (doto (NDList.) (.add input-array) (.add mask-array))))

(defn forward-pass
  "执行一次前向传播

   参数:
     predictor - Predictor 实例
     input-ids - 输入 token ID 序列

   返回:
     最后一个 token 的 logits float 数组"
  [^Predictor predictor input-ids]
  (let [manager (NDManager/newBaseManager)]
    (try
      (let [inputs (create-input-tensors manager input-ids)
            outputs (.predict predictor inputs)
            logits-tensor (.get outputs 0)
            ;; 返回完整的 logits 数组
            ;; 调用者需要根据序列长度自行提取最后一个 token 的 logits
            result (.toFloatArray logits-tensor)]
        ;; 释放中间张量
        (.close inputs)
        (.close outputs)
        result)
      (finally
        (.close manager)))))

(defn model-loaded?
  "检查模型是否已加载"
  []
  (realized? gpt-model))

(comment
  ;; 测试模型加载
  (get-model)

  ;; 测试推理
  (let [predictor (create-predictor)
        input-ids (token/encode "Hello")]
    (try
      (let [logits (forward-pass predictor input-ids)]
        (println "Logits 长度:" (alength logits))
        (println "前10个值:" (take 10 logits)))
      (finally
        (.close predictor)))))
