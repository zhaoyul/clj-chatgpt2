(ns gpt2.generate-test
  "生成算法测试"
  (:require [clojure.test :refer :all]
            [gpt2.generate :as generate]))

(deftest test-argmax
  (testing "贪婪选择"
    (is (= 0 (generate/argmax (float-array [1.0 0.5 0.3]))))
    (is (= 1 (generate/argmax (float-array [0.5 1.0 0.3]))))
    (is (= 2 (generate/argmax (float-array [0.3 0.5 1.0]))))
    (is (= -1 (generate/argmax (float-array []))))
    (is (= 0 (generate/argmax (float-array [0.5]))))))

(deftest test-top-k-sample
  (testing "Top-K 采样"
    (let [logits (float-array [0.1 0.2 0.3 0.4 0.5])
          results (repeatedly 100 #(generate/top-k-sample logits 3))]
      ;; Top-3 应该只返回索引 2, 3, 4
      (is (every? #(contains? #{2 3 4} %) results)))

    ;; K=1 时应该总是返回最大值索引
    (let [logits (float-array [0.1 0.5 0.3])]
      (is (= 1 (generate/top-k-sample logits 1))))))
