(ns gpt2.token-test
  "分词器测试"
  (:require [clojure.test :refer :all]
            [gpt2.token :as token]))

(deftest test-encode
  (testing "文本编码"
    (is (vector? (token/encode "Hello")))
    (is (= [15496] (token/encode "Hello")))
    (is (= [0] (token/encode "!")))
    (is (empty? (token/encode "")))
    (is (nil? (token/encode nil)))))

(deftest test-decode
  (testing "Token 解码"
    (is (= "Hello" (token/decode [15496])))
    (is (= "!" (token/decode [0])))
    (is (= "Hello, world!" (token/decode [15496 11 995 0])))
    (is (= "" (token/decode nil)))
    (is (= "" (token/decode [])))))

(deftest test-roundtrip
  (testing "编解码一致性"
    (let [texts ["Hello, world!"
                 "Clojure is awesome."
                 "GPT-2 model"
                 "123456"]]
      (doseq [text texts]
        (is (= text (token/decode (token/encode text))))))))

(deftest test-count-tokens
  (testing "Token 计数"
    (is (= 1 (token/count-tokens "Hello")))
    (is (= 4 (token/count-tokens "Hello, world!")))
    (is (= 0 (token/count-tokens "")))
    (is (= 0 (token/count-tokens nil)))))

(deftest test-eos-token
  (testing "结束标记"
    (is (= 50256 token/eos-token))))
