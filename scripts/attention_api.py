#!/usr/bin/env python3
"""
GPT-2 注意力权重 API 服务器
提供真实的注意力权重数据供前端可视化
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局加载模型
print("[INFO] 正在加载 GPT-2 模型...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
model.eval()
print("[INFO] 模型加载完成")

@app.route("/api/attention", methods=["POST"])
def get_attention():
    """
    获取输入文本的注意力权重
    
    请求体: {"text": "Hello world", "layers": [0, 5, 11], "heads": [0, 3, 11]}
    返回: 注意力权重矩阵
    """
    try:
        data = request.get_json()
        text = data.get("text", "Hello world")
        layers = data.get("layers", list(range(12)))  # 默认所有层
        heads = data.get("heads", list(range(12)))    # 默认所有头
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 获取 tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # 推理
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
            all_attentions = outputs.attentions  # Tuple of 12 tensors
        
        # 提取指定层和头的注意力权重
        # all_attentions[i] shape: [batch, num_heads, seq_len, seq_len]
        attention_data = {}
        
        for layer_idx in layers:
            if layer_idx >= len(all_attentions):
                continue
                
            layer_attention = all_attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
            attention_data[f"layer_{layer_idx}"] = {}
            
            for head_idx in heads:
                if head_idx >= layer_attention.shape[0]:
                    continue
                    
                head_attention = layer_attention[head_idx].numpy()
                # 转换为列表以便 JSON 序列化
                attention_data[f"layer_{layer_idx}"][f"head_{head_idx}"] = 
                    head_attention.tolist()
        
        return jsonify({
            "tokens": tokens,
            "text": text,
            "num_layers": len(all_attentions),
            "num_heads": 12,
            "attention": attention_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-info", methods=["GET"])
def model_info():
    """获取模型信息"""
    return jsonify({
        "model": "gpt2",
        "num_layers": 12,
        "num_heads": 12,
        "hidden_size": 768,
        "vocab_size": 50257,
        "max_position": 1024
    })

@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    
    print(f"[INFO] 启动注意力权重 API 服务器...")
    print(f"[INFO] 地址: http://{args.host}:{args.port}")
    print(f"[INFO] API 端点:")
    print(f"       POST /api/attention - 获取注意力权重")
    print(f"       GET  /api/model-info - 模型信息")
    print(f"       GET  /health - 健康检查")
    
    app.run(host=args.host, port=args.port, debug=False)
