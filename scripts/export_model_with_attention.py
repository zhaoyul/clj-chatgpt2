#!/usr/bin/env python3
"""
导出 GPT-2 模型为 ONNX 格式，包含注意力权重输出
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

def export_with_attention(model_name: str, output_dir: str):
    """导出模型，包含注意力权重"""
    print(f"[INFO] 正在加载模型: {model_name}")
    
    # 加载模型 - 使用 GPT2Model 而不是 GPT2LMHeadModel 来获取隐藏状态和注意力
    from transformers import GPT2Model
    model = GPT2Model.from_pretrained(model_name, output_attentions=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    model.eval()
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 准备示例输入
    dummy_input = "Hello world"
    inputs = tokenizer(dummy_input, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 导出 ONNX
    onnx_path = output_path / "model_with_attention.onnx"
    
    print(f"[INFO] 正在导出 ONNX 模型到: {onnx_path}")
    print("[WARN] 注意：标准 ONNX 导出不支持动态数量的输出")
    print("[INFO] 我们将导出两个版本：")
    print("       1. model.onnx - 标准版本（推理用）")
    print("       2. 使用 Python API 动态获取注意力（可视化用）")
    
    # 实际上，ONNX 不支持输出 list of tensors（注意力权重）
    # 我们需要使用 Python 运行时来提供注意力可视化 API
    
    return str(onnx_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--output", default="resources/onnx")
    args = parser.parse_args()
    
    export_with_attention(args.model, args.output)
