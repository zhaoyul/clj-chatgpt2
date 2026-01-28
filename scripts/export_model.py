#!/usr/bin/env python3
"""
GPT-2 模型导出脚本
将 Hugging Face 的 GPT-2 模型导出为 ONNX 格式

使用方法:
  python scripts/export_model.py [--model gpt2] [--output resources/onnx/]
"""

import argparse
import os
import sys
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import transformers
        import torch
        print(f"[INFO] transformers 版本: {transformers.__version__}")
        print(f"[INFO] torch 版本: {torch.__version__}")
    except ImportError as e:
        print(f"[ERROR] 缺少依赖: {e}")
        print("[INFO] 请安装依赖: pip install transformers torch")
        sys.exit(1)

def export_onnx(model_name: str, output_dir: str):
    """导出 ONNX 模型"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    
    print(f"[INFO] 正在加载模型: {model_name}")
    
    # 加载模型和分词器
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    model.eval()
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 准备示例输入
    dummy_input = "Hello, world!"
    inputs = tokenizer(dummy_input, return_tensors="pt")
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 导出 ONNX
    onnx_path = output_path / "model.onnx"
    
    print(f"[INFO] 正在导出 ONNX 模型到: {onnx_path}")
    
    # 定义输入和输出名称
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    # 动态轴配置（支持变长序列）
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }
    
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"[INFO] ONNX 模型导出成功!")
    print(f"[INFO] 模型路径: {onnx_path}")
    print(f"[INFO] 模型大小: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 保存分词器配置（供参考）
    tokenizer.save_pretrained(output_path)
    print(f"[INFO] 分词器配置已保存")
    
    return str(onnx_path)

def verify_onnx(onnx_path: str):
    """验证导出的 ONNX 模型"""
    try:
        import onnx
        import onnxruntime as ort
        
        print(f"[INFO] 正在验证 ONNX 模型...")
        
        # 检查模型结构
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("[INFO] ONNX 模型结构检查通过")
        
        # 测试推理
        session = ort.InferenceSession(onnx_path)
        print(f"[INFO] 输入名称: {[i.name for i in session.get_inputs()]}")
        print(f"[INFO] 输出名称: {[o.name for o in session.get_outputs()]}")
        print("[INFO] ONNX Runtime 可以正常加载模型")
        
    except ImportError:
        print("[WARN] 未安装 onnx 或 onnxruntime，跳过验证")
    except Exception as e:
        print(f"[WARN] ONNX 验证失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="导出 GPT-2 模型到 ONNX 格式")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Hugging Face 模型名称 (默认: gpt2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resources/onnx",
        help="输出目录 (默认: resources/onnx)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GPT-2 ONNX 模型导出工具")
    print("=" * 60)
    
    check_dependencies()
    
    try:
        onnx_path = export_onnx(args.model, args.output)
        verify_onnx(onnx_path)
        print("=" * 60)
        print("[SUCCESS] 模型导出完成!")
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] 导出失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
