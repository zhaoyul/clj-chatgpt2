#!/usr/bin/env python3
"""
提取 GPT-2 ONNX 模型的权重参数
"""

import onnx
import numpy as np
import json
import os
from pathlib import Path

def extract_weights(onnx_path, output_dir):
    """提取 ONNX 模型的所有权重参数"""
    print(f"[INFO] 加载 ONNX 模型: {onnx_path}")
    model = onnx.load(onnx_path)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    weights_dir = output_path / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    # 提取初始化器（权重）
    weights_info = []
    
    print(f"[INFO] 提取权重参数...")
    for init in model.graph.initializer:
        name = init.name
        dims = list(init.dims)
        
        # 转换为 numpy 数组
        arr = onnx.numpy_helper.to_array(init)
        
        # 只保存较小的权重用于可视化（避免文件过大）
        total_elements = np.prod(dims) if dims else 1
        
        weight_info = {
            "name": name,
            "shape": dims,
            "dtype": str(arr.dtype),
            "total_elements": int(total_elements),
            "saved": False
        }
        
        # 只保存形状较小的权重用于可视化
        # 限制：最多 10000 个元素，且维度不超过 2
        if total_elements <= 10000 and len(dims) <= 2 and total_elements > 0:
            # 保存为 numpy 文件
            np.save(weights_dir / f"{name.replace('/', '_')}.npy", arr)
            weight_info["saved"] = True
            
            # 同时保存为 JSON 方便查看
            if total_elements <= 1000:
                json_path = weights_dir / f"{name.replace('/', '_')}.json"
                with open(json_path, 'w') as f:
                    json.dump({
                        "name": name,
                        "shape": dims,
                        "data": arr.flatten().tolist()[:100]  # 只保存前100个值
                    }, f, indent=2)
        
        weights_info.append(weight_info)
    
    # 保存权重信息索引
    with open(output_path / "weights_index.json", 'w') as f:
        json.dump({
            "model_path": str(onnx_path),
            "total_weights": len(weights_info),
            "weights": weights_info
        }, f, indent=2)
    
    print(f"[INFO] 共找到 {len(weights_info)} 个权重参数")
    print(f"[INFO] 权重信息已保存到: {output_path / 'weights_index.json'}")
    print(f"[INFO] 权重文件已保存到: {weights_dir}")
    
    return weights_info

def print_weight_summary(weights_info):
    """打印权重摘要"""
    print("\n" + "="*60)
    print("权重参数摘要")
    print("="*60)
    
    # 按形状分组
    saved_weights = [w for w in weights_info if w["saved"]]
    
    print(f"\n可可视化的权重 ({len(saved_weights)} 个):")
    for w in saved_weights[:20]:  # 只显示前20个
        print(f"  - {w['name']}: {w['shape']} ({w['dtype']})")
    
    if len(saved_weights) > 20:
        print(f"  ... 还有 {len(saved_weights) - 20} 个")
    
    # 找出最大的权重
    largest = max(weights_info, key=lambda x: x["total_elements"])
    print(f"\n最大权重:")
    print(f"  - {largest['name']}: {largest['shape']} = {largest['total_elements']:,} 元素")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="提取 GPT-2 ONNX 模型权重")
    parser.add_argument("--model", default="resources/onnx/model.onnx", help="ONNX 模型路径")
    parser.add_argument("--output", default="resources/weights", help="输出目录")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[ERROR] 模型文件不存在: {args.model}")
        exit(1)
    
    weights = extract_weights(args.model, args.output)
    print_weight_summary(weights)
