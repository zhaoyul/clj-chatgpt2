#!/usr/bin/env python3
"""
GPT-2 模型设置脚本
自动下载并导出 ONNX 模型

用法:
  python scripts/setup_model.py [--model gpt2] [--output resources/onnx/]
"""

import argparse
import os
import sys
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装"""
    required = {
        'transformers': 'transformers>=4.30.0',
        'torch': 'torch>=2.0.0',
        'onnx': 'onnx>=1.14.0',
    }
    
    missing = []
    for pkg, install_name in required.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(install_name)
    
    if missing:
        print("❌ 缺少依赖包:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n请安装依赖:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    
    import transformers
    import torch
    print(f"✅ transformers {transformers.__version__}")
    print(f"✅ torch {torch.__version__}")

def download_tokenizer(model_name: str, output_dir: Path):
    """下载分词器配置文件"""
    from transformers import GPT2Tokenizer
    
    print(f"\n📥 正在下载分词器: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 分词器已保存到: {output_dir}")
    
    # 列出下载的文件
    for f in output_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size
            print(f"   - {f.name} ({size/1024:.1f} KB)" if size > 1024 else f"   - {f.name} ({size} B)")

def export_onnx(model_name: str, output_dir: Path):
    """导出 ONNX 模型"""
    import torch
    from transformers import GPT2LMHeadModel
    
    print(f"\n📥 正在加载模型: {model_name}")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    # 创建包装器
    class GPT2Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False
            )
            return outputs.logits
    
    wrapped_model = GPT2Wrapper(model)
    wrapped_model.eval()
    
    # 准备示例输入
    dummy_input_ids = torch.randint(0, 50257, (1, 10))
    dummy_attention_mask = torch.ones(1, 10, dtype=torch.long)
    
    # 导出路径
    onnx_path = output_dir / "model.onnx"
    
    print(f"\n🔧 正在导出 ONNX 模型...")
    print(f"   输出路径: {onnx_path}")
    
    # 动态轴配置
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length", 2: "vocab_size"}
    }
    
    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )
    
    # 显示文件大小
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"✅ ONNX 模型导出成功!")
    print(f"   文件大小: {size_mb:.1f} MB")
    
    return str(onnx_path)

def verify_model(onnx_path: Path):
    """验证导出的模型"""
    try:
        import onnx
        import onnxruntime as ort
        
        print("\n🔍 验证 ONNX 模型...")
        
        # 检查模型结构
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("   ✅ 模型结构检查通过")
        
        # 测试推理
        session = ort.InferenceSession(str(onnx_path))
        print(f"   ✅ 输入: {[i.name for i in session.get_inputs()]}")
        print(f"   ✅ 输出: {[o.name for o in session.get_outputs()]}")
        print("   ✅ ONNX Runtime 可以正常加载模型")
        
    except ImportError:
        print("   ⚠️  未安装 onnx/onnxruntime，跳过验证")
    except Exception as e:
        print(f"   ⚠️  验证失败: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="下载并导出 GPT-2 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认下载 gpt2 基础版 (124M)
  python scripts/setup_model.py
  
  # 下载更大的模型
  python scripts/setup_model.py --model gpt2-medium
  
  # 指定输出目录
  python scripts/setup_model.py --output /path/to/models
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="模型名称 (默认: gpt2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resources/onnx",
        help="输出目录 (默认: resources/onnx)"
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="跳过下载分词器"
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="跳过导出 ONNX 模型"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GPT-2 模型设置工具")
    print("=" * 60)
    
    # 检查依赖
    check_dependencies()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载分词器
    if not args.skip_tokenizer:
        download_tokenizer(args.model, output_dir)
    
    # 导出 ONNX
    if not args.skip_onnx:
        onnx_path = export_onnx(args.model, output_dir)
        verify_model(Path(onnx_path))
    
    print("\n" + "=" * 60)
    print("✅ 设置完成!")
    print("=" * 60)
    print(f"\n模型文件位置: {output_dir}")
    print("\n你现在可以:")
    print("  1. 运行测试: clojure -M:test")
    print("  2. 启动服务: clojure -M -m gpt2.server")
    print("  3. 启动 Clerk: ./scripts/clerk.sh")

if __name__ == "__main__":
    main()
