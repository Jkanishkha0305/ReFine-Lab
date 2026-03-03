"""
Quantization and On-Device Deployment for RV-GRPO Models

Converts trained models to GGUF format for on-device deployment
via llama.cpp or MLC-LLM.

Usage:
    python deployment/quantize.py \
        --model_path ./outputs/rv_grpo/qwen/final \
        --model_id Qwen/Qwen2.5-1.5B-Instruct \
        --output_dir ./deployment/gguf

    python deployment/quantize.py \
        --model_path ./outputs/rv_grpo/llama/final \
        --model_id meta-llama/Llama-3.2-1B-Instruct \
        --quant_type q4_k_m

Author: Jkanishkha0305
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(model_path: str, base_model_id: str, output_dir: str):
    """Merge LoRA weights and save full model for quantization."""
    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        print("Merging LoRA adapters...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        print("LoRA merged successfully")

    merged_path = os.path.join(output_dir, "merged_model")
    os.makedirs(merged_path, exist_ok=True)

    print(f"Saving merged model to {merged_path}")
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    return merged_path


def convert_to_gguf(merged_path: str, output_dir: str, quant_type: str = "q4_k_m"):
    """Convert model to GGUF format using llama.cpp."""
    gguf_path = os.path.join(output_dir, f"model-{quant_type}.gguf")

    print(f"\nConverting to GGUF ({quant_type})...")
    print("Note: Requires llama.cpp to be installed.")
    print("Install: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")

    # Try llama.cpp conversion
    convert_cmd = [
        "python", "-m", "llama_cpp.convert",
        "--outfile", gguf_path,
        "--outtype", quant_type,
        merged_path,
    ]

    print(f"Command: {' '.join(convert_cmd)}")
    print("\nIf this fails, run manually:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {merged_path} --outfile {gguf_path} --outtype {quant_type}")

    try:
        subprocess.run(convert_cmd, check=True)
        print(f"GGUF saved to: {gguf_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Automatic conversion failed. Please run manually:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {merged_path} --outfile {gguf_path} --outtype {quant_type}")

    return gguf_path


def benchmark_inference(model_path: str, base_model_id: str, n_samples: int = 5):
    """Benchmark inference speed for on-device deployment metrics."""
    print("\nBenchmarking inference speed...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id if os.path.exists(os.path.join(model_path, "adapter_config.json")) else model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    test_prompts = [
        "I've been feeling really depressed lately.",
        "My anxiety won't let me sleep.",
        "I feel like nobody understands me.",
        "I'm struggling with grief after losing someone.",
        "Work stress is overwhelming me.",
    ][:n_samples]

    latencies = []
    token_counts = []

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )
        elapsed = time.time() - start

        n_tokens = outputs.shape[1] - input_len
        latencies.append(elapsed)
        token_counts.append(n_tokens)

    # Compute metrics
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_tok_per_sec = avg_tokens / avg_latency

    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 2 / 1e6  # float16

    results = {
        "avg_latency_s": round(avg_latency, 3),
        "avg_tokens_generated": round(avg_tokens, 1),
        "avg_tokens_per_sec": round(avg_tok_per_sec, 1),
        "param_count": param_count,
        "model_size_mb_fp16": round(model_size_mb, 1),
        "estimated_gguf_q4_mb": round(model_size_mb * 0.3, 1),
        "device": str(model.device),
    }

    print("\nInference Benchmark Results:")
    print(f"  Avg latency:        {results['avg_latency_s']:.3f}s")
    print(f"  Avg tokens/sec:     {results['avg_tokens_per_sec']:.1f}")
    print(f"  Model size (FP16):  {results['model_size_mb_fp16']:.1f} MB")
    print(f"  Est. GGUF Q4 size:  {results['estimated_gguf_q4_mb']:.1f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Quantize RV-GRPO models for deployment")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True,
                        help="Base model HF ID")
    parser.add_argument("--output_dir", type=str, default="./deployment/gguf")
    parser.add_argument("--quant_type", type=str, default="q4_k_m",
                        choices=["q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"])
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark")
    parser.add_argument("--skip_convert", action="store_true",
                        help="Skip GGUF conversion, only benchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_convert:
        # Step 1: Merge LoRA and save
        merged_path = merge_and_save(args.model_path, args.model_id, args.output_dir)

        # Step 2: Convert to GGUF
        gguf_path = convert_to_gguf(merged_path, args.output_dir, args.quant_type)

    # Step 3: Benchmark
    if args.benchmark:
        bench_results = benchmark_inference(args.model_path, args.model_id)

        bench_path = os.path.join(args.output_dir, "benchmark_results.json")
        with open(bench_path, "w") as f:
            json.dump(bench_results, f, indent=2)
        print(f"\nBenchmark saved to: {bench_path}")


if __name__ == "__main__":
    main()
