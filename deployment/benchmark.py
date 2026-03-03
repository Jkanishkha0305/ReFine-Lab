"""
Inference Benchmarking for On-Device Deployment

Measures latency, throughput, and memory usage for trained models
across different formats (PyTorch, GGUF) and precision levels.

Usage:
    # Benchmark PyTorch model
    python deployment/benchmark.py \
        --model_path ./outputs/rv_grpo/qwen/final \
        --model_id Qwen/Qwen2.5-1.5B-Instruct

    # Compare multiple models
    python deployment/benchmark.py \
        --batch_config ./evaluation/eval_config.json \
        --output_dir ./results/benchmarks

    # Benchmark GGUF model (requires llama-cpp-python)
    python deployment/benchmark.py \
        --gguf_path ./deployment/gguf/model-q4_k_m.gguf

Author: Jkanishkha0305
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))


TEST_PROMPTS = [
    "I've been feeling really depressed lately.",
    "My anxiety won't let me sleep at night.",
    "I feel like nobody understands me.",
    "Work stress is overwhelming me completely.",
    "I'm struggling with grief after losing someone close.",
    "I keep having panic attacks and they scare me.",
    "I feel disconnected from everyone around me.",
    "My self-esteem is at rock bottom right now.",
    "I'm terrified of the future and what it holds.",
    "I feel burned out and have no motivation.",
]


def load_pytorch_model(model_path: str, base_model_id: str = None):
    """Load PyTorch model for benchmarking."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config) and base_model_id:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.float16,
        )

    model.eval()
    return model, tokenizer


def benchmark_pytorch(model, tokenizer, prompts: list, max_new_tokens: int = 200, warmup: int = 2):
    """Benchmark PyTorch model inference."""
    results = {
        "format": "pytorch_fp16",
        "max_new_tokens": max_new_tokens,
        "latencies": [],
        "tokens_generated": [],
        "tokens_per_sec": [],
        "time_to_first_token": [],
    }

    # Warmup
    for prompt in prompts[:warmup]:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10, do_sample=False)

    # Benchmark
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # Time to first token
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1, do_sample=False,
            )
        ttft = time.perf_counter() - start

        # Full generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        n_tokens = outputs.shape[1] - input_len
        results["latencies"].append(elapsed)
        results["tokens_generated"].append(n_tokens)
        results["tokens_per_sec"].append(n_tokens / elapsed if elapsed > 0 else 0)
        results["time_to_first_token"].append(ttft)

    # Aggregate
    n = len(results["latencies"])
    results["summary"] = {
        "avg_latency_s": sum(results["latencies"]) / n,
        "avg_tokens_generated": sum(results["tokens_generated"]) / n,
        "avg_tokens_per_sec": sum(results["tokens_per_sec"]) / n,
        "avg_ttft_s": sum(results["time_to_first_token"]) / n,
        "p50_latency_s": sorted(results["latencies"])[n // 2],
        "p90_latency_s": sorted(results["latencies"])[int(n * 0.9)],
    }

    # Model size info
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["model_info"] = {
        "total_params": param_count,
        "trainable_params": trainable,
        "size_fp16_mb": round(param_count * 2 / 1e6, 1),
        "size_q4_est_mb": round(param_count * 0.5 / 1e6, 1),
        "device": str(model.device),
    }

    # GPU memory
    if torch.cuda.is_available():
        results["model_info"]["gpu_allocated_mb"] = round(
            torch.cuda.memory_allocated() / 1e6, 1
        )
        results["model_info"]["gpu_reserved_mb"] = round(
            torch.cuda.memory_reserved() / 1e6, 1
        )

    return results


def benchmark_gguf(gguf_path: str, prompts: list, max_new_tokens: int = 200):
    """Benchmark GGUF model using llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: llama-cpp-python not installed.")
        print("Install: pip install llama-cpp-python")
        return None

    print(f"Loading GGUF model: {gguf_path}")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )

    results = {
        "format": "gguf",
        "model_path": gguf_path,
        "max_new_tokens": max_new_tokens,
        "latencies": [],
        "tokens_generated": [],
        "tokens_per_sec": [],
    }

    for prompt in prompts:
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"

        start = time.perf_counter()
        output = llm(
            formatted,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - start

        n_tokens = output["usage"]["completion_tokens"]
        results["latencies"].append(elapsed)
        results["tokens_generated"].append(n_tokens)
        results["tokens_per_sec"].append(n_tokens / elapsed if elapsed > 0 else 0)

    n = len(results["latencies"])
    results["summary"] = {
        "avg_latency_s": sum(results["latencies"]) / n,
        "avg_tokens_generated": sum(results["tokens_generated"]) / n,
        "avg_tokens_per_sec": sum(results["tokens_per_sec"]) / n,
        "p50_latency_s": sorted(results["latencies"])[n // 2],
        "p90_latency_s": sorted(results["latencies"])[int(n * 0.9)],
    }

    # File size
    results["model_info"] = {
        "file_size_mb": round(os.path.getsize(gguf_path) / 1e6, 1),
    }

    return results


def print_benchmark_results(results_list: list):
    """Print formatted benchmark comparison table."""
    print(f"\n{'='*90}")
    print("INFERENCE BENCHMARK RESULTS")
    print(f"{'='*90}")

    header = (
        f"{'Model':<15} {'Method':<12} {'Format':<10} "
        f"{'Tok/s':>8} {'Latency':>8} {'TTFT':>8} "
        f"{'Size(MB)':>10} {'GPU(MB)':>10}"
    )
    print(header)
    print("-" * 90)

    for r in results_list:
        s = r.get("summary", {})
        info = r.get("model_info", {})

        ttft = f"{s.get('avg_ttft_s', 0):.3f}" if "avg_ttft_s" in s else "N/A"
        gpu = f"{info.get('gpu_allocated_mb', 0):.0f}" if "gpu_allocated_mb" in info else "N/A"
        size = info.get("size_fp16_mb", info.get("file_size_mb", 0))

        print(
            f"{r.get('model_name', 'unknown'):<15} "
            f"{r.get('method', 'unknown'):<12} "
            f"{r.get('format', 'unknown'):<10} "
            f"{s.get('avg_tokens_per_sec', 0):>8.1f} "
            f"{s.get('avg_latency_s', 0):>8.3f} "
            f"{ttft:>8} "
            f"{size:>10.1f} "
            f"{gpu:>10}"
        )

    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="Inference benchmarking")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--method", type=str, default="rv-grpo")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--gguf_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/benchmarks")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--n_prompts", type=int, default=10)
    parser.add_argument("--batch_config", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = TEST_PROMPTS[:args.n_prompts]

    all_results = []

    if args.gguf_path:
        # Benchmark GGUF model
        results = benchmark_gguf(args.gguf_path, prompts, args.max_new_tokens)
        if results:
            results["model_name"] = args.model_name or Path(args.gguf_path).stem
            results["method"] = "gguf"
            all_results.append(results)

    elif args.batch_config:
        with open(args.batch_config) as f:
            configs = json.load(f)

        for cfg in configs:
            print(f"\n--- Benchmarking {cfg['model_name']}/{cfg['method']} ---")
            model, tokenizer = load_pytorch_model(
                cfg["model_path"], cfg.get("model_id")
            )
            results = benchmark_pytorch(
                model, tokenizer, prompts, args.max_new_tokens,
            )
            results["model_name"] = cfg["model_name"]
            results["method"] = cfg["method"]
            all_results.append(results)

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    elif args.model_path:
        model_name = args.model_name or Path(args.model_path).parent.name
        print(f"Benchmarking {model_name}/{args.method}...")

        model, tokenizer = load_pytorch_model(args.model_path, args.model_id)
        results = benchmark_pytorch(model, tokenizer, prompts, args.max_new_tokens)
        results["model_name"] = model_name
        results["method"] = args.method
        all_results.append(results)

    else:
        print("Error: provide --model_path, --gguf_path, or --batch_config")
        sys.exit(1)

    print_benchmark_results(all_results)

    # Save results
    save_path = os.path.join(args.output_dir, "benchmark_results.json")
    # Strip per-prompt details for cleaner output
    summary_results = []
    for r in all_results:
        summary_results.append({
            "model_name": r.get("model_name"),
            "method": r.get("method"),
            "format": r.get("format"),
            "summary": r.get("summary"),
            "model_info": r.get("model_info"),
        })

    with open(save_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
