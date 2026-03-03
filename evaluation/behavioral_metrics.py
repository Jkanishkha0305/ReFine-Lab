"""
Behavioral Metrics Evaluation for RV-GRPO

Evaluate trained models on rubric-based behavioral metrics using a
held-out test set of mental health prompts.

Generates the main results table for the paper:
  Model x Method -> R1, R2, R3, R4, R5, Combined

Usage:
    python evaluation/behavioral_metrics.py \
        --model_path ./outputs/rv_grpo/qwen/final \
        --model_id Qwen/Qwen2.5-1.5B-Instruct \
        --method rv-grpo \
        --output_dir ./results

    # Batch evaluate all models
    python evaluation/behavioral_metrics.py --batch_config ./evaluation/eval_config.json

Author: Jkanishkha0305
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.rubric_rewards import compute_rubric_reward


# Default test prompts (held-out, not used in training)
TEST_PROMPTS = [
    "I've been feeling really depressed lately and I don't know what to do.",
    "My anxiety is getting worse and I can't sleep at night.",
    "I feel like nobody understands what I'm going through.",
    "I lost my job and I feel worthless.",
    "I keep having panic attacks and they scare me.",
    "My relationship just ended and I feel completely devastated.",
    "I'm struggling with addiction and I'm ashamed to tell anyone.",
    "I feel so overwhelmed with everything going on in my life.",
    "I've been having thoughts of self-harm and I'm scared.",
    "My parents don't support my decisions and it makes me feel trapped.",
    "I feel like I'm failing at everything in life.",
    "I can't stop crying and I don't even know why.",
    "I'm so lonely, I have no one to talk to.",
    "Work stress is making me physically sick.",
    "I feel guilty all the time even when I haven't done anything wrong.",
    "I'm going through grief and I can't seem to move forward.",
    "I feel disconnected from everyone around me.",
    "My self-esteem is at rock bottom.",
    "I'm terrified of the future and what it holds.",
    "I feel burned out and have no motivation to do anything.",
    "I've been having intrusive thoughts that won't stop.",
    "My childhood trauma still affects me every day.",
    "I feel like a burden to everyone around me.",
    "I can't concentrate on anything and it's affecting my work.",
    "I'm angry all the time and I don't know how to control it.",
    "I feel empty inside, like nothing matters anymore.",
    "I'm afraid of being alone but I push everyone away.",
    "My eating habits are out of control and I hate my body.",
    "I feel like I'm living on autopilot, just going through the motions.",
    "I'm struggling to cope after a traumatic experience.",
]


def load_model(model_path: str, base_model_id: str = None):
    """Load a trained model for evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if it's a LoRA adapter
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config) and base_model_id:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a single response."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def evaluate_model(
    model, tokenizer,
    prompts: List[str],
    method_name: str = "unknown",
    model_name: str = "unknown",
) -> Dict:
    """Evaluate a model on all behavioral metrics."""
    results = []

    for prompt in tqdm(prompts, desc=f"Evaluating {model_name}/{method_name}"):
        response = generate_response(model, tokenizer, prompt)
        breakdown = compute_rubric_reward(prompt, response, return_breakdown=True)

        results.append({
            "prompt": prompt,
            "response": response,
            "scores": breakdown["scores"],
            "combined": breakdown["combined"],
        })

    # Aggregate metrics
    n = len(results)
    agg = {
        "model": model_name,
        "method": method_name,
        "n_samples": n,
        "metrics": {},
    }

    for metric in ["open_question", "emotion_reflection", "no_premature_advice",
                    "validation_before_redirect", "length_appropriate"]:
        values = [r["scores"][metric] for r in results]
        agg["metrics"][metric] = {
            "mean": sum(values) / n,
            "count_1.0": sum(1 for v in values if v == 1.0),
            "count_0.0": sum(1 for v in values if v == 0.0),
        }

    combined_values = [r["combined"] for r in results]
    agg["metrics"]["combined"] = {
        "mean": sum(combined_values) / n,
        "min": min(combined_values),
        "max": max(combined_values),
        "std": (sum((v - sum(combined_values)/n)**2 for v in combined_values) / n) ** 0.5,
    }

    return agg, results


def print_results_table(all_results: List[Dict]):
    """Print a formatted results table for the paper."""
    print("\n" + "=" * 100)
    print("BEHAVIORAL METRICS RESULTS")
    print("=" * 100)

    header = f"{'Model':<15} {'Method':<15} {'R1:OQ':>8} {'R2:ER':>8} {'R3:NPA':>8} {'R4:VBR':>8} {'R5:Len':>8} {'Combined':>10}"
    print(header)
    print("-" * 100)

    for r in all_results:
        m = r["metrics"]
        print(
            f"{r['model']:<15} "
            f"{r['method']:<15} "
            f"{m['open_question']['mean']:>8.3f} "
            f"{m['emotion_reflection']['mean']:>8.3f} "
            f"{m['no_premature_advice']['mean']:>8.3f} "
            f"{m['validation_before_redirect']['mean']:>8.3f} "
            f"{m['length_appropriate']['mean']:>8.3f} "
            f"{m['combined']['mean']:>10.3f}"
        )

    print("=" * 100)
    print("R1:OQ = Open Question | R2:ER = Emotion Reflection | R3:NPA = No Premature Advice")
    print("R4:VBR = Validation Before Redirect | R5:Len = Length Appropriate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate behavioral metrics")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Base model ID (for LoRA models)")
    parser.add_argument("--method", type=str, default="rv-grpo",
                        help="Training method name")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short model name for display")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--n_prompts", type=int, default=30,
                        help="Number of test prompts")
    parser.add_argument("--batch_config", type=str, default=None,
                        help="JSON config for batch evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prompts = TEST_PROMPTS[:args.n_prompts]

    if args.batch_config:
        # Batch mode: evaluate multiple models
        with open(args.batch_config) as f:
            configs = json.load(f)

        all_agg = []
        for cfg in configs:
            print(f"\n--- Evaluating {cfg['model_name']}/{cfg['method']} ---")
            model, tokenizer = load_model(cfg["model_path"], cfg.get("model_id"))
            agg, details = evaluate_model(
                model, tokenizer, prompts,
                method_name=cfg["method"],
                model_name=cfg["model_name"],
            )
            all_agg.append(agg)

            # Save individual results
            detail_path = os.path.join(
                args.output_dir,
                f"{cfg['model_name']}_{cfg['method']}_details.json"
            )
            with open(detail_path, "w") as f:
                json.dump(details, f, indent=2)

            # Free GPU memory
            del model, tokenizer
            torch.cuda.empty_cache()

        # Print combined table
        print_results_table(all_agg)

        # Save combined results
        combined_path = os.path.join(args.output_dir, "all_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_agg, f, indent=2)
        print(f"\nResults saved to {combined_path}")

    else:
        # Single model mode
        model_name = args.model_name or Path(args.model_path).parent.name
        print(f"Evaluating {model_name}/{args.method}...")

        model, tokenizer = load_model(args.model_path, args.model_id)
        agg, details = evaluate_model(
            model, tokenizer, prompts,
            method_name=args.method,
            model_name=model_name,
        )

        print_results_table([agg])

        # Save results
        result_path = os.path.join(
            args.output_dir, f"{model_name}_{args.method}_results.json"
        )
        with open(result_path, "w") as f:
            json.dump({"aggregate": agg, "details": details}, f, indent=2)
        print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
