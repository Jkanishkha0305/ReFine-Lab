"""
Ablation Study Runner for RV-GRPO

Systematically removes each rubric reward (R1-R5) one at a time to
measure each signal's contribution to therapeutic behavioral alignment.

Also runs a weight sweep to find optimal reward weighting.

Usage:
    # Leave-one-out ablation
    python experiments/grpo/run_ablation.py \
        --mode leave_one_out \
        --model qwen \
        --sft_checkpoint ./outputs/sft/qwen/final_model

    # Weight sweep
    python experiments/grpo/run_ablation.py \
        --mode weight_sweep \
        --model qwen \
        --sft_checkpoint ./outputs/sft/qwen/final_model

    # Evaluate ablation results
    python experiments/grpo/run_ablation.py \
        --mode evaluate \
        --ablation_dir ./outputs/ablation/qwen

Author: Jkanishkha0305
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Default weights
DEFAULT_WEIGHTS = {
    "open_question": 0.20,
    "emotion_reflection": 0.25,
    "no_premature_advice": 0.25,
    "validation_before_redirect": 0.20,
    "length_appropriate": 0.10,
}

# Weight flag names
WEIGHT_FLAGS = {
    "open_question": "--w_open_question",
    "emotion_reflection": "--w_emotion_reflection",
    "no_premature_advice": "--w_no_premature_advice",
    "validation_before_redirect": "--w_validation",
    "length_appropriate": "--w_length",
}

# Ablation configurations: each removes one reward
LEAVE_ONE_OUT_CONFIGS = {
    "full": DEFAULT_WEIGHTS,
    "no_R1_open_question": {**DEFAULT_WEIGHTS, "open_question": 0.0},
    "no_R2_emotion_reflection": {**DEFAULT_WEIGHTS, "emotion_reflection": 0.0},
    "no_R3_no_premature_advice": {**DEFAULT_WEIGHTS, "no_premature_advice": 0.0},
    "no_R4_validation": {**DEFAULT_WEIGHTS, "validation_before_redirect": 0.0},
    "no_R5_length": {**DEFAULT_WEIGHTS, "length_appropriate": 0.0},
    "only_R1": {"open_question": 1.0, "emotion_reflection": 0.0, "no_premature_advice": 0.0, "validation_before_redirect": 0.0, "length_appropriate": 0.0},
    "only_R2": {"open_question": 0.0, "emotion_reflection": 1.0, "no_premature_advice": 0.0, "validation_before_redirect": 0.0, "length_appropriate": 0.0},
    "only_R3": {"open_question": 0.0, "emotion_reflection": 0.0, "no_premature_advice": 1.0, "validation_before_redirect": 0.0, "length_appropriate": 0.0},
}

# Weight sweep grid
WEIGHT_SWEEP_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]


def build_train_command(
    model: str,
    sft_checkpoint: str,
    weights: dict,
    output_dir: str,
    extra_args: dict = None,
) -> list:
    """Build the training command for a single ablation run."""
    cmd = [
        sys.executable, "experiments/grpo/train_rv_grpo.py",
        "--model", model,
        "--sft_checkpoint", sft_checkpoint,
        "--output_dir", output_dir,
        "--num_epochs", "1",
    ]

    for weight_name, flag in WEIGHT_FLAGS.items():
        cmd.extend([flag, str(weights[weight_name])])

    if extra_args:
        for k, v in extra_args.items():
            cmd.extend([str(k), str(v)])

    return cmd


def build_eval_command(model_path: str, model_id: str, method: str, output_dir: str) -> list:
    """Build evaluation command."""
    return [
        sys.executable, "evaluation/behavioral_metrics.py",
        "--model_path", model_path,
        "--model_id", model_id,
        "--method", method,
        "--output_dir", output_dir,
    ]


def run_leave_one_out(args):
    """Run leave-one-out ablation study."""
    print("=" * 60)
    print("ABLATION STUDY: Leave-One-Out")
    print("=" * 60)

    ablation_dir = Path(args.output_dir) / "ablation" / args.model
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Save ablation plan
    plan = {
        "mode": "leave_one_out",
        "model": args.model,
        "sft_checkpoint": args.sft_checkpoint,
        "configs": {k: v for k, v in LEAVE_ONE_OUT_CONFIGS.items()},
        "timestamp": datetime.now().isoformat(),
    }
    with open(ablation_dir / "ablation_plan.json", "w") as f:
        json.dump(plan, f, indent=2)

    results = {}

    for config_name, weights in LEAVE_ONE_OUT_CONFIGS.items():
        print(f"\n--- Running ablation: {config_name} ---")
        print(f"    Weights: {weights}")

        run_dir = str(ablation_dir / config_name)
        cmd = build_train_command(
            model=args.model,
            sft_checkpoint=args.sft_checkpoint,
            weights=weights,
            output_dir=run_dir,
            extra_args={"--max_samples": str(args.max_samples)} if args.max_samples else None,
        )

        if args.use_wandb:
            cmd.extend(["--use_wandb", "--wandb_project", f"rv-grpo-ablation-{args.model}"])

        print(f"    Command: {' '.join(cmd)}")

        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
                results[config_name] = {"status": "success", "output_dir": run_dir}
            except subprocess.CalledProcessError as e:
                print(f"    FAILED: {e}")
                results[config_name] = {"status": "failed", "error": str(e)}
        else:
            results[config_name] = {"status": "dry_run", "command": " ".join(cmd)}

    # Save results summary
    with open(ablation_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation complete! Results in: {ablation_dir}")
    return results


def run_weight_sweep(args):
    """Run weight sweep to find optimal reward weighting."""
    print("=" * 60)
    print("ABLATION STUDY: Weight Sweep")
    print("=" * 60)

    ablation_dir = Path(args.output_dir) / "weight_sweep" / args.model
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Generate sweep configurations
    # Focus on R1-R4 weights (R5 length is less important)
    sweep_configs = []

    # Targeted sweep: vary 2 key weights at a time
    key_pairs = [
        ("open_question", "emotion_reflection"),
        ("emotion_reflection", "no_premature_advice"),
        ("no_premature_advice", "validation_before_redirect"),
    ]

    for w1_name, w2_name in key_pairs:
        for v1, v2 in product(WEIGHT_SWEEP_VALUES, WEIGHT_SWEEP_VALUES):
            if v1 == 0.0 and v2 == 0.0:
                continue
            weights = dict(DEFAULT_WEIGHTS)
            weights[w1_name] = v1
            weights[w2_name] = v2
            config_name = f"{w1_name[:3]}={v1}_{w2_name[:3]}={v2}"
            sweep_configs.append((config_name, weights))

    print(f"Total sweep configurations: {len(sweep_configs)}")

    if args.max_sweep_runs and len(sweep_configs) > args.max_sweep_runs:
        import random
        random.seed(42)
        sweep_configs = random.sample(sweep_configs, args.max_sweep_runs)
        print(f"Sampled {args.max_sweep_runs} configurations")

    # Save sweep plan
    plan = {
        "mode": "weight_sweep",
        "model": args.model,
        "n_configs": len(sweep_configs),
        "configs": {name: w for name, w in sweep_configs},
        "timestamp": datetime.now().isoformat(),
    }
    with open(ablation_dir / "sweep_plan.json", "w") as f:
        json.dump(plan, f, indent=2)

    if args.dry_run:
        print("Dry run — not executing. Plan saved.")
        return

    results = {}
    for i, (config_name, weights) in enumerate(sweep_configs):
        print(f"\n[{i+1}/{len(sweep_configs)}] Running: {config_name}")

        run_dir = str(ablation_dir / config_name)
        cmd = build_train_command(
            model=args.model,
            sft_checkpoint=args.sft_checkpoint,
            weights=weights,
            output_dir=run_dir,
            extra_args={"--max_samples": str(args.max_samples)} if args.max_samples else None,
        )

        try:
            subprocess.run(cmd, check=True)
            results[config_name] = {"status": "success", "weights": weights}
        except subprocess.CalledProcessError as e:
            results[config_name] = {"status": "failed", "error": str(e)}

    with open(ablation_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWeight sweep complete! Results in: {ablation_dir}")


def evaluate_ablation(args):
    """Evaluate all ablation runs and produce comparison table."""
    from data.rubric_rewards import compute_rubric_reward

    ablation_dir = Path(args.ablation_dir)
    if not ablation_dir.exists():
        print(f"Ablation directory not found: {ablation_dir}")
        return

    # Find all final model directories
    model_dirs = []
    for subdir in sorted(ablation_dir.iterdir()):
        final_path = subdir / args.model / "final"
        if final_path.exists():
            model_dirs.append((subdir.name, str(final_path)))
        elif (subdir / "final").exists():
            model_dirs.append((subdir.name, str(subdir / "final")))

    if not model_dirs:
        print("No trained models found in ablation directory")
        return

    print(f"Found {len(model_dirs)} ablation models to evaluate")

    # Import evaluation
    from evaluation.behavioral_metrics import (
        load_model, evaluate_model, print_results_table, TEST_PROMPTS,
    )

    all_agg = []
    for config_name, model_path in model_dirs:
        print(f"\n--- Evaluating: {config_name} ---")
        model, tokenizer = load_model(model_path, args.model_id)

        agg, details = evaluate_model(
            model, tokenizer,
            TEST_PROMPTS[:args.n_prompts],
            method_name=config_name,
            model_name=args.model,
        )
        all_agg.append(agg)

        del model, tokenizer
        import torch
        torch.cuda.empty_cache()

    print_results_table(all_agg)

    # Save
    eval_path = ablation_dir / "ablation_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(all_agg, f, indent=2)
    print(f"\nEvaluation saved to {eval_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="RV-GRPO Ablation Study")
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["leave_one_out", "weight_sweep", "evaluate"],
    )
    parser.add_argument("--model", type=str, default="qwen",
                        choices=["llama", "qwen", "phi"])
    parser.add_argument("--sft_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Reduced samples for faster ablation runs")
    parser.add_argument("--max_sweep_runs", type=int, default=20,
                        help="Max configurations for weight sweep")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")

    # Evaluate mode args
    parser.add_argument("--ablation_dir", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--n_prompts", type=int, default=30)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "leave_one_out":
        if not args.sft_checkpoint:
            print("Error: --sft_checkpoint required for training modes")
            sys.exit(1)
        run_leave_one_out(args)

    elif args.mode == "weight_sweep":
        if not args.sft_checkpoint:
            print("Error: --sft_checkpoint required for training modes")
            sys.exit(1)
        run_weight_sweep(args)

    elif args.mode == "evaluate":
        if not args.ablation_dir:
            print("Error: --ablation_dir required for evaluate mode")
            sys.exit(1)
        evaluate_ablation(args)


if __name__ == "__main__":
    main()
