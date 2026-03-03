"""
Data Preparation for RV-GRPO: Merge and prepare SFT datasets.

Combines multiple mental health conversation datasets into a unified
format for SFT training.

Datasets:
    - ShenLab/MentalChat16K (primary, 9.7K samples)
    - thu-coai/esconv (Emotional Support Conversations)

Output:
    - Merged dataset pushed to HuggingFace or saved locally
    - Prompt-only dataset for GRPO training

Usage:
    python data/prepare_sft_data.py --output_dir ./data/processed
    python data/prepare_sft_data.py --push_to_hub --hub_name jkanishkha0305/rv-grpo-sft

Author: Jkanishkha0305
"""

import argparse
import json
import os
from typing import List, Dict

from datasets import load_dataset, Dataset, concatenate_datasets


def load_mentalchat16k(max_samples: int = None) -> Dataset:
    """Load and format MentalChat16K dataset."""
    print("Loading ShenLab/MentalChat16K...")
    ds = load_dataset("ShenLab/MentalChat16K", split="train")
    print(f"  Loaded {len(ds)} samples")

    def format_example(example):
        return {
            "prompt": example["input"],
            "response": example["output"],
            "source": "mentalchat16k",
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["prompt"]) > 10 and len(x["response"]) > 10)

    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))

    print(f"  After filtering: {len(ds)} samples")
    return ds


def load_esconv(max_samples: int = None) -> Dataset:
    """Load and format ESConv (Emotional Support Conversations) dataset."""
    print("Loading thu-coai/esconv...")
    try:
        ds = load_dataset("thu-coai/esconv", split="train")
        print(f"  Loaded {len(ds)} conversations")
    except Exception as e:
        print(f"  Warning: Could not load ESConv: {e}")
        print("  Skipping ESConv dataset")
        return None

    # ESConv has multi-turn conversations, extract turn pairs
    examples = []
    for conv in ds:
        dialog = conv.get("dialog", conv.get("conversation", []))
        if not dialog:
            continue

        for i in range(len(dialog) - 1):
            # Look for user->supporter turn pairs
            turn = dialog[i]
            next_turn = dialog[i + 1]

            speaker = turn.get("speaker", turn.get("role", ""))
            next_speaker = next_turn.get("speaker", next_turn.get("role", ""))

            content = turn.get("content", turn.get("text", turn.get("utterance", "")))
            next_content = next_turn.get("content", next_turn.get("text", next_turn.get("utterance", "")))

            if not content or not next_content:
                continue

            # Map various speaker labels
            is_user = speaker.lower() in ["seeker", "user", "client", "patient", "help_seeker"]
            is_supporter = next_speaker.lower() in ["supporter", "assistant", "counselor", "therapist", "helper"]

            if is_user and is_supporter:
                examples.append({
                    "prompt": content,
                    "response": next_content,
                    "source": "esconv",
                })

    if not examples:
        print("  Warning: No valid turn pairs found in ESConv")
        return None

    ds = Dataset.from_list(examples)
    ds = ds.filter(lambda x: len(x["prompt"]) > 10 and len(x["response"]) > 10)

    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))

    print(f"  Extracted {len(ds)} turn pairs")
    return ds


def create_prompt_only_dataset(merged_ds: Dataset) -> Dataset:
    """
    Create a prompt-only dataset for GRPO training.

    GRPO only needs prompts — it generates completions itself and
    scores them with the reward function.
    """
    # Deduplicate prompts
    seen = set()
    unique_prompts = []
    for example in merged_ds:
        prompt = example["prompt"].strip()
        if prompt not in seen and len(prompt) > 20:
            seen.add(prompt)
            unique_prompts.append({"prompt": prompt})

    ds = Dataset.from_list(unique_prompts)
    print(f"Created prompt-only dataset: {len(ds)} unique prompts")
    return ds


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for RV-GRPO")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--max_mentalchat", type=int, default=None,
                        help="Max samples from MentalChat16K")
    parser.add_argument("--max_esconv", type=int, default=5000,
                        help="Max samples from ESConv")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_name", type=str, default="jkanishkha0305/rv-grpo-sft")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    print("=" * 60)
    print("Loading and preparing datasets")
    print("=" * 60)

    datasets_to_merge = []

    # Primary dataset
    mentalchat = load_mentalchat16k(max_samples=args.max_mentalchat)
    datasets_to_merge.append(mentalchat)

    # Supplementary datasets
    esconv = load_esconv(max_samples=args.max_esconv)
    if esconv is not None:
        datasets_to_merge.append(esconv)

    # Merge
    print("\n" + "=" * 60)
    print("Merging datasets")
    print("=" * 60)

    merged = concatenate_datasets(datasets_to_merge)
    merged = merged.shuffle(seed=args.seed)

    print(f"Total merged samples: {len(merged)}")

    # Source distribution
    source_counts = {}
    for example in merged:
        src = example["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    print("Source distribution:")
    for src, count in source_counts.items():
        print(f"  {src}: {count}")

    # Create train/eval split
    split = merged.train_test_split(test_size=0.05, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # Create prompt-only dataset for GRPO
    prompts_ds = create_prompt_only_dataset(merged)

    # Save locally
    print(f"\nSaving to {args.output_dir}...")
    train_ds.save_to_disk(os.path.join(args.output_dir, "sft_train"))
    eval_ds.save_to_disk(os.path.join(args.output_dir, "sft_eval"))
    prompts_ds.save_to_disk(os.path.join(args.output_dir, "grpo_prompts"))

    # Save stats
    stats = {
        "total_samples": len(merged),
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "grpo_prompts": len(prompts_ds),
        "source_distribution": source_counts,
    }
    with open(os.path.join(args.output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Push to hub if requested
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_name}")
        train_ds.push_to_hub(args.hub_name, split="train")
        eval_ds.push_to_hub(args.hub_name, split="eval")
        prompts_ds.push_to_hub(f"{args.hub_name}-prompts", split="train")

    print("\nDone!")
    print(f"  SFT train: {args.output_dir}/sft_train")
    print(f"  SFT eval:  {args.output_dir}/sft_eval")
    print(f"  GRPO prompts: {args.output_dir}/grpo_prompts")


if __name__ == "__main__":
    main()
