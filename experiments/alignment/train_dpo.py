"""
Direct Preference Optimization (DPO) Training Script

Train SLMs for mental health therapy using DPO alignment.
Requires a preference dataset with chosen/rejected pairs.

Supported Models:
    - google/gemma-3-270m
    - meta-llama/Llama-3.2-1B
    - microsoft/Phi-4-mini-instruct
    - Qwen/Qwen2.5-0.5B-Instruct

Usage:
    python train_dpo.py --model gemma --max_samples 1000
    python train_dpo.py --model llama --sft_checkpoint ./outputs/sft/llama/final
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================================================
# Model Registry
# ============================================================================
MODEL_REGISTRY = {
    "llama": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended_batch_size": 4,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended_batch_size": 4,
    },
    "phi": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended_batch_size": 2,
    },
}


def setup_logging(model_name: str, output_dir: str) -> logging.Logger:
    """Setup separate logging for each model."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dpo_{model_name}_{timestamp}.log"

    logger = logging.getLogger(f"dpo_{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training for Mental Health SLMs")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train: gemma, llama, phi, qwen",
    )
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        default=None,
        help="Path to SFT checkpoint (recommended). If None, trains from base.",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="argilla/ultrafeedback-binarized-preferences-cleaned",
        help="Preference dataset with chosen/rejected pairs",
    )
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)

    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/dpo")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="refine-lab-dpo")
    parser.add_argument("--seed", type=int, default=42)

    # HuggingFace Hub
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push final model to HuggingFace Hub")
    parser.add_argument("--hub_username", type=str, default="jkanishkha0305")

    return parser.parse_args()


def load_model_and_tokenizer(
    model_id: str,
    sft_checkpoint: str = None,
    load_in_4bit: bool = True,
):
    """Load model with optional SFT checkpoint."""

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        sft_checkpoint if sft_checkpoint else model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO requires left padding

    # Load model
    if sft_checkpoint:
        # Load SFT fine-tuned model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model = PeftModel.from_pretrained(base_model, sft_checkpoint)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

    return model, tokenizer


def prepare_preference_dataset(dataset_name: str, max_samples: int, tokenizer, logger):
    """
    Prepare preference dataset for DPO.
    Expected format: prompt, chosen, rejected
    """
    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset
    dataset = load_dataset(dataset_name, split="train")

    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        logger.info(f"Selected {max_samples} samples")

    # Format for DPO - needs prompt, chosen, rejected columns
    def format_example(example):
        # Handle different dataset formats
        if "prompt" in example and "chosen" in example and "rejected" in example:
            return example
        elif "instruction" in example:
            # Alpaca-style format
            prompt = example["instruction"]
            if example.get("input"):
                prompt += f"\n\nInput: {example['input']}"
            return {
                "prompt": prompt,
                "chosen": example.get("chosen", example.get("output", "")),
                "rejected": example.get("rejected", ""),
            }
        elif "question" in example:
            return {
                "prompt": example["question"],
                "chosen": example.get("chosen", ""),
                "rejected": example.get("rejected", ""),
            }
        else:
            # Try to extract from messages format
            if "chosen" in example and isinstance(example["chosen"], list):
                prompt = example["chosen"][0]["content"] if example["chosen"] else ""
                chosen = example["chosen"][-1]["content"] if len(example["chosen"]) > 1 else ""
                rejected = example["rejected"][-1]["content"] if example.get("rejected") and len(example["rejected"]) > 1 else ""
                return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            return example

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # Filter out empty examples
    dataset = dataset.filter(
        lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0
    )

    logger.info(f"Prepared {len(dataset)} preference pairs")
    return dataset


def save_config(args, output_dir: str, model_config: dict):
    """Save training configuration for reproducibility."""
    config_path = Path(output_dir) / "training_config.json"

    config = {
        "model": args.model,
        "model_id": model_config["model_id"],
        "sft_checkpoint": args.sft_checkpoint,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "dpo_beta": args.beta,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_length": args.max_length,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def main():
    args = parse_args()

    # Get model config
    model_config = MODEL_REGISTRY[args.model]
    model_id = model_config["model_id"]

    # Set batch size
    if args.batch_size is None:
        args.batch_size = model_config["recommended_batch_size"]

    # Setup output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.model, args.output_dir)
    logger.info("=" * 60)
    logger.info("DPO Training Started")
    logger.info("=" * 60)
    logger.info(f"Model: {model_id}")
    logger.info(f"SFT Checkpoint: {args.sft_checkpoint or 'None (training from base)'}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"DPO Beta: {args.beta}")

    # Save config
    config_path = save_config(args, output_dir, model_config)
    logger.info(f"Config saved to: {config_path}")

    # Setup wandb
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"dpo_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
        )

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        sft_checkpoint=args.sft_checkpoint,
    )

    # Log GPU usage
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_used = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"GPU Memory: {gpu_used:.1f}GB / {gpu_mem:.1f}GB")

    # Load and prepare dataset
    dataset = prepare_preference_dataset(
        args.dataset, args.max_samples, tokenizer, logger
    )

    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=model_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # DPO training config
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed,
        # DPO specific
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    # Initialize DPO trainer
    logger.info("Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will be created automatically
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    logger.info("Training complete!")

    # Save final model
    final_model_path = output_dir / "final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    logger.info(f"Model saved to: {final_model_path}")

    # Push to HuggingFace Hub
    if args.push_to_hub:
        hub_repo = f"{args.hub_username}/rv-grpo-dpo-{args.model}"
        logger.info(f"Pushing to HuggingFace Hub: {hub_repo}")
        try:
            trainer.push_to_hub(hub_repo)
            logger.info(f"Pushed to: https://huggingface.co/{hub_repo}")
        except Exception as e:
            logger.warning(f"Hub push failed: {e}. Model saved locally at {final_model_path}")

    # Evaluate
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()

    # Log results
    logger.info("Evaluation Results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save eval results
    eval_path = output_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    if args.use_wandb:
        artifact = wandb.Artifact(f"dpo-{args.model}-results", type="results")
        artifact.add_file(str(eval_path))
        artifact.add_file(str(config_path))
        wandb.log_artifact(artifact)
        logger.info("Results & config uploaded to W&B artifacts")
        wandb.finish()

    logger.info("=" * 60)
    logger.info("DPO Training Complete!")
    logger.info(f"Model: {final_model_path}")
    logger.info(f"Logs: {output_dir / 'logs'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
