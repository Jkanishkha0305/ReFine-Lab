"""
Supervised Fine-Tuning with LoRA

Train a language model using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

Usage:
    python train_lora.py --model_name google/gemma-2-270m --dataset yahma/alpaca-cleaned
    python train_lora.py --config configs/training_configs/sft_lora_default.yaml
"""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

from utils.model_loader import load_model_and_tokenizer, print_trainable_parameters
from utils.tracking import setup_tracking
from utils.gpu_monitor import log_gpu_stats, print_gpu_info
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with LoRA")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="google/gemma-2-270m",
                       help="HuggingFace model name or path")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit quantization")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned",
                       help="Dataset name or path")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use")
    parser.add_argument("--max_seq_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/sft_lora",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="Warmup ratio")
    
    # Misc arguments
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (overrides CLI args)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases tracking")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_alpaca_prompt(example):
    """Format example in Alpaca style."""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    
    if input_text:
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""
    
    return {"text": text}


def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config (config takes precedence)
        for key, value in config.get('training', {}).items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Print GPU info
    print_gpu_info()
    log_gpu_stats(prefix="[Initial] ")
    
    logger.info(f"Loading model: {args.model_name}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_flash_attention=True,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    log_gpu_stats(prefix="[After Model Load] ")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    
    # Limit samples if specified
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Format dataset
    train_dataset = train_dataset.map(format_alpaca_prompt)
    eval_dataset = eval_dataset.map(format_alpaca_prompt)
    
    # Setup experiment tracking
    if args.use_wandb:
        tracker = setup_tracking(
            project_name="refine-lab",
            experiment_name=f"sft_lora_{args.model_name.split('/')[-1]}",
            config=vars(args),
            tracker="wandb"
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="wandb" if args.use_wandb else "none",
        load_best_model_at_end=True,
        seed=args.seed,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    log_gpu_stats(prefix="[Before Training] ")
    
    trainer.train()
    
    logger.info("Training complete!")
    log_gpu_stats(prefix="[After Training] ")
    
    # Save model
    output_model_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    
    logger.info(f"Model saved to: {output_model_dir}")
    
    # Evaluate
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    logger.info("Evaluation results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Finish tracking
    if args.use_wandb:
        tracker.finish()
    
    logger.info("✅ Training pipeline complete!")


if __name__ == "__main__":
    main()

