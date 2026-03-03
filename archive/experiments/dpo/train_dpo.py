"""
Direct Preference Optimization (DPO) Training

Train a model using DPO - a simpler alternative to RLHF that directly optimizes
on preference data without needing a reward model.

Usage:
    python train_dpo.py --model_name google/gemma-2-270m --dataset HuggingFaceH4/ultrafeedback_binarized
"""

import argparse
import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model

from utils.model_loader import load_model_and_tokenizer, print_trainable_parameters
from utils.gpu_monitor import log_gpu_stats, print_gpu_info
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with DPO")
    
    parser.add_argument("--model_name", type=str, default="google/gemma-2-270m")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true", default=True)
    
    parser.add_argument("--output_dir", type=str, default="./outputs/dpo")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO beta parameter (temperature)")
    
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print_gpu_info()
    log_gpu_stats(prefix="[Initial] ")
    
    logger.info(f"Loading model: {args.model_name}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_flash_attention=True,
    )
    
    # Apply LoRA if requested
    if args.use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
    
    # Load reference model (frozen copy for DPO)
    logger.info("Loading reference model...")
    ref_model, _ = load_model_and_tokenizer(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_flash_attention=True,
    )
    
    log_gpu_stats(prefix="[After Model Load] ")
    
    # Load preference dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train_prefs")
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="wandb" if args.use_wandb else "none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    
    # Train
    logger.info("Starting DPO training...")
    trainer.train()
    
    logger.info("Training complete!")
    
    # Save model
    output_model_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    
    logger.info(f"Model saved to: {output_model_dir}")
    logger.info("✅ DPO training complete!")


if __name__ == "__main__":
    main()

