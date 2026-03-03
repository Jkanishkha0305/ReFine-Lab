"""
Supervised Fine-Tuning (SFT) for Mental Health SLM

Supported Models:
    - google/gemma-3-270m
    - meta-llama/Llama-3.2-1B
    - microsoft/Phi-4-mini-instruct
    - Qwen/Qwen2.5-0.5B-Instruct

Dataset: ShenLab/MentalChat16K

Usage:
    python train_sft.py --model gemma --max_samples 1000
    python train_sft.py --model llama --load_in_4bit --use_wandb
    python train_sft.py --model phi --num_epochs 2
    python train_sft.py --model qwen --max_samples 2000

Author: Jkanishkha0305
Date: 2025
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional

import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from trl import SFTTrainer

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


# ============================================================================
# SECTION 2: MODEL REGISTRY
# ============================================================================

# Your exact models - DO NOT EDIT
MODEL_REGISTRY = {
    "llama": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended_batch_size": 8,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended_batch_size": 8,
    },
    "phi": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended_batch_size": 4,
    },
}

# Dataset
DATASET_NAME = "ShenLab/MentalChat16K"

# Default hyperparameters
DEFAULTS = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_epochs": 1,           # Quick training
    "max_seq_length": 512,
    "warmup_ratio": 0.03,
    "max_samples": 2000,       # Use 2k samples by default
    "seed": 42,                # Reproducibility
}


# ============================================================================
# SECTION 3: ARGUMENT PARSER
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SFT Training for Mental Health SLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_sft.py --model gemma --max_samples 1000
  python train_sft.py --model llama --load_in_4bit --use_wandb
  python train_sft.py --model qwen --num_epochs 2 --batch_size 8
        """
    )

    # Required
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train: gemma, llama, phi, qwen"
    )

    # Data
    parser.add_argument("--max_samples", type=int, default=DEFAULTS["max_samples"],
                        help="Number of samples to use (default: 2000)")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULTS["max_seq_length"],
                        help="Max sequence length")

    # Training
    parser.add_argument("--num_epochs", type=int, default=DEFAULTS["num_epochs"],
                        help="Number of epochs (default: 1 for quick training)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: auto based on model)")
    parser.add_argument("--learning_rate", type=float, default=DEFAULTS["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=DEFAULTS["lora_r"],
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=DEFAULTS["lora_alpha"],
                        help="LoRA alpha")

    # Quantization
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit quantization (saves memory)")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Use 8-bit quantization")

    # Output & Logging
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Base output directory")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="mental-health-sft",
                        help="W&B project name")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                        help="Random seed for reproducibility")

    return parser.parse_args()


# ============================================================================
# SECTION 4: DATA LOADING
# ============================================================================

def format_to_chat(example: Dict, tokenizer) -> Dict:
    """
    Convert MentalChat16K format to chat template.

    MentalChat16K format:
        instruction: system prompt
        input: user's mental health question
        output: counselor's response
    """
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]

    # Apply model's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


def load_mental_health_dataset(tokenizer, max_samples: int, seed: int, logger):
    """
    Load and prepare MentalChat16K dataset.

    Returns:
        train_dataset, eval_dataset
    """
    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    logger.info(f"Total samples available: {len(dataset)}")

    # Shuffle and select samples
    dataset = dataset.shuffle(seed=seed)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    logger.info(f"Using {len(dataset)} samples")

    # Format to chat template
    logger.info("Formatting dataset...")
    dataset = dataset.map(
        lambda x: format_to_chat(x, tokenizer),
        desc="Formatting"
    )

    # Split: 90% train, 10% eval
    split = dataset.train_test_split(test_size=0.1, seed=seed)

    logger.info(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")

    return split["train"], split["test"]


# ============================================================================
# SECTION 5: MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    logger = None
):
    """Load model with optional quantization."""

    logger.info(f"Loading model: {model_id}")

    # Quantization config
    quant_config = None
    if load_in_4bit:
        logger.info("Using 4-bit quantization")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        logger.info("Using 8-bit quantization")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not quant_config else None,
    )

    # Prepare for training if quantized
    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    logger.info("Model loaded successfully")
    return model, tokenizer


def apply_lora(model, target_modules: list, lora_r: int, lora_alpha: int, logger):
    """Apply LoRA adapters to model."""

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=DEFAULTS["lora_dropout"],
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Log trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percent = 100 * trainable / total

    logger.info(f"LoRA applied: {trainable:,} / {total:,} params ({percent:.2f}%)")

    return model


# ============================================================================
# SECTION 6: TRAINING & SAVING
# ============================================================================

def setup_logging(model_name: str, output_dir: str) -> logging.Logger:
    """Setup logging for this specific model."""

    # Create model-specific log directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(f"sft_{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # File handler - separate log file per model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")

    return logger


def get_training_args(
    output_dir: str,
    run_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    use_wandb: bool,
    seed: int,
) -> TrainingArguments:
    """Create training arguments."""

    return TrainingArguments(
        # Output
        output_dir=output_dir,
        run_name=run_name,

        # Training
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # Learning rate
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=DEFAULTS["warmup_ratio"],

        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),

        # Memory
        gradient_checkpointing=True,
        optim="adamw_torch",

        # Logging
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        logging_first_step=True,
        report_to=["wandb"] if use_wandb and WANDB_AVAILABLE else ["none"],

        # Evaluation
        eval_strategy="steps",
        eval_steps=50,

        # Checkpoints
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Reproducibility
        seed=seed,
        data_seed=seed,
    )


def save_training_config(output_dir: str, args, model_config: dict):
    """Save training configuration for reproducibility."""

    config = {
        "model": model_config,
        "training": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_samples": args.max_samples,
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "seed": args.seed,
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
        },
        "dataset": DATASET_NAME,
        "timestamp": datetime.now().isoformat(),
    }

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


# ============================================================================
# SECTION 7: MAIN FUNCTION
# ============================================================================

def main():
    """Main training pipeline."""

    # Parse arguments
    args = parse_args()

    # Get model config
    model_config = MODEL_REGISTRY[args.model]
    model_id = model_config["model_id"]
    model_short_name = model_id.split("/")[-1].lower()

    # Setup output directory: outputs/{model_name}/
    output_dir = os.path.join(args.output_dir, model_short_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging (separate log per model)
    logger = setup_logging(model_short_name, output_dir)

    # Banner
    logger.info("=" * 60)
    logger.info("SFT Training for Mental Health SLM")
    logger.info("=" * 60)
    logger.info(f"Model:      {model_id}")
    logger.info(f"Dataset:    {DATASET_NAME}")
    logger.info(f"Samples:    {args.max_samples}")
    logger.info(f"Epochs:     {args.num_epochs}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Seed:       {args.seed}")
    logger.info("=" * 60)

    # Set batch size (from arg or model default)
    batch_size = args.batch_size or model_config["recommended_batch_size"]
    args.batch_size = batch_size
    logger.info(f"Batch size: {batch_size}")

    # Save training config for reproducibility
    config_path = save_training_config(output_dir, args, model_config)
    logger.info(f"Config saved: {config_path}")

    # Initialize W&B
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=f"sft-{model_short_name}",
            config={
                "model": model_id,
                "dataset": DATASET_NAME,
                "max_samples": args.max_samples,
                "num_epochs": args.num_epochs,
                "batch_size": batch_size,
                "learning_rate": args.learning_rate,
                "lora_r": args.lora_r,
                "seed": args.seed,
            }
        )
        logger.info(f"W&B initialized: {wandb.run.url}")

    # Step 1: Load model and tokenizer
    logger.info("-" * 40)
    logger.info("Step 1: Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        logger=logger,
    )

    # Step 2: Apply LoRA
    logger.info("-" * 40)
    logger.info("Step 2: Applying LoRA adapters")
    model = apply_lora(
        model=model,
        target_modules=model_config["target_modules"],
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        logger=logger,
    )

    # Step 3: Load dataset
    logger.info("-" * 40)
    logger.info("Step 3: Loading dataset")
    train_dataset, eval_dataset = load_mental_health_dataset(
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        seed=args.seed,
        logger=logger,
    )

    # Step 4: Setup training
    logger.info("-" * 40)
    logger.info("Step 4: Setting up trainer")

    run_name = f"sft-{model_short_name}-{datetime.now().strftime('%Y%m%d_%H%M')}"

    training_args = get_training_args(
        output_dir=os.path.join(output_dir, "checkpoints"),
        run_name=run_name,
        num_epochs=args.num_epochs,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=args.use_wandb,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    # Step 5: Train!
    logger.info("-" * 40)
    logger.info("Step 5: Starting training...")
    logger.info("-" * 40)

    train_result = trainer.train()

    logger.info("-" * 40)
    logger.info("Training complete!")

    # Step 6: Save final model
    logger.info("-" * 40)
    logger.info("Step 6: Saving final model")

    final_model_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    logger.info(f"Model saved: {final_model_dir}")

    # Step 7: Final evaluation
    logger.info("-" * 40)
    logger.info("Step 7: Running final evaluation")

    eval_results = trainer.evaluate()

    logger.info("Evaluation results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    metrics = {
        "train_loss": train_result.training_loss,
        "train_steps": train_result.global_step,
        "eval_results": eval_results,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved: {metrics_path}")

    # Cleanup
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    # Final summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Model:       {model_id}")
    logger.info(f"Final model: {final_model_dir}")
    logger.info(f"Config:      {config_path}")
    logger.info(f"Metrics:     {metrics_path}")
    logger.info("=" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
