"""
RV-GRPO: Rubric-Verifiable Group Relative Policy Optimization

Train SLMs for therapeutic behavioral alignment using GRPO with
clinically-grounded rubric rewards derived from Motivational Interviewing.

This is the CORE CONTRIBUTION of the paper.

Supported Models:
    - meta-llama/Llama-3.2-1B-Instruct
    - Qwen/Qwen2.5-1.5B-Instruct
    - microsoft/Phi-4-mini-instruct

Usage:
    python experiments/grpo/train_rv_grpo.py \
        --model llama \
        --sft_checkpoint ./outputs/sft/llama/final_model \
        --use_wandb

    python experiments/grpo/train_rv_grpo.py \
        --model qwen \
        --sft_checkpoint ./outputs/sft/qwen/final_model \
        --num_generations 8

Author: Jkanishkha0305
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.rubric_rewards import rubric_reward_for_grpo, compute_rubric_reward

# ============================================================================
# Model Registry — Updated for RV-GRPO
# ============================================================================
MODEL_REGISTRY = {
    "llama": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended_batch_size": 4,
        "num_generations": 8,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended_batch_size": 4,
        "num_generations": 8,
    },
    "phi": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "recommended_batch_size": 2,
        "num_generations": 4,
    },
}


def setup_logging(model_name: str, output_dir: str) -> logging.Logger:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rv_grpo_{model_name}_{timestamp}.log"

    logger = logging.getLogger(f"rv_grpo_{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
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
    parser = argparse.ArgumentParser(
        description="RV-GRPO: Rubric-Verifiable GRPO Training"
    )

    # Model
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train: llama, qwen, phi",
    )
    parser.add_argument(
        "--sft_checkpoint", type=str, required=True,
        help="Path to SFT checkpoint (required for GRPO)",
    )

    # Dataset
    parser.add_argument(
        "--prompts_dataset", type=str, default="./data/processed/grpo_prompts",
        help="Path to prompt-only dataset for GRPO",
    )
    parser.add_argument(
        "--prompts_hub_name", type=str, default=None,
        help="HuggingFace dataset name for prompts (alternative to local)",
    )
    parser.add_argument("--max_samples", type=int, default=2000)

    # GRPO Hyperparameters
    parser.add_argument("--num_generations", type=int, default=None,
                        help="Group size for GRPO (default: model-specific)")
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--kl_coef", type=float, default=0.05,
                        help="KL divergence coefficient")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Rubric Reward Weights (tunable)
    parser.add_argument("--w_open_question", type=float, default=0.20)
    parser.add_argument("--w_emotion_reflection", type=float, default=0.25)
    parser.add_argument("--w_no_premature_advice", type=float, default=0.25)
    parser.add_argument("--w_validation", type=float, default=0.20)
    parser.add_argument("--w_length", type=float, default=0.10)

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/rv_grpo")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rv-grpo")
    parser.add_argument("--seed", type=int, default=42)

    # HuggingFace Hub
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push final model to HuggingFace Hub")
    parser.add_argument("--hub_username", type=str, default="jkanishkha0305")

    return parser.parse_args()


def load_model_from_sft(
    model_id: str,
    sft_checkpoint: str,
    logger,
):
    """Load SFT-finetuned model for GRPO training."""
    logger.info(f"Loading base model: {model_id}")
    logger.info(f"Loading SFT checkpoint: {sft_checkpoint}")

    # Load tokenizer from SFT checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        sft_checkpoint, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load and merge SFT LoRA weights
    if os.path.exists(os.path.join(sft_checkpoint, "adapter_config.json")):
        logger.info("Loading LoRA adapters from SFT checkpoint...")
        model = PeftModel.from_pretrained(model, sft_checkpoint)
        model = model.merge_and_unload()
        logger.info("LoRA adapters merged successfully")
    else:
        logger.info("No LoRA adapters found, using full model from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(
            sft_checkpoint,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    return model, tokenizer


def prepare_prompts_dataset(args, logger) -> Dataset:
    """Load prompt-only dataset for GRPO."""
    if args.prompts_hub_name:
        logger.info(f"Loading prompts from HuggingFace: {args.prompts_hub_name}")
        ds = load_dataset(args.prompts_hub_name, split="train")
    elif os.path.exists(args.prompts_dataset):
        logger.info(f"Loading prompts from disk: {args.prompts_dataset}")
        ds = load_from_disk(args.prompts_dataset)
    else:
        # Fallback: use MentalChat16K prompts directly
        logger.info("Fallback: Loading prompts from ShenLab/MentalChat16K")
        ds = load_dataset("ShenLab/MentalChat16K", split="train")
        ds = ds.map(
            lambda x: {"prompt": x["input"]},
            remove_columns=ds.column_names,
        )

    ds = ds.filter(lambda x: len(x["prompt"]) > 20)

    if args.max_samples and args.max_samples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.max_samples))

    logger.info(f"Loaded {len(ds)} prompts for GRPO training")
    return ds


def create_reward_fn(weights: dict):
    """Create a reward function with specific rubric weights."""
    from data.rubric_rewards import (
        check_open_question,
        check_emotion_reflection,
        check_no_premature_advice,
        check_validation_before_redirect,
        check_length_appropriate,
    )

    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []

        for prompt, completion in zip(prompts, completions):
            # Extract text from various formats
            if isinstance(completion, list):
                comp_text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, dict):
                comp_text = completion.get("content", str(completion))
            else:
                comp_text = str(completion)

            if isinstance(prompt, list):
                prompt_text = prompt[-1]["content"] if prompt else ""
            elif isinstance(prompt, dict):
                prompt_text = prompt.get("content", str(prompt))
            else:
                prompt_text = str(prompt)

            # Compute individual rubric scores
            r1 = check_open_question(comp_text)
            r2 = check_emotion_reflection(prompt_text, comp_text)
            r3 = check_no_premature_advice(comp_text)
            r4 = check_validation_before_redirect(comp_text)
            r5 = check_length_appropriate(comp_text)

            total = (
                weights["open_question"] * r1
                + weights["emotion_reflection"] * r2
                + weights["no_premature_advice"] * r3
                + weights["validation_before_redirect"] * r4
                + weights["length_appropriate"] * r5
            )

            rewards.append(total)

        return rewards

    return reward_fn


def save_config(args, output_dir: str, model_config: dict):
    config_path = Path(output_dir) / "training_config.json"
    config = {
        "method": "rv-grpo",
        "model": args.model,
        "model_id": model_config["model_id"],
        "sft_checkpoint": args.sft_checkpoint,
        "num_generations": args.num_generations or model_config["num_generations"],
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "kl_coef": args.kl_coef,
        "rubric_weights": {
            "open_question": args.w_open_question,
            "emotion_reflection": args.w_emotion_reflection,
            "no_premature_advice": args.w_no_premature_advice,
            "validation_before_redirect": args.w_validation,
            "length_appropriate": args.w_length,
        },
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def main():
    args = parse_args()

    model_config = MODEL_REGISTRY[args.model]
    model_id = model_config["model_id"]

    if args.batch_size is None:
        args.batch_size = model_config["recommended_batch_size"]
    if args.num_generations is None:
        args.num_generations = model_config["num_generations"]

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(args.model, str(output_dir))

    # Banner
    logger.info("=" * 60)
    logger.info("RV-GRPO: Rubric-Verifiable GRPO Training")
    logger.info("=" * 60)
    logger.info(f"Model:            {model_id}")
    logger.info(f"SFT Checkpoint:   {args.sft_checkpoint}")
    logger.info(f"Num Generations:  {args.num_generations}")
    logger.info(f"Learning Rate:    {args.learning_rate}")
    logger.info(f"KL Coefficient:   {args.kl_coef}")
    logger.info(f"Rubric Weights:")
    logger.info(f"  R1 (open_question):      {args.w_open_question}")
    logger.info(f"  R2 (emotion_reflection): {args.w_emotion_reflection}")
    logger.info(f"  R3 (no_premature_advice):{args.w_no_premature_advice}")
    logger.info(f"  R4 (validation):         {args.w_validation}")
    logger.info(f"  R5 (length):             {args.w_length}")
    logger.info("=" * 60)

    # Save config
    config_path = save_config(args, output_dir, model_config)
    logger.info(f"Config saved: {config_path}")

    # Init W&B
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"rv-grpo-{args.model}-{datetime.now().strftime('%m%d_%H%M')}",
            config=vars(args),
            tags=["rv-grpo", args.model, "rubric-rewards"],
        )

    # Step 1: Load model from SFT checkpoint
    logger.info("-" * 40)
    logger.info("Step 1: Loading SFT model")
    model, tokenizer = load_model_from_sft(model_id, args.sft_checkpoint, logger)

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_used = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"GPU: {gpu_used:.1f}GB / {gpu_mem:.1f}GB")

    # Step 2: Load prompt dataset
    logger.info("-" * 40)
    logger.info("Step 2: Loading prompt dataset")
    prompts_dataset = prepare_prompts_dataset(args, logger)

    # Step 3: Create rubric reward function
    logger.info("-" * 40)
    logger.info("Step 3: Setting up rubric reward function")

    rubric_weights = {
        "open_question": args.w_open_question,
        "emotion_reflection": args.w_emotion_reflection,
        "no_premature_advice": args.w_no_premature_advice,
        "validation_before_redirect": args.w_validation,
        "length_appropriate": args.w_length,
    }
    reward_fn = create_reward_fn(rubric_weights)
    logger.info("Rubric reward function ready")

    # Step 4: Configure GRPO
    logger.info("-" * 40)
    logger.info("Step 4: Configuring GRPO trainer")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=model_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed,
        # GRPO specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
    )

    # Step 5: Initialize trainer
    logger.info("-" * 40)
    logger.info("Step 5: Initializing GRPO Trainer")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=prompts_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Step 6: Train!
    logger.info("-" * 40)
    logger.info("Step 6: Starting RV-GRPO training!")
    logger.info("-" * 40)

    trainer.train()

    logger.info("Training complete!")

    # Step 7: Save final model
    logger.info("-" * 40)
    logger.info("Step 7: Saving final model")

    final_model_path = output_dir / "final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    logger.info(f"Model saved: {final_model_path}")

    # Push to HuggingFace Hub
    if args.push_to_hub:
        hub_repo = f"{args.hub_username}/rv-grpo-{args.model}"
        logger.info(f"Pushing to HuggingFace Hub: {hub_repo}")
        try:
            trainer.push_to_hub(hub_repo)
            logger.info(f"Pushed to: https://huggingface.co/{hub_repo}")
        except Exception as e:
            logger.warning(f"Hub push failed: {e}. Model saved locally at {final_model_path}")

    # Step 8: Quick sanity check — generate and score a few responses
    logger.info("-" * 40)
    logger.info("Step 8: Sanity check — generating sample responses")

    test_prompts = [
        "I've been feeling really depressed lately and I don't know what to do.",
        "My anxiety is getting worse and I can't sleep at night.",
        "I feel like nobody understands what I'm going through.",
        "I lost my job and I feel worthless.",
        "I keep having panic attacks and they scare me.",
    ]

    model.eval()
    for tp in test_prompts:
        inputs = tokenizer(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": tp}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        result = compute_rubric_reward(tp, response, return_breakdown=True)

        logger.info(f"\nPrompt: {tp[:60]}...")
        logger.info(f"Response: {response[:100]}...")
        logger.info(f"Score: {result['combined']:.3f}")
        for k, v in result["scores"].items():
            logger.info(f"  {k}: {v:.1f}")

    # Cleanup
    if args.use_wandb:
        import wandb
        wandb.finish()

    logger.info("=" * 60)
    logger.info("RV-GRPO TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Model:  {final_model_path}")
    logger.info(f"Config: {config_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
