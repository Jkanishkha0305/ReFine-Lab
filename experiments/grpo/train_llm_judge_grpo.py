"""
LLM-Judge GRPO Baseline Training Script

GRPO with generic LLM-as-judge reward instead of rubric rewards.
This baseline demonstrates that RV-GRPO's rubric design matters,
not just "doing GRPO."

Uses a small reward model (e.g., deberta-v3-base for sentiment/empathy)
or a prompted LLM scoring approach.

Supported Models:
    - meta-llama/Llama-3.2-1B-Instruct
    - Qwen/Qwen2.5-1.5B-Instruct
    - microsoft/Phi-4-mini-instruct

Usage:
    python experiments/grpo/train_llm_judge_grpo.py \
        --model llama \
        --sft_checkpoint ./outputs/sft/llama/final_model \
        --judge_type reward_model

    python experiments/grpo/train_llm_judge_grpo.py \
        --model qwen \
        --sft_checkpoint ./outputs/sft/qwen/final_model \
        --judge_type prompted

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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================================================
# Model Registry
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

# Reward model options
JUDGE_MODELS = {
    "reward_model": {
        "model_id": "OpenAssistant/reward-model-deberta-v3-large-v2",
        "description": "General-purpose reward model trained on OASST data",
    },
    "empathy": {
        "model_id": "SamLowe/roberta-base-go_emotions",
        "description": "Emotion classification for empathy scoring",
    },
}


def setup_logging(model_name: str, output_dir: str) -> logging.Logger:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"llm_judge_grpo_{model_name}_{timestamp}.log"

    logger = logging.getLogger(f"llm_judge_grpo_{model_name}")
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
        description="LLM-Judge GRPO Baseline Training"
    )

    # Model
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--sft_checkpoint", type=str, required=True,
        help="Path to SFT checkpoint",
    )

    # Judge
    parser.add_argument(
        "--judge_type", type=str, default="reward_model",
        choices=["reward_model", "empathy", "prompted"],
        help="Type of LLM judge to use as reward",
    )
    parser.add_argument(
        "--judge_model", type=str, default=None,
        help="Override judge model ID",
    )

    # Dataset
    parser.add_argument(
        "--prompts_dataset", type=str, default="./data/processed/grpo_prompts",
    )
    parser.add_argument("--prompts_hub_name", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=2000)

    # GRPO Hyperparameters (same as RV-GRPO for fair comparison)
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--kl_coef", type=float, default=0.05)

    # LoRA (same config for fair comparison)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/llm_judge_grpo")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rv-grpo")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_model_from_sft(model_id: str, sft_checkpoint: str, logger):
    """Load SFT-finetuned model for GRPO training."""
    logger.info(f"Loading base model: {model_id}")
    logger.info(f"Loading SFT checkpoint: {sft_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(
        sft_checkpoint, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if os.path.exists(os.path.join(sft_checkpoint, "adapter_config.json")):
        logger.info("Loading LoRA adapters from SFT checkpoint...")
        model = PeftModel.from_pretrained(model, sft_checkpoint)
        model = model.merge_and_unload()
        logger.info("LoRA adapters merged successfully")
    else:
        logger.info("No LoRA adapters found, using full model")
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


def create_reward_model_judge(judge_model_id: str, logger):
    """Create a reward function using a pretrained reward model."""
    logger.info(f"Loading reward model: {judge_model_id}")

    reward_pipe = pipeline(
        "text-classification",
        model=judge_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        truncation=True,
        max_length=512,
    )

    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []

        for prompt, completion in zip(prompts, completions):
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

            # Format as conversation for reward model
            text = f"User: {prompt_text}\nAssistant: {comp_text}"

            try:
                result = reward_pipe(text[:512])
                score = result[0]["score"]
                # Normalize to [0, 1]
                rewards.append(score)
            except Exception:
                rewards.append(0.5)

        return rewards

    return reward_fn


def create_empathy_judge(judge_model_id: str, logger):
    """Create reward based on empathy/emotion classification."""
    logger.info(f"Loading empathy model: {judge_model_id}")

    emotion_pipe = pipeline(
        "text-classification",
        model=judge_model_id,
        device_map="auto",
        top_k=None,
        truncation=True,
        max_length=512,
    )

    # Positive empathetic emotions
    EMPATHY_LABELS = {"caring", "love", "gratitude", "optimism", "approval"}
    # Negative/harmful labels
    NEGATIVE_LABELS = {"annoyance", "anger", "disgust", "disapproval"}

    def reward_fn(completions, **kwargs):
        rewards = []

        for completion in completions:
            if isinstance(completion, list):
                comp_text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, dict):
                comp_text = completion.get("content", str(completion))
            else:
                comp_text = str(completion)

            try:
                results = emotion_pipe(comp_text[:512])
                scores = {r["label"]: r["score"] for r in results[0]}

                empathy_score = sum(scores.get(l, 0) for l in EMPATHY_LABELS)
                negative_score = sum(scores.get(l, 0) for l in NEGATIVE_LABELS)

                # Reward = empathetic emotions - harmful ones, clamped to [0, 1]
                reward = max(0.0, min(1.0, empathy_score - negative_score))
                rewards.append(reward)
            except Exception:
                rewards.append(0.5)

        return rewards

    return reward_fn


def create_prompted_judge(logger):
    """
    Create reward using a prompted small LLM as judge.
    Uses a lightweight model to score responses on empathy 1-5.
    """
    logger.info("Loading prompted LLM judge (Qwen2.5-0.5B-Instruct)")

    judge_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_id)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    judge_model.eval()

    JUDGE_TEMPLATE = (
        "Rate the following therapeutic response on a scale of 1-5 for empathy "
        "and active listening quality.\n\n"
        "User message: {prompt}\n"
        "Response: {response}\n\n"
        "Rating (just the number 1-5):"
    )

    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []

        for prompt, completion in zip(prompts, completions):
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

            judge_prompt = JUDGE_TEMPLATE.format(
                prompt=prompt_text[:200], response=comp_text[:300]
            )

            inputs = judge_tokenizer(
                judge_prompt, return_tensors="pt", truncation=True, max_length=512,
            ).to(judge_model.device)

            try:
                with torch.no_grad():
                    outputs = judge_model.generate(
                        **inputs, max_new_tokens=5, do_sample=False,
                    )
                generated = judge_tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                # Extract score from generated text
                score = None
                for char in generated:
                    if char in "12345":
                        score = int(char)
                        break

                if score is None:
                    score = 3  # default middle score

                # Normalize 1-5 to 0-1
                rewards.append((score - 1) / 4.0)
            except Exception:
                rewards.append(0.5)

        return rewards

    return reward_fn


def save_config(args, output_dir: str, model_config: dict):
    config_path = Path(output_dir) / "training_config.json"
    config = {
        "method": "llm-judge-grpo",
        "model": args.model,
        "model_id": model_config["model_id"],
        "sft_checkpoint": args.sft_checkpoint,
        "judge_type": args.judge_type,
        "judge_model": args.judge_model,
        "num_generations": args.num_generations or model_config["num_generations"],
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "kl_coef": args.kl_coef,
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

    logger.info("=" * 60)
    logger.info("LLM-Judge GRPO Baseline Training")
    logger.info("=" * 60)
    logger.info(f"Model:           {model_id}")
    logger.info(f"SFT Checkpoint:  {args.sft_checkpoint}")
    logger.info(f"Judge Type:      {args.judge_type}")
    logger.info(f"Num Generations: {args.num_generations}")
    logger.info(f"Learning Rate:   {args.learning_rate}")
    logger.info("=" * 60)

    config_path = save_config(args, output_dir, model_config)
    logger.info(f"Config saved: {config_path}")

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"llm-judge-grpo-{args.model}-{datetime.now().strftime('%m%d_%H%M')}",
            config=vars(args),
            tags=["llm-judge-grpo", args.model, "baseline"],
        )

    # Load model
    logger.info("Loading SFT model...")
    model, tokenizer = load_model_from_sft(model_id, args.sft_checkpoint, logger)

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_used = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"GPU: {gpu_used:.1f}GB / {gpu_mem:.1f}GB")

    # Load prompts
    logger.info("Loading prompt dataset...")
    prompts_dataset = prepare_prompts_dataset(args, logger)

    # Create judge reward function
    logger.info(f"Setting up {args.judge_type} reward function...")
    judge_model_id = args.judge_model

    if args.judge_type == "reward_model":
        if judge_model_id is None:
            judge_model_id = JUDGE_MODELS["reward_model"]["model_id"]
        reward_fn = create_reward_model_judge(judge_model_id, logger)
    elif args.judge_type == "empathy":
        if judge_model_id is None:
            judge_model_id = JUDGE_MODELS["empathy"]["model_id"]
        reward_fn = create_empathy_judge(judge_model_id, logger)
    elif args.judge_type == "prompted":
        reward_fn = create_prompted_judge(logger)
    else:
        raise ValueError(f"Unknown judge type: {args.judge_type}")

    logger.info("Judge reward function ready")

    # Configure GRPO (identical config to RV-GRPO for fair comparison)
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
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
    )

    # Initialize trainer
    logger.info("Initializing GRPO Trainer with LLM-judge reward...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=prompts_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting LLM-Judge GRPO training...")
    trainer.train()
    logger.info("Training complete!")

    # Save
    final_model_path = output_dir / "final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    logger.info(f"Model saved: {final_model_path}")

    if args.use_wandb:
        import wandb
        wandb.finish()

    logger.info("=" * 60)
    logger.info("LLM-Judge GRPO Training Complete!")
    logger.info(f"Model:  {final_model_path}")
    logger.info(f"Config: {config_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
