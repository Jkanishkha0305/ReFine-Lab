"""
PPO (Proximal Policy Optimization) Training for RLHF

Train a language model using PPO with a reward model.

Usage:
    python train_ppo.py \
        --model_name google/gemma-2-270m \
        --reward_model_name ./outputs/reward_model/final_model \
        --dataset Anthropic/hh-rlhf
"""

import argparse
import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model

from utils.model_loader import load_model_and_tokenizer
from utils.gpu_monitor import log_gpu_stats, print_gpu_info
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with PPO")
    
    parser.add_argument("--model_name", type=str, default="google/gemma-2-270m",
                       help="Base model or SFT model path")
    parser.add_argument("--reward_model_name", type=str, required=True,
                       help="Path to trained reward model")
    parser.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true", default=True)
    
    parser.add_argument("--output_dir", type=str, default="./outputs/ppo")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=1000)
    
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print_gpu_info()
    log_gpu_stats(prefix="[Initial] ")
    
    logger.info(f"Loading policy model: {args.model_name}")
    
    # Load policy model
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_flash_attention=True,
    )
    
    # Wrap model for PPO (adds value head)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
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
    
    # Load reward model
    logger.info(f"Loading reward model: {args.reward_model_name}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
    
    log_gpu_stats(prefix="[After Model Load] ")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    # Extract prompts (for PPO we generate responses)
    def extract_prompt(example):
        # Assuming dataset has 'chosen' field with full conversation
        # Extract just the prompt part
        text = example.get("chosen", example.get("prompt", ""))
        # Simple heuristic: take first part before response
        if "Human:" in text and "Assistant:" in text:
            prompt = text.split("Assistant:")[0] + "Assistant:"
        else:
            prompt = text[:min(len(text)//2, 256)]  # Take first half as prompt
        return {"query": prompt}
    
    dataset = dataset.map(extract_prompt)
    
    logger.info(f"Training samples: {len(dataset)}")
    
    # PPO Configuration
    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        log_with="wandb" if args.use_wandb else None,
        seed=args.seed,
    )
    
    # Initialize PPO trainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    
    # Training loop
    logger.info("Starting PPO training...")
    
    for step, batch in enumerate(trainer.dataloader):
        if step >= args.num_steps:
            break
        
        # Get queries
        query_tensors = batch["input_ids"]
        
        # Generate responses
        response_tensors = trainer.generate(
            query_tensors,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        # Decode responses
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) 
                    for r in response_tensors]
        
        # Get rewards from reward model
        rewards = []
        for response in responses:
            inputs = reward_tokenizer(response, return_tensors="pt", 
                                     padding=True, truncation=True, 
                                     max_length=args.max_length).to(reward_model.device)
            with torch.no_grad():
                reward = reward_model(**inputs).logits.squeeze()
            rewards.append(reward)
        
        # Run PPO step
        stats = trainer.step(query_tensors, response_tensors, rewards)
        
        # Log
        if step % 10 == 0:
            logger.info(f"Step {step}: {stats}")
    
    logger.info("Training complete!")
    
    # Save model
    output_model_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    
    logger.info(f"Model saved to: {output_model_dir}")
    logger.info("✅ PPO training complete!")


if __name__ == "__main__":
    main()

