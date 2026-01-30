"""
Complete RLHF Pipeline Orchestration

Runs the full RLHF pipeline: SFT → Reward Model → PPO

Usage:
    python pipeline.py --model_name google/gemma-2-270m --run_all
    
    # Or run stages individually:
    python pipeline.py --model_name google/gemma-2-270m --stage sft
    python pipeline.py --model_name google/gemma-2-270m --stage reward_model
    python pipeline.py --model_name google/gemma-2-270m --stage ppo
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RLHF Pipeline Orchestration")
    
    parser.add_argument("--model_name", type=str, default="google/gemma-2-270m",
                       help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./outputs/rlhf_pipeline",
                       help="Root output directory")
    
    # Pipeline control
    parser.add_argument("--run_all", action="store_true",
                       help="Run all stages sequentially")
    parser.add_argument("--stage", type=str, choices=["sft", "reward_model", "ppo"],
                       help="Run specific stage only")
    
    # Dataset options
    parser.add_argument("--sft_dataset", type=str, default="yahma/alpaca-cleaned",
                       help="Dataset for SFT")
    parser.add_argument("--preference_dataset", type=str, default="Anthropic/hh-rlhf",
                       help="Dataset for reward model and PPO")
    
    # Training options
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Max samples per stage (for quick testing)")
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of epochs per stage")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable W&B tracking")
    
    return parser.parse_args()


def run_command(cmd, description):
    """Run a command and log output."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*80}\n")
    
    result = subprocess.run(cmd, check=True)
    
    if result.returncode == 0:
        logger.info(f"✅ {description} completed successfully!\n")
    else:
        logger.error(f"❌ {description} failed!")
        sys.exit(1)


def run_sft(args):
    """Stage 1: Supervised Fine-Tuning"""
    logger.info("\n🎯 Stage 1/3: Supervised Fine-Tuning (SFT)")
    
    sft_output = os.path.join(args.output_dir, "sft_model")
    
    cmd = [
        "python", "../../experiments/sft/train_lora.py",
        "--model_name", args.model_name,
        "--dataset", args.sft_dataset,
        "--output_dir", sft_output,
        "--max_samples", str(args.max_samples),
        "--num_epochs", str(args.num_epochs),
    ]
    
    if args.load_in_4bit:
        cmd.append("--load_in_4bit")
    if args.use_wandb:
        cmd.append("--use_wandb")
    
    run_command(cmd, "SFT Training")
    
    return sft_output


def run_reward_model(args):
    """Stage 2: Reward Model Training"""
    logger.info("\n🎯 Stage 2/3: Reward Model Training")
    
    rm_output = os.path.join(args.output_dir, "reward_model")
    
    cmd = [
        "python", "../../experiments/reward_model/train_rm.py",
        "--model_name", args.model_name,
        "--dataset", args.preference_dataset,
        "--output_dir", rm_output,
        "--max_samples", str(args.max_samples),
        "--num_epochs", str(args.num_epochs),
    ]
    
    if args.load_in_4bit:
        cmd.append("--load_in_4bit")
    if args.use_wandb:
        cmd.append("--use_wandb")
    
    run_command(cmd, "Reward Model Training")
    
    return rm_output


def run_ppo(args, sft_model_path, reward_model_path):
    """Stage 3: PPO Training"""
    logger.info("\n🎯 Stage 3/3: PPO Training with Reward Model")
    
    ppo_output = os.path.join(args.output_dir, "ppo_model")
    
    cmd = [
        "python", "../../experiments/ppo/train_ppo.py",
        "--model_name", os.path.join(sft_model_path, "final_model"),
        "--reward_model_name", os.path.join(reward_model_path, "final_model"),
        "--dataset", args.preference_dataset,
        "--output_dir", ppo_output,
        "--max_samples", str(args.max_samples),
        "--num_steps", "500",  # PPO uses steps not epochs
    ]
    
    if args.load_in_4bit:
        cmd.append("--load_in_4bit")
    if args.use_wandb:
        cmd.append("--use_wandb")
    
    run_command(cmd, "PPO Training")
    
    return ppo_output


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("🚀 RLHF Pipeline Starting")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"SFT Dataset: {args.sft_dataset}")
    logger.info(f"Preference Dataset: {args.preference_dataset}")
    logger.info("="*80 + "\n")
    
    if args.run_all:
        # Run complete pipeline
        logger.info("Running complete RLHF pipeline (all 3 stages)")
        
        sft_path = run_sft(args)
        rm_path = run_reward_model(args)
        ppo_path = run_ppo(args, sft_path, rm_path)
        
        logger.info("\n" + "="*80)
        logger.info("🎉 RLHF Pipeline Complete!")
        logger.info("="*80)
        logger.info(f"SFT Model: {sft_path}/final_model")
        logger.info(f"Reward Model: {rm_path}/final_model")
        logger.info(f"Final RLHF Model: {ppo_path}/final_model")
        logger.info("="*80 + "\n")
        
    elif args.stage:
        # Run specific stage
        if args.stage == "sft":
            sft_path = run_sft(args)
            logger.info(f"\n✅ SFT complete! Model saved to: {sft_path}/final_model")
            
        elif args.stage == "reward_model":
            rm_path = run_reward_model(args)
            logger.info(f"\n✅ Reward model complete! Model saved to: {rm_path}/final_model")
            
        elif args.stage == "ppo":
            # For PPO stage, need to specify existing SFT and RM paths
            sft_path = os.path.join(args.output_dir, "sft_model")
            rm_path = os.path.join(args.output_dir, "reward_model")
            
            if not os.path.exists(sft_path) or not os.path.exists(rm_path):
                logger.error("❌ SFT and Reward Model must be trained first!")
                logger.error(f"   Expected SFT at: {sft_path}")
                logger.error(f"   Expected RM at: {rm_path}")
                logger.error("   Run with --run_all or train stages 1-2 first")
                sys.exit(1)
            
            ppo_path = run_ppo(args, sft_path, rm_path)
            logger.info(f"\n✅ PPO complete! Model saved to: {ppo_path}/final_model")
    
    else:
        logger.error("Error: Must specify either --run_all or --stage")
        sys.exit(1)


if __name__ == "__main__":
    main()

