#!/bin/bash
#SBATCH --job-name=rvgrpo-align
#SBATCH --partition=a100_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/alignment_%j.out
#SBATCH --error=slurm_logs/alignment_%j.err

# ============================================================
# RV-GRPO: Alignment Training — RV-GRPO + DPO + LLM-Judge
# ============================================================
# Requires SFT checkpoints from 01_sft.sbatch
#
# Strategy: Run pairs in parallel on 2 GPUs where possible.
# Order: RV-GRPO first (main method), then DPO, then Judge-GRPO.
#
# Estimated time: ~5-6 hours
#   - RV-GRPO: ~2 hrs per model (6 hrs sequential, 3 hrs with 2 GPUs)
#   - DPO: ~1 hr per model
#   - Judge-GRPO: ~2 hrs per model
#
# Usage:
#   sbatch slurm/02_alignment.sbatch
# ============================================================

set -e

echo "=========================================="
echo "  RV-GRPO Alignment Training — $(date)"
echo "  Node: $(hostname), Job: $SLURM_JOB_ID"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

module purge
module load python/3.10 cuda/12.1 2>/dev/null || true

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

if [ -f .env ]; then
    set -a && source .env && set +a
fi

TRAIN_FLAGS="--use_wandb --push_to_hub"

# ── RV-GRPO (Main Method) ──────────────────────────────────
echo ""
echo "=== RV-GRPO Training ==="
echo ""

# LLaMA + Qwen in parallel
CUDA_VISIBLE_DEVICES=0 python3 experiments/grpo/train_rv_grpo.py \
    --model llama \
    --sft_checkpoint ./outputs/sft/llama/final_model \
    $TRAIN_FLAGS &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python3 experiments/grpo/train_rv_grpo.py \
    --model qwen \
    --sft_checkpoint ./outputs/sft/qwen/final_model \
    $TRAIN_FLAGS &
PID2=$!

wait $PID1 && echo "  RV-GRPO LLaMA complete!"
wait $PID2 && echo "  RV-GRPO Qwen complete!"

# Phi-4 alone (larger model)
CUDA_VISIBLE_DEVICES=0 python3 experiments/grpo/train_rv_grpo.py \
    --model phi \
    --sft_checkpoint ./outputs/sft/phi/final_model \
    $TRAIN_FLAGS
echo "  RV-GRPO Phi complete!"

# ── DPO Baseline ───────────────────────────────────────────
echo ""
echo "=== DPO Baseline Training ==="
echo ""

# LLaMA + Qwen in parallel
CUDA_VISIBLE_DEVICES=0 python3 experiments/alignment/train_dpo.py \
    --model llama $TRAIN_FLAGS &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python3 experiments/alignment/train_dpo.py \
    --model qwen $TRAIN_FLAGS &
PID2=$!

wait $PID1 && echo "  DPO LLaMA complete!"
wait $PID2 && echo "  DPO Qwen complete!"

CUDA_VISIBLE_DEVICES=0 python3 experiments/alignment/train_dpo.py \
    --model phi $TRAIN_FLAGS
echo "  DPO Phi complete!"

# ── LLM-Judge GRPO Baseline ────────────────────────────────
echo ""
echo "=== LLM-Judge GRPO Baseline ==="
echo ""

CUDA_VISIBLE_DEVICES=0 python3 experiments/grpo/train_llm_judge_grpo.py \
    --model llama \
    --sft_checkpoint ./outputs/sft/llama/final_model \
    $TRAIN_FLAGS &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python3 experiments/grpo/train_llm_judge_grpo.py \
    --model qwen \
    --sft_checkpoint ./outputs/sft/qwen/final_model \
    $TRAIN_FLAGS &
PID2=$!

wait $PID1 && echo "  Judge-GRPO LLaMA complete!"
wait $PID2 && echo "  Judge-GRPO Qwen complete!"

CUDA_VISIBLE_DEVICES=0 python3 experiments/grpo/train_llm_judge_grpo.py \
    --model phi \
    --sft_checkpoint ./outputs/sft/phi/final_model \
    $TRAIN_FLAGS
echo "  Judge-GRPO Phi complete!"

echo ""
echo "=========================================="
echo "  All alignment training complete! — $(date)"
echo "=========================================="
echo ""
echo "Next: sbatch slurm/03_eval.sbatch"
