#!/bin/bash
#SBATCH --job-name=rvgrpo-sft
#SBATCH --partition=a100_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/sft_%j.out
#SBATCH --error=slurm_logs/sft_%j.err

# ============================================================
# RV-GRPO: SFT Training — All 3 Models
# ============================================================
# Strategy: Run LLaMA (1B) and Qwen (1.5B) in PARALLEL on 2 GPUs,
# then Phi-4 (3.8B) on 1 GPU sequentially.
#
# Estimated time: ~2-3 hours total
#   - LLaMA 1B + Qwen 1.5B in parallel: ~1.5 hrs
#   - Phi-4 Mini 3.8B: ~1.5 hrs
#
# Usage:
#   mkdir -p slurm_logs && sbatch slurm/01_sft.sbatch
# ============================================================

set -e

echo "=========================================="
echo "  RV-GRPO SFT Training — $(date)"
echo "  Node: $(hostname), Job: $SLURM_JOB_ID"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Load modules
module purge
module load python/3.10 cuda/12.1 2>/dev/null || true

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    set -a && source .env && set +a
fi

TRAIN_FLAGS="--use_wandb --push_to_hub"

echo ""
echo "=== Phase 1: LLaMA 1B + Qwen 1.5B in parallel ==="
echo ""

# Run LLaMA on GPU 0 and Qwen on GPU 1 simultaneously
CUDA_VISIBLE_DEVICES=0 python3 experiments/sft/train_sft.py \
    --model llama $TRAIN_FLAGS &
PID_LLAMA=$!

CUDA_VISIBLE_DEVICES=1 python3 experiments/sft/train_sft.py \
    --model qwen $TRAIN_FLAGS &
PID_QWEN=$!

# Wait for both to complete
echo "  LLaMA PID: $PID_LLAMA, Qwen PID: $PID_QWEN"
wait $PID_LLAMA
echo "  LLaMA SFT complete!"
wait $PID_QWEN
echo "  Qwen SFT complete!"

echo ""
echo "=== Phase 2: Phi-4 Mini 3.8B (needs more memory) ==="
echo ""

# Phi-4 is larger, give it 1 GPU with full memory
# (or 2 GPUs with accelerate if needed)
CUDA_VISIBLE_DEVICES=0 python3 experiments/sft/train_sft.py \
    --model phi $TRAIN_FLAGS

echo ""
echo "=========================================="
echo "  All SFT training complete! — $(date)"
echo "=========================================="
echo ""
echo "Next: sbatch slurm/02_alignment.sbatch"
