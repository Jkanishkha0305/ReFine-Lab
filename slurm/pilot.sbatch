#!/bin/bash
#SBATCH --job-name=rvgrpo-pilot
#SBATCH --partition=a100_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/pilot_%j.out
#SBATCH --error=slurm_logs/pilot_%j.err

# ============================================================
# RV-GRPO: Pilot Run (Quick Validation)
# ============================================================
# Quick test on Qwen 1.5B only — ~2 hours
# Use this to verify everything works before full training.
#
# Usage:
#   mkdir -p slurm_logs && sbatch slurm/pilot.sbatch
# ============================================================

set -e

echo "=========================================="
echo "  RV-GRPO Pilot Run — $(date)"
echo "  Node: $(hostname), Job: $SLURM_JOB_ID"
echo "=========================================="

module purge
module load python/3.10 cuda/12.1 2>/dev/null || true

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

if [ -f .env ]; then
    set -a && source .env && set +a
fi

# Run pilot (SFT + RV-GRPO + DPO on Qwen only)
make pilot

# Run pilot evaluation
make pilot-eval

echo ""
echo "=========================================="
echo "  Pilot complete! — $(date)"
echo "=========================================="
echo "Check results in: ./results/pilot/"
echo "If pilot looks good, submit full training:"
echo "  sbatch slurm/01_sft.sbatch"
