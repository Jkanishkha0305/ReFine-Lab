#!/bin/bash
#SBATCH --job-name=rvgrpo-setup
#SBATCH --partition=a100_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/setup_%j.out
#SBATCH --error=slurm_logs/setup_%j.err

# ============================================================
# RV-GRPO: Environment Setup on BigPurple
# ============================================================
# Run once before training:
#   mkdir -p slurm_logs && sbatch slurm/00_setup.sbatch
# ============================================================

set -e

echo "=========================================="
echo "  RV-GRPO Setup — $(date)"
echo "  Node: $(hostname), Job: $SLURM_JOB_ID"
echo "=========================================="

# Load modules (adjust to your cluster's module system)
module purge
module load python/3.10 cuda/12.1 2>/dev/null || true

cd $SLURM_SUBMIT_DIR

# Run the standard setup
bash setup.sh

echo ""
echo "Setup complete! Now submit training jobs."
echo "  sbatch slurm/01_sft.sbatch"
