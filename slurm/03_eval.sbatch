#!/bin/bash
#SBATCH --job-name=rvgrpo-eval
#SBATCH --partition=a100_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/eval_%j.out
#SBATCH --error=slurm_logs/eval_%j.err

# ============================================================
# RV-GRPO: Evaluation Suite
# ============================================================
# Requires all trained models from 01_sft and 02_alignment.
#
# Runs: behavioral metrics (R1-R5) + external benchmarks
# Estimated time: ~1-2 hours
#
# Usage:
#   sbatch slurm/03_eval.sbatch
# ============================================================

set -e

echo "=========================================="
echo "  RV-GRPO Evaluation — $(date)"
echo "  Node: $(hostname), Job: $SLURM_JOB_ID"
echo "=========================================="

module purge
module load python/3.10 cuda/12.1 2>/dev/null || true

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

if [ -f .env ]; then
    set -a && source .env && set +a
fi

echo ""
echo "=== Behavioral Metrics (R1-R5) ==="
make eval-rubric

echo ""
echo "=== External Benchmarks ==="
make eval-benchmarks

echo ""
echo "=== Inference Benchmarks ==="
make eval-inference

echo ""
echo "=========================================="
echo "  Evaluation complete! — $(date)"
echo "=========================================="
echo "Results saved to: ./results/"
