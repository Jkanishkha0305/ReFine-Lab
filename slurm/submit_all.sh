#!/bin/bash
# ============================================================
# RV-GRPO: Submit Full Pipeline with Job Dependencies
# ============================================================
# This chains all jobs so each waits for the previous to finish.
#
# Usage:
#   bash slurm/submit_all.sh          # submit full pipeline
#   bash slurm/submit_all.sh --pilot  # submit pilot only
# ============================================================

set -e

mkdir -p slurm_logs

if [ "$1" == "--pilot" ]; then
    echo "Submitting PILOT run..."
    PILOT_ID=$(sbatch --parsable slurm/pilot.sbatch)
    echo "  Pilot job: $PILOT_ID"
    echo ""
    echo "Monitor with: squeue -u $USER"
    echo "View logs:    tail -f slurm_logs/pilot_${PILOT_ID}.out"
    exit 0
fi

echo "=========================================="
echo "  Submitting Full RV-GRPO Pipeline"
echo "=========================================="
echo ""

# Job 1: SFT (no dependency)
SFT_ID=$(sbatch --parsable slurm/01_sft.sbatch)
echo "  [1/3] SFT training:   Job $SFT_ID"

# Job 2: Alignment (depends on SFT)
ALIGN_ID=$(sbatch --parsable --dependency=afterok:$SFT_ID slurm/02_alignment.sbatch)
echo "  [2/3] Alignment:      Job $ALIGN_ID (after $SFT_ID)"

# Job 3: Evaluation (depends on Alignment)
EVAL_ID=$(sbatch --parsable --dependency=afterok:$ALIGN_ID slurm/03_eval.sbatch)
echo "  [3/3] Evaluation:     Job $EVAL_ID (after $ALIGN_ID)"

echo ""
echo "=========================================="
echo "  All jobs submitted!"
echo "=========================================="
echo ""
echo "  Monitor:    squeue -u $USER"
echo "  Cancel all: scancel $SFT_ID $ALIGN_ID $EVAL_ID"
echo "  SFT logs:   tail -f slurm_logs/sft_${SFT_ID}.out"
echo ""
echo "  Job chain: SFT ($SFT_ID) → Alignment ($ALIGN_ID) → Eval ($EVAL_ID)"
