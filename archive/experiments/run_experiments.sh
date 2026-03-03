#!/bin/bash
# ReFine-Lab Experiment Runner
# Run SFT + Alignment experiments for all models

set -e

# Default values
MAX_SAMPLES=1000
USE_WANDB=""
MODELS="gemma llama phi qwen"
METHODS="dpo simpo orpo kto"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --sft_only)
            METHODS=""
            shift
            ;;
        --alignment_only)
            SFT_SKIP=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "ReFine-Lab Experiment Runner"
echo "============================================"
echo "Models: $MODELS"
echo "Methods: $METHODS"
echo "Max samples: $MAX_SAMPLES"
echo "W&B: ${USE_WANDB:-disabled}"
echo "============================================"
echo ""

# Step 1: Run SFT for all models
if [ -z "$SFT_SKIP" ]; then
    echo ">>> PHASE 1: Supervised Fine-Tuning"
    echo "============================================"

    for model in $MODELS; do
        echo ""
        echo ">>> SFT Training: $model"
        echo "--------------------------------------------"
        python experiments/sft/train_sft.py \
            --model "$model" \
            --max_samples "$MAX_SAMPLES" \
            $USE_WANDB
        echo ">>> SFT Complete: $model"
    done
fi

# Step 2: Run alignment for all models and methods
if [ -n "$METHODS" ]; then
    echo ""
    echo ">>> PHASE 2: Alignment Training"
    echo "============================================"

    for method in $METHODS; do
        for model in $MODELS; do
            echo ""
            echo ">>> $method Training: $model"
            echo "--------------------------------------------"

            # Get SFT checkpoint path
            SFT_CHECKPOINT="./outputs/sft/$model/final"

            python "experiments/alignment/train_$method.py" \
                --model "$model" \
                --sft_checkpoint "$SFT_CHECKPOINT" \
                --max_samples "$MAX_SAMPLES" \
                $USE_WANDB

            echo ">>> $method Complete: $model"
        done
    done
fi

echo ""
echo "============================================"
echo "All experiments complete!"
echo "============================================"
echo ""
echo "Output locations:"
echo "  SFT models:  ./outputs/sft/{model}/final"
echo "  DPO models:  ./outputs/dpo/{model}/final"
echo "  SimPO models: ./outputs/simpo/{model}/final"
echo "  ORPO models: ./outputs/orpo/{model}/final"
echo "  KTO models:  ./outputs/kto/{model}/final"
echo ""
echo "Logs: ./outputs/{method}/logs/"
echo "============================================"
