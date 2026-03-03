#!/bin/bash
# ============================================================
# ReFine-Lab: One-Command Setup for Cloud GPU (Lambda Labs)
# ============================================================
#
# Usage (on a fresh Lambda Labs instance):
#   git clone https://github.com/jkanishkha0305/ReFine-Lab.git
#   cd ReFine-Lab
#   cp .env.example .env   # fill in your tokens
#   bash setup.sh
#
# ============================================================

set -e

echo ""
echo "============================================"
echo "  ReFine-Lab Environment Setup"
echo "============================================"
echo ""

# Step 1: Check GPU
echo "[1/7] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: No GPU detected."
fi
echo ""

# Step 2: Create venv
echo "[2/7] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "  Existing .venv found, reusing..."
else
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "  Python: $(python3 --version)"
echo ""

# Step 3: Upgrade pip
echo "[3/7] Upgrading pip..."
pip install --upgrade pip wheel setuptools --quiet
echo ""

# Step 4: Install PyTorch + CUDA
echo "[4/7] Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
echo ""

# Step 5: Install flash-attention
echo "[5/7] Installing flash-attention..."
pip install packaging ninja --quiet
pip install flash-attn --no-build-isolation --quiet 2>/dev/null || {
    echo "  flash-attn build failed (optional). Training will use default attention."
}
echo ""

# Step 6: Install project
echo "[6/7] Installing project dependencies..."
pip install -e ".[all]" --quiet 2>/dev/null || pip install -r requirements.txt --quiet
echo ""

# Step 7: Auth (HuggingFace + W&B)
echo "[7/7] Setting up authentication..."
if [ -f .env ]; then
    echo "  Loading .env file..."
    set -a && source .env && set +a
fi

if [ -n "$HF_TOKEN" ]; then
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null && \
        echo "  HuggingFace: logged in" || echo "  HuggingFace: login failed"
else
    echo "  HF_TOKEN not set. Run: huggingface-cli login"
fi

if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" --relogin 2>/dev/null && \
        echo "  W&B: logged in" || echo "  W&B: login failed"
else
    echo "  WANDB_API_KEY not set. Run: wandb login"
fi
echo ""

# Verify
echo "============================================"
echo "  Verification"
echo "============================================"
python3 -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA:          {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:           {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory:    {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

import transformers, peft, trl, datasets
print(f'  Transformers:  {transformers.__version__}')
print(f'  TRL:           {trl.__version__}')
print(f'  PEFT:          {peft.__version__}')
print()
print('  All packages OK!')
"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Activate:  source .venv/bin/activate"
echo ""
echo "  Quick start:"
echo "    make check-gpu       # verify GPU"
echo "    make test-rewards    # test rubric rewards"
echo "    make pilot           # run pilot experiment (~2hrs)"
echo ""
echo "  Full pipeline:"
echo "    make train-sft       # SFT all 3 models"
echo "    make train-grpo      # RV-GRPO all 3 models"
echo "    make train-dpo       # DPO baseline"
echo "    make eval            # run evaluation"
echo ""
