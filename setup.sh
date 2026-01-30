#!/bin/bash
# ReFine-Lab Setup Script for GPU Instance
# Run this script on your rented GPU (A100/H100)

set -e  # Exit on error

echo "============================================"
echo "ReFine-Lab Environment Setup"
echo "============================================"

# Check CUDA availability
echo ""
echo "[1/6] Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. CUDA may not be available."
fi

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "[3/6] Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support first
echo ""
echo "[4/6] Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attention (requires special handling)
echo ""
echo "[5/6] Installing flash-attention..."
pip install packaging ninja
pip install flash-attn --no-build-isolation

# Install remaining requirements
echo ""
echo "[6/6] Installing remaining dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "============================================"
echo "Verifying Installation..."
echo "============================================"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers
print(f'Transformers version: {transformers.__version__}')

import peft
print(f'PEFT version: {peft.__version__}')

import trl
print(f'TRL version: {trl.__version__}')

import bitsandbytes
print(f'BitsAndBytes version: {bitsandbytes.__version__}')
"

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run SFT training:"
echo "  python experiments/sft/train_sft.py --model gemma --max_samples 1000"
echo ""
