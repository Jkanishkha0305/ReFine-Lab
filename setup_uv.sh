#!/bin/bash
# ReFine-Lab Setup Script using UV (faster alternative)
# Run this script on your rented GPU (A100/H100)

set -e

echo "============================================"
echo "ReFine-Lab Environment Setup (UV)"
echo "============================================"

# Install uv if not present
echo ""
echo "[1/5] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Check CUDA
echo ""
echo "[2/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "GPU check skipped"

# Create venv with uv
echo ""
echo "[3/5] Creating virtual environment..."
uv venv --python 3.11
source .venv/bin/activate

# Install PyTorch first
echo ""
echo "[4/5] Installing PyTorch with CUDA..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn separately (needs special handling)
echo ""
echo "[5/5] Installing dependencies..."
uv pip install packaging ninja
uv pip install flash-attn --no-build-isolation
uv pip install -r requirements.txt

# Verify
echo ""
echo "Verification:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers, peft, trl; print('All packages installed successfully!')"

echo ""
echo "Setup complete! Activate with: source .venv/bin/activate"
