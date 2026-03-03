#!/bin/bash
# ============================================================
# ReFine-Lab: Fast Setup using UV (10-100x faster than pip)
# ============================================================
#
# Usage:
#   git clone https://github.com/jkanishkha0305/ReFine-Lab.git
#   cd ReFine-Lab
#   cp .env.example .env   # fill in your tokens
#   bash setup_uv.sh
#
# ============================================================

set -e

echo ""
echo "============================================"
echo "  ReFine-Lab Setup (UV - Fast Mode)"
echo "============================================"
echo ""

# Step 1: Install uv
echo "[1/5] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv: $(uv --version)"
echo ""

# Step 2: Check GPU
echo "[2/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU detected"
echo ""

# Step 3: Create venv + PyTorch
echo "[3/5] Creating environment + installing PyTorch..."
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# Step 4: Install deps
echo "[4/5] Installing dependencies..."
uv pip install packaging ninja
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped"
uv pip install -e ".[all]" 2>/dev/null || uv pip install -r requirements.txt
echo ""

# Step 5: Auth
echo "[5/5] Authentication..."
if [ -f .env ]; then
    set -a && source .env && set +a
fi

[ -n "$HF_TOKEN" ] && python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null && echo "  HuggingFace: OK"
[ -n "$WANDB_API_KEY" ] && wandb login "$WANDB_API_KEY" --relogin 2>/dev/null && echo "  W&B: OK"
echo ""

# Verify
python3 -c "
import torch, transformers, peft, trl
print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | TRL {trl.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB)')
print('All packages OK!')
"

echo ""
echo "Setup complete! Run: source .venv/bin/activate && make pilot"
