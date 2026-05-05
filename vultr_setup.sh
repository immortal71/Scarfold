#!/bin/bash
# vultr_setup.sh — Run this ONCE on a fresh Vultr GPU instance to set up Scarfold.
#
# Usage (from your local machine):
#   ssh root@YOUR_VULTR_IP "bash -s" < vultr_setup.sh
#
# Or copy it first and run it:
#   scp vultr_setup.sh root@YOUR_VULTR_IP:~/
#   ssh root@YOUR_VULTR_IP "bash vultr_setup.sh"

set -e
echo "=== Scarfold GPU setup on Vultr ==="
echo "Host: $(hostname)  |  Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(nvidia-smi not found)"

# ── 1. System dependencies ────────────────────────────────────────────────────
apt-get update -q
apt-get install -y -q python3-pip python3-venv git wget curl

# ── 2. Clone the repo ─────────────────────────────────────────────────────────
cd ~
if [ ! -d "Scarfold" ]; then
    git clone https://github.com/immortal71/Scarfold.git
fi
cd Scarfold
git pull origin main

# ── 3. Python virtual environment ─────────────────────────────────────────────
python3 -m venv .venv
source .venv/bin/activate

# ── 4. Install PyTorch with CUDA support ─────────────────────────────────────
# Detects CUDA version and installs matching PyTorch
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "12.1")
CUDA_SHORT=$(echo $CUDA_VER | tr -d '.' | cut -c1-3)
echo "Installing PyTorch for CUDA ${CUDA_VER} ..."

# PyTorch 2.x with CUDA 12.x (adjust if your Vultr instance has CUDA 11.x)
if [[ "$CUDA_SHORT" == "12"* ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
elif [[ "$CUDA_SHORT" == "11"* ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
else
    pip install torch torchvision -q
fi

# ── 5. Project dependencies ───────────────────────────────────────────────────
pip install numpy plotly biopython scipy fair-esm -q

# ── 6. Verify GPU is visible to PyTorch ──────────────────────────────────────
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# ── 7. Create data directory ──────────────────────────────────────────────────
mkdir -p data/pdbs results

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python src/download_data.py --n 537 --out data/pdbs   # re-download PDB files"
echo "  # OR: scp -r your_local_data/pdbs root@IP:~/Scarfold/data/"
echo ""
echo "  # Upload your best checkpoint from local machine:"
echo "  # scp model_v9.pt.best_so_far.pt root@IP:~/Scarfold/"
echo ""
echo "  # Then train v10 on GPU (60 epochs takes ~25 minutes on GPU vs 10 hours on CPU):"
echo "  python src/train_v10.py --base-model model_v9.pt.best_so_far.pt --epochs 200 --device cuda"
