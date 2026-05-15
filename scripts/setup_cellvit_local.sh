#!/usr/bin/env bash
# Setup local CellViT inference environment.
#
# Creates conda env "cellvit" (Python 3.10), clones the CellViT repo,
# installs its dependencies, and downloads the CellViT-256 checkpoint.
#
# Usage:
#   bash scripts/setup_cellvit_local.sh [--model CellViT-256|CellViT-SAM-H]
#
# Env vars (override defaults):
#   CELLVIT_ENV       conda env name           (default: cellvit)
#   CELLVIT_REPO      clone target             (default: $HOME/CellViT)
#   CELLVIT_CKPT_DIR  checkpoint directory     (default: $HOME/checkpoints)

set -euo pipefail

MODEL="${MODEL:-CellViT-256}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

CELLVIT_ENV="${CELLVIT_ENV:-cellvit}"
CELLVIT_REPO="${CELLVIT_REPO:-$HOME/CellViT}"
CELLVIT_CKPT_DIR="${CELLVIT_CKPT_DIR:-$HOME/checkpoints}"

declare -A GDRIVE_IDS=(
    [CellViT-256]="1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q"
    [CellViT-SAM-H]="1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV"
)
GDRIVE_ID="${GDRIVE_IDS[$MODEL]:-}"
if [[ -z "$GDRIVE_ID" ]]; then
    echo "Unknown MODEL=$MODEL (expected CellViT-256 or CellViT-SAM-H)" >&2
    exit 1
fi

# -- conda env -----------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$CELLVIT_ENV"; then
    echo ">>> Creating conda env $CELLVIT_ENV (python=3.10)"
    conda create -y -n "$CELLVIT_ENV" python=3.10
else
    echo ">>> Reusing existing conda env $CELLVIT_ENV"
fi
conda activate "$CELLVIT_ENV"

# -- clone repo ----------------------------------------------------------------
if [[ ! -d "$CELLVIT_REPO/.git" ]]; then
    echo ">>> Cloning CellViT into $CELLVIT_REPO"
    git clone --quiet https://github.com/TIO-IKIM/CellViT.git "$CELLVIT_REPO"
else
    echo ">>> CellViT repo already present at $CELLVIT_REPO"
fi

# -- python deps ---------------------------------------------------------------
# CellViT requires pydantic 1.x; install torch first with CUDA wheels.
echo ">>> Installing CellViT requirements"
pip install --upgrade pip
# CUDA 12.1 wheels work for T4/A10/A100. Switch index URL if your driver differs.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.2 torchvision==0.16.2
pip install -r "$CELLVIT_REPO/requirements.txt"
pip install "pydantic==1.10.4" gdown tqdm

# -- checkpoint ----------------------------------------------------------------
mkdir -p "$CELLVIT_CKPT_DIR"
CKPT_PATH="$CELLVIT_CKPT_DIR/$MODEL.pth"
if [[ -f "$CKPT_PATH" ]]; then
    echo ">>> Checkpoint already present: $CKPT_PATH ($(du -h "$CKPT_PATH" | cut -f1))"
else
    echo ">>> Downloading $MODEL checkpoint to $CKPT_PATH"
    gdown "https://drive.google.com/uc?id=$GDRIVE_ID" -O "$CKPT_PATH"
fi

echo
echo "Setup complete."
echo "  env:        $CELLVIT_ENV"
echo "  repo:       $CELLVIT_REPO"
echo "  checkpoint: $CKPT_PATH"
echo
echo "Run inference:"
echo "  conda activate $CELLVIT_ENV"
echo "  python stages/run_cellvit_local.py \\"
echo "    --zip processed/he.zip \\"
echo "    --out processed/cellvit \\"
echo "    --checkpoint $CKPT_PATH \\"
echo "    --cellvit-repo $CELLVIT_REPO"
