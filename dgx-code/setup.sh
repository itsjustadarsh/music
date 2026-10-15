#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing dependencies ==="
pip install torch torchaudio transformers datasets librosa \
    scikit-learn peft accelerate bitsandbytes \
    pyyaml pandas numpy matplotlib seaborn \
    einops timm audiomentations

echo "=== Downloading FMA-small ==="
python -m data.download

echo "=== Setup complete ==="
