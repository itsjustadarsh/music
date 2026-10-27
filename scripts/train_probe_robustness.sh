#!/bin/bash
# Train all 16 probe robustness configs (4 probe types × 2 backbones × 2 protocols)
set -e
cd /Users/ady/Code/clgprjcts/music

CONFIGS=(
    clap_probe_linear
    clap_probe_linear_cliprandom
    clap_probe_mlp64
    clap_probe_mlp64_cliprandom
    clap_probe_mlp128_64
    clap_probe_mlp128_64_cliprandom
    clap_probe_mlp256_128
    clap_probe_mlp256_128_cliprandom
    mert_flat_linear
    mert_flat_linear_cliprandom
    mert_flat_mlp64
    mert_flat_mlp64_cliprandom
    mert_flat_mlp128_64
    mert_flat_mlp128_64_cliprandom
    mert_flat_mlp256_128
    mert_flat_mlp256_128_cliprandom
)

for cfg in "${CONFIGS[@]}"; do
    if [ -f "outputs/${cfg}/summary.json" ]; then
        echo "SKIP ${cfg} (already done)"
        continue
    fi
    echo "====== Training ${cfg} ======"
    python3 -m src.training.train --config "src/training/configs/${cfg}.yaml"
    echo ""
done

echo "All probe robustness training complete!"
