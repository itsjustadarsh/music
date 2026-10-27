#!/bin/bash
set -e
cd /Users/ady/Code/clgprjcts/music
source .venv/bin/activate

for layer in $(seq 0 12); do
    for suffix in "" "_randomtrack" "_cliprandom"; do
        run="mert_layer${layer}${suffix}"
        config="src/training/configs/mert_layer${layer}${suffix}.yaml"
        if [ -f "outputs/${run}/eval_metrics.json" ]; then
            echo "SKIP ${run} (already done)"
            continue
        fi
        echo "=== Training ${run} ==="
        python -m src.training.train --config "${config}" 2>&1 | tail -3
    done
done
echo "=== All layer probes done ==="
