#!/bin/bash
set -e
cd /Users/ady/Code/clgprjcts/music
source .venv/bin/activate

for config in src/training/configs/*frac*.yaml; do
    run=$(python3 -c "import yaml; print(yaml.safe_load(open('$config'))['run_name'])")
    if [ -f "outputs/${run}/eval_metrics.json" ]; then
        echo "SKIP ${run} (already done)"
        continue
    fi
    echo "=== Training ${run} ==="
    python -m src.training.train --config "${config}" 2>&1 | tail -3
done
echo "=== All data efficiency runs done ==="
