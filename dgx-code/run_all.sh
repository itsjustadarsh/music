#!/usr/bin/env bash
# Run all DGX experiments in recommended order.
#
# Priority order:
#   1. MERT-330M LoRA (most likely to beat 58.5%)
#   2. MERT-95M LoRA (fast, good baseline)
#   3. MERT-330M partial unfreeze (higher capacity)
#   4. AST fine-tune (different backbone for ensemble)
#   5. HTS-AT from scratch (architectural diversity)
#   6. EfficientAT (lightweight, ensemble filler)
#   7. MERT-330M multi-crop (longer context)
#   8. Ensemble of all successful runs
#
# Adjust --nproc_per_node to match available GPUs.

set -euo pipefail

NGPU=${NGPU:-4}
echo "=== Running with ${NGPU} GPUs ==="

# --- Phase 1: Primary runs ---
echo ""
echo "=== [1/7] MERT-330M LoRA ==="
torchrun --nproc_per_node=$NGPU train.py --config configs/mert_lora_330m.yaml

echo ""
echo "=== [2/7] MERT-95M LoRA ==="
torchrun --nproc_per_node=$NGPU train.py --config configs/mert_lora_95m.yaml

echo ""
echo "=== [3/7] MERT-330M Partial Unfreeze ==="
torchrun --nproc_per_node=$NGPU train.py --config configs/mert_unfreeze_330m.yaml

# --- Phase 2: Alternative architectures ---
echo ""
echo "=== [4/7] AST Fine-tune ==="
torchrun --nproc_per_node=$NGPU train.py --config configs/ast_finetune.yaml

echo ""
echo "=== [5/7] HTS-AT ==="
torchrun --nproc_per_node=$NGPU train.py --config configs/htsat_genre.yaml

echo ""
echo "=== [6/7] EfficientAT ==="
torchrun --nproc_per_node=$NGPU train.py --config configs/efficientat_genre.yaml

# --- Phase 3: Multi-crop (memory-intensive) ---
echo ""
echo "=== [7/7] MERT-330M Multi-crop ==="
torchrun --nproc_per_node=$NGPU train.py --config configs/mert_multicrop_330m.yaml

# --- Ensemble ---
echo ""
echo "=== Ensemble ==="
python ensemble.py \
    --runs outputs/mert_lora_330m outputs/mert_lora_95m outputs/ast_finetune outputs/htsat_genre

echo ""
echo "=== All runs complete. Check outputs/ for results. ==="
echo ""
echo "Quick results summary:"
for run in outputs/*/eval_metrics.json; do
    name=$(basename $(dirname $run))
    acc=$(python -c "import json; d=json.load(open('$run')); print(f\"{d['track_level']['accuracy']:.4f}\")" 2>/dev/null || echo "N/A")
    f1=$(python -c "import json; d=json.load(open('$run')); print(f\"{d['track_level']['macro_f1']:.4f}\")" 2>/dev/null || echo "N/A")
    printf "  %-35s acc=%s  f1=%s\n" "$name" "$acc" "$f1"
done
