#!/usr/bin/env bash
# Train every remaining architecture down the protocol ladder.
#
# Strictly sequential: these all contend for the single MPS device, and running two
# at once took DenseNet from 350s/epoch to 1800s/epoch.
#
# Ordered so the headline inflation table fills first. The artist-filtered and
# clip-random rungs give the total inflation per architecture; the random-track rung
# only refines the artist-vs-clip decomposition, so it runs last and can be cut.
set -u
cd "$(dirname "$0")/.."

run() {
  local cfg="$1"
  if [ -f "outputs/$cfg/eval_metrics.json" ]; then
    echo "=== SKIP $cfg (already evaluated) ==="
    return 0
  fi
  echo "=== TRAIN $cfg  ($(date +%H:%M:%S)) ==="
  if ! .venv/bin/python -m src.training.train --config "src/training/configs/$cfg.yaml"; then
    echo "!!! TRAIN FAILED: $cfg -- continuing to next run"
    return 0
  fi
  echo "=== EVAL $cfg ==="
  .venv/bin/python -m src.evaluation.metrics --run "$cfg" \
    || echo "!!! EVAL FAILED: $cfg"
}

# Tier 1 -- total inflation for the four architectures never trained at all.
for a in dual_cnn_v1 cnn_bigru_v1 capsnet_v1 cnn_transformer_v1; do
  run "$a"
  run "${a}_cliprandom"
done

# Tier 2 -- finish the two log-mel rows that have gaps.
run densenet_v1_cliprandom
run cnn_improved_randomtrack

# Tier 3 -- the middle rung, which only sharpens the decomposition.
run densenet_v1_randomtrack
for a in dual_cnn_v1 cnn_bigru_v1 capsnet_v1 cnn_transformer_v1; do
  run "${a}_randomtrack"
done

echo "=== LADDER QUEUE DONE ($(date +%H:%M:%S)) ==="
.venv/bin/python scripts/protocol_leakage_analysis.py
