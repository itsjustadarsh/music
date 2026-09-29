#!/usr/bin/env bash
# Trains every config in src/training/configs/ (5 runs: 3 CNN, 2 AST) in sequence.
# Skips a config if its checkpoint already exists (pass FORCE=1 to retrain everything).
#
# Usage:
#   ./scripts/03_train_all.sh                # all 5 configs
#   CONFIGS="cnn_baseline ast_baseline" ./scripts/03_train_all.sh   # just these two
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

if [ -n "${CONFIGS:-}" ]; then
  CONFIG_FILES=()
  for name in $CONFIGS; do
    CONFIG_FILES+=("src/training/configs/${name}.yaml")
  done
else
  CONFIG_FILES=(src/training/configs/*.yaml)
fi

for cfg in "${CONFIG_FILES[@]}"; do
  run_name=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$cfg")
  ckpt="outputs/${run_name}/best.pt"

  if [ -f "$ckpt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "==> $run_name already trained ($ckpt exists), skipping (set FORCE=1 to retrain)"
    continue
  fi

  echo "==> Training $run_name ($cfg)"
  python -m src.training.train --config "$cfg"
done

echo "Done. See outputs/<run_name>/summary.json for each run's result."
