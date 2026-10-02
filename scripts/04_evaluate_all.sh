#!/usr/bin/env bash
# Evaluates every trained run, then regenerates docs/RESULTS.md and
# docs/EXPERIMENT_LOG.md from outputs/. Those two files are written in full and
# must never be hand-edited.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

RUN_NAMES=()
for cfg in src/training/configs/*.yaml; do
  run_name=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$cfg")
  if [ -f "outputs/${run_name}/best.pt" ]; then
    RUN_NAMES+=("$run_name")
  fi
done

if [ ${#RUN_NAMES[@]} -eq 0 ]; then
  echo "No trained runs found under outputs/ -- run ./scripts/03_train_all.sh first."
  exit 1
fi

for run_name in "${RUN_NAMES[@]}"; do
  echo "==> Evaluating $run_name"
  python -m src.evaluation.metrics --run "$run_name"
done

echo "==> Regenerating docs/RESULTS.md and docs/EXPERIMENT_LOG.md"
python scripts/update_results.py

echo "Done. See docs/RESULTS.md, docs/EXPERIMENT_LOG.md and docs/figures/*.png"
