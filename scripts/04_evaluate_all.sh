#!/usr/bin/env bash
# Evaluates every trained run, builds the published-baseline comparison table,
# and patches docs/RESULTS.md with the real numbers automatically.
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

echo "==> Building published-baseline comparison"
python -m src.evaluation.compare_baselines --runs "${RUN_NAMES[@]}"

echo "==> Updating docs/RESULTS.md with real numbers"
python scripts/update_results.py

echo "Done. See docs/RESULTS.md, outputs/comparison_table.md, and docs/figures/*.png"
