#!/usr/bin/env bash
# Runs the pipeline end to end: fetch data -> preprocess -> train every config ->
# evaluate -> regenerate docs/RESULTS.md and docs/EXPERIMENT_LOG.md.
#
# This is the "just run this and wait" command, and the wait is long: a 7.2GB
# download, ~8000 mp3s to decode, then one training run per config, strictly
# sequential because they contend for a single MPS device. Run it under tmux or
# screen if you want to disconnect.
#
# It does not cover the protocol ladder or the frozen-backbone probes. For those:
#   scripts/06_extract_embeddings.sh   cache the backbone features once
#   scripts/run_ladder_queue.sh        the same configs down all three rungs
#
# Prerequisite: ./scripts/00_setup.sh has been run once and .env has Kaggle creds.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "############################################"
echo "# 1/4 Fetching data"
echo "############################################"
./scripts/01_fetch_data.sh

echo "############################################"
echo "# 2/4 Preprocessing audio"
echo "############################################"
./scripts/02_preprocess.sh

echo "############################################"
echo "# 3/4 Training every config in src/training/configs"
echo "############################################"
./scripts/03_train_all.sh

echo "############################################"
echo "# 4/4 Evaluating + building results"
echo "############################################"
./scripts/04_evaluate_all.sh

echo
echo "All done. Results are in docs/RESULTS.md; the per-run inventory is in"
echo "docs/EXPERIMENT_LOG.md. Both are generated — do not edit them by hand."
