#!/usr/bin/env bash
# Runs the entire pipeline end to end: fetch data -> preprocess -> train all 5
# configs -> evaluate all -> build the benchmark comparison -> update docs/RESULTS.md.
#
# This is the "just run this and wait" command. It will take a long time (the
# 7.2GB FMA download, decoding ~8000 mp3s, and training 5 model configs) -- run
# it in a terminal you can leave open, or in tmux/screen if you want to disconnect.
#
# Prerequisite: ./scripts/00_setup.sh has been run once, and .env is filled in
# (at least Kaggle; Spotify creds are only needed later for scripts/05_serve.sh's
# "push to Spotify" button, not for this script).
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
echo "# 3/4 Training all 5 model configs"
echo "############################################"
./scripts/03_train_all.sh

echo "############################################"
echo "# 4/4 Evaluating + building results"
echo "############################################"
./scripts/04_evaluate_all.sh

echo
echo "All done. Real results are now in docs/RESULTS.md and outputs/comparison_table.md."
echo "Next: ./scripts/05_serve.sh to start the API + demo frontend."
