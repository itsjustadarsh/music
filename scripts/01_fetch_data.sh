#!/usr/bin/env bash
# Downloads FMA-small + metadata (SHA1-verified) and the Kaggle Spotify Tracks
# Dataset, then validates row counts.
#
# Usage:
#   ./scripts/01_fetch_data.sh            # full download (~7.5GB total)
#   SKIP_AUDIO=1 ./scripts/01_fetch_data.sh   # metadata + Spotify only, skip the 7.2GB audio zip
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

echo "==> Checking credentials"
python scripts/check_env.py --require kaggle

FMA_ARGS=()
if [ "${SKIP_AUDIO:-0}" = "1" ]; then
  echo "==> SKIP_AUDIO=1: fetching FMA metadata only (no 7.2GB audio download)"
  echo "    NOTE: preprocessing/training need the actual audio -- this mode is only"
  echo "    for trying the metadata/mood/curator parts without the big download."
  FMA_ARGS+=(--skip-audio)
fi

echo "==> Fetching FMA-small"
python -m src.data.fetch_fma "${FMA_ARGS[@]}"

echo "==> Fetching Spotify Tracks Dataset"
python -m src.data.fetch_spotify

echo "==> Validating datasets"
python -m src.data.validate

echo "Done. See docs/DATA_CARD.md for the validation results."
