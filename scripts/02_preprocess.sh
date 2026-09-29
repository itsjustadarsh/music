#!/usr/bin/env bash
# Decodes FMA-small audio into cached log-mel spectrograms + 16kHz waveforms.
# Skips if a manifest already exists (pass FORCE=1 to redo).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

MANIFEST="data/processed/manifest.csv"
if [ -f "$MANIFEST" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "==> $MANIFEST already exists, skipping preprocessing (set FORCE=1 to redo)"
  exit 0
fi

WORKERS="${WORKERS:-4}"
echo "==> Preprocessing FMA-small with $WORKERS workers (this decodes ~8000 mp3s, expect it to take a while)"
python -m src.data.preprocess --workers "$WORKERS"

echo "Done. Manifest written to $MANIFEST"
