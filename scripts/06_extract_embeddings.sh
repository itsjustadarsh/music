#!/usr/bin/env bash
# Caches frozen MERT-v1-95M embeddings for every episode in the preprocessing manifest.
# Skips if the embedding manifest already exists (pass FORCE=1 to redo).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

MANIFEST="data/embeddings/mert/manifest.csv"
if [ -f "$MANIFEST" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "==> $MANIFEST already exists, skipping extraction (set FORCE=1 to redo)"
  exit 0
fi

WORKERS="${WORKERS:-4}"
echo "==> Extracting MERT embeddings with $WORKERS decode workers (~45 min for 8000 tracks)"
python -m src.data.extract_embeddings --workers "$WORKERS"

echo "Done. Embedding manifest written to $MANIFEST"
