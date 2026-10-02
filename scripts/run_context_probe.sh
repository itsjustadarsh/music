#!/usr/bin/env bash
# Phase 2, lever 1: does a longer episode help?
#
# Every embedding in this project is computed from 3 seconds of audio. Genre is
# partly a long-timescale property -- form, instrumentation turnover, section
# changes -- so 3s may be a ceiling on achievable accuracy rather than a neutral
# choice (docs/METHODOLOGY.md §5). This re-extracts MERT-95M on a 10s grid and
# trains the flat probe on it with every other setting held fixed, so the window
# is the only varying factor.
#
# The extraction is a full pass over ~8,000 mp3s through MERT and takes hours. It
# is resumable: re-running skips tracks whose cache already has the right shape.
#
# Run this only when nothing else is using the GPU. Two jobs contending for the
# single MPS device took DenseNet from 350s/epoch to 1800s/epoch.
set -u
cd "$(dirname "$0")/.."

echo "=== EXTRACT MERT-95M @ 10s  ($(date +%H:%M:%S)) ==="
if ! .venv/bin/python -m src.data.extract_embeddings --episode-seconds 10 --workers 4; then
  echo "!!! EXTRACTION FAILED -- not training a probe on a partial cache"
  exit 1
fi

echo "=== TRAIN mert_flat_probe_10s  ($(date +%H:%M:%S)) ==="
.venv/bin/python -m src.training.train --config src/training/configs/mert_flat_probe_10s.yaml \
  && .venv/bin/python -m src.evaluation.metrics --run mert_flat_probe_10s \
  && .venv/bin/python -m src.evaluation.cache_outputs --run mert_flat_probe_10s

echo "=== DONE ($(date +%H:%M:%S)) ==="
echo "Compare against mert_flat_probe_v1 in docs/RESULTS.md after:"
echo "  python scripts/update_results.py"
