#!/usr/bin/env bash
# Starts the FastAPI server and opens the frontend demo in your browser.
# Runs in the foreground -- Ctrl+C to stop.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

python scripts/check_env.py

PORT="${PORT:-8000}"
echo "==> Starting API on http://127.0.0.1:${PORT} (Ctrl+C to stop)"

if [[ "$(uname)" == "Darwin" ]]; then
  FRONTEND_URL="frontend/index.html"
  if [ "$PORT" != "8000" ]; then
    FRONTEND_URL="frontend/index.html?api=http://127.0.0.1:${PORT}"
  fi
  ( sleep 1.5 && open "$FRONTEND_URL" ) &
fi

uvicorn src.api.main:app --reload --port "$PORT"
