#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, .env file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d .venv ]; then
  echo "==> Creating virtualenv (.venv)"
  python3 -m venv .venv
fi

source .venv/bin/activate
echo "==> Installing dependencies (this can take a few minutes the first time)"
pip install --upgrade pip -q
pip install -e ".[dev]" -q

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example"
  cp .env.example .env
  echo "    Now edit .env and paste in your Kaggle + Spotify credentials."
  echo "    See FINALSTEPS.md section 1 for exactly where to get each one."
else
  echo "==> .env already exists, leaving it alone"
fi

echo
python scripts/check_env.py
echo
echo "Setup done. Next: fill in .env (if not done), then run ./scripts/run_all.sh"
