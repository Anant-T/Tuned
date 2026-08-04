#!/usr/bin/env bash
# One-shot setup on a fresh Lightning Studio. Idempotent - safe to re-run.
set -euo pipefail

if [ ! -d "$HOME/tuned" ]; then
  git clone https://github.com/Anant-T/Tuned "$HOME/tuned"
fi
cd "$HOME/tuned"
git pull --ff-only

pip install --quiet uv
uv pip install --system -e ".[dev,train]"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: set HF_TOKEN in the Studio environment (Settings -> Environment variables)" >&2
  exit 1
fi

python -m pytest tests/ -q
echo "bootstrap OK - next: python -m tuned.data.smoke && python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke"
