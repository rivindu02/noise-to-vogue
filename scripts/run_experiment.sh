#!/usr/bin/env bash
# Train and evaluate a paired schedule experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
PYTHON=.venv/bin/python
"$PYTHON" -m pip install --no-cache-dir -e '.[test]'
"$PYTHON" -m pytest -q
GPU_LINEAR="${GPU_LINEAR:-0}" GPU_COSINE="${GPU_COSINE:-1}" PYTHON="$PYTHON" bash scripts/train_pair.sh --steps "${TRAIN_STEPS:-8000}"
CUDA_VISIBLE_DEVICES="${GPU_LINEAR:-0}" "$PYTHON" -m noise_to_vogue.evaluate --samples 1024
"$PYTHON" scripts/verify_training.py
"$PYTHON" scripts/smoke_test.py
"$PYTHON" scripts/record_environment.py
"$PYTHON" -m pytest -q > results/test-output.txt
printf '\nExperiment complete. Results: results/REPORT.md\n'
printf 'Start the viewer with: bash scripts/serve.sh\n'
