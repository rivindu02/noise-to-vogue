#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p logs
"$PYTHON" -m noise_to_vogue.prepare
# Set GPU_LINEAR and GPU_COSINE to available devices on your machine.
CUDA_VISIBLE_DEVICES="${GPU_LINEAR:-0}" "$PYTHON" -m noise_to_vogue.train --schedule linear "$@" > logs/linear.log 2>&1 &
linear_pid=$!
CUDA_VISIBLE_DEVICES="${GPU_COSINE:-1}" "$PYTHON" -m noise_to_vogue.train --schedule cosine "$@" > logs/cosine.log 2>&1 &
cosine_pid=$!
status=0
wait "$linear_pid" || status=1
wait "$cosine_pid" || status=1
exit "$status"
