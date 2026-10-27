#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-.venv/bin/python}" -m uvicorn noise_to_vogue.server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-7860}"
