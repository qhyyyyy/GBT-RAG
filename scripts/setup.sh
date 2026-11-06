#!/usr/bin/env bash
set -euo pipefail
GBTRAG_PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python -m pip install -r "$GBTRAG_PROJECT_ROOT/requirements.txt"
mkdir -p "$GBTRAG_PROJECT_ROOT/results" "$GBTRAG_PROJECT_ROOT/logs" "$GBTRAG_PROJECT_ROOT/cache"
echo "Client dependencies installed. torch, sentence-transformers and vLLM are separate server dependencies."
