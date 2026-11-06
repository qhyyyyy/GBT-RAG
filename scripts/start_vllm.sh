#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/start_vllm.sh MODEL_DIRECTORY [GPU_ID=2] [PORT=8001] [SERVED_NAME=gbtrag-llama8b]" >&2
  exit 2
fi
MODEL_DIRECTORY="$1"
GBTRAG_GPU_ID="${2:-2}"
GBTRAG_PORT="${3:-8001}"
GBTRAG_SERVED_NAME="${4:-gbtrag-llama8b}"
[[ -d "$MODEL_DIRECTORY" ]] || { echo "Model directory missing: $MODEL_DIRECTORY" >&2; exit 2; }
CUDA_VISIBLE_DEVICES="$GBTRAG_GPU_ID" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIRECTORY" \
  --served-model-name "$GBTRAG_SERVED_NAME" \
  --generation-config vllm \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.50 \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --dtype auto \
  --host 127.0.0.1 \
  --port "$GBTRAG_PORT"
