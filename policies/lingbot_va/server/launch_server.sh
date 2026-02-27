#!/bin/bash
# Launch LingbotVA inference server for benchmark evaluation.
#
# Prerequisites:
#   1. Clone the lingbot-va repo: git clone <lingbot-va-url>
#   2. Install dependencies (torch, transformers, etc.) in a Python env
#   3. Download model weights to MODEL_PATH
#
# Usage:
#   GPU_ID=0 MODEL_PATH=/path/to/lingbot-va-posttrain-robotwin \
#   LINGBOT_VA_ROOT=/path/to/lingbot-va \
#   bash policies/lingbot_va/server/launch_server.sh
#
# The server will listen on ws://127.0.0.1:${START_PORT} (default 8000).
# Once the server is ready, run the benchmark client:
#   bash scripts/run_lingbot_va.sh

set -e

LINGBOT_VA_ROOT=${LINGBOT_VA_ROOT:?'Set LINGBOT_VA_ROOT to lingbot-va repo path'}
MODEL_PATH=${MODEL_PATH:?'Set MODEL_PATH to model checkpoint directory'}
PYTHON=${PYTHON:-python}
START_PORT=${START_PORT:-8000}
MASTER_PORT=${MASTER_PORT:-29061}
GPU_ID=${GPU_ID:-0}

# Unset proxy to avoid intercepting local WebSocket connections
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

cd "$LINGBOT_VA_ROOT"

save_root='visualization/'
mkdir -p "$save_root"

echo "Launching LingbotVA server on GPU ${GPU_ID}, port ${START_PORT}..."
echo "Model: ${MODEL_PATH}"

CUDA_VISIBLE_DEVICES=${GPU_ID} $PYTHON -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root "$save_root"
