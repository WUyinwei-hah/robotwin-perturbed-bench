#!/bin/bash
# Run perturbed benchmark for LingbotVA policy
#
# IMPORTANT: The LingbotVA server must be launched separately before running this.
# See: lingbot-va/evaluation/robotwin/launch_server.sh
#
# Usage:
#   bash scripts/run_lingbot_va.sh [GPU_ID] [SETTINGS] [TASKS]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$BENCH_ROOT"

GPU_ID="${1:-0}"
SETTINGS="${2:-}"
TASKS="${3:-}"

PYTHON="${PYTHON:-/gemini/code/envs/robotwin/bin/python}"

# Bypass proxy for local websocket server (same pattern as lingbot-va experiments)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

EXTRA_ARGS=""
if [ -n "$SETTINGS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --settings $SETTINGS"
fi
if [ -n "$TASKS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --tasks $TASKS"
fi

echo "Running LingbotVA benchmark on GPU $GPU_ID"
echo "Settings filter: ${SETTINGS:-all}"
echo "Tasks filter: ${TASKS:-all}"
echo "NOTE: Ensure the LingbotVA server is running!"

$PYTHON -m benchmark.eval_runner \
    --policy lingbot_va \
    --policy-config configs/lingbot_va.yml \
    --spec benchmark/benchmark_spec.json \
    --output results \
    --task-config demo_clean \
    --gpu "$GPU_ID" \
    $EXTRA_ARGS

echo "LingbotVA benchmark complete."
