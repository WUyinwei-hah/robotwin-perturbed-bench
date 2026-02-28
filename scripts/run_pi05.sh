#!/bin/bash
# Run perturbed benchmark for Pi0.5 policy
#
# Usage:
#   bash scripts/run_pi05.sh [GPU_ID] [SETTINGS] [TASKS]
#
# Environment variables:
#   SKIP_EXPERT_CHECK=1    Skip expert demo verification (use when CuroboPlanner unavailable)
#   POLICY_CONFIG=<path>   Override policy config file (default: configs/pi05_robotwin2.yml)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$BENCH_ROOT"

GPU_ID="${1:-0}"
SETTINGS="${2:-}"
TASKS="${3:-}"

PYTHON="${PYTHON:-/gemini/code/envs/robotwin/bin/python}"

EXTRA_ARGS=""
if [ -n "$SETTINGS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --settings $SETTINGS"
fi
if [ -n "$TASKS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --tasks $TASKS"
fi
if [ "${SKIP_EXPERT_CHECK:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip-expert-check"
fi

echo "Running Pi0.5 benchmark on GPU $GPU_ID"
echo "Settings filter: ${SETTINGS:-all}"
echo "Tasks filter: ${TASKS:-all}"

PI05_CONFIG="${POLICY_CONFIG:-configs/pi05_robotwin2.yml}"

$PYTHON -m benchmark.eval_runner \
    --policy pi05 \
    --policy-config "$PI05_CONFIG" \
    --spec benchmark/benchmark_spec.json \
    --output results \
    --task-config demo_clean \
    --gpu "$GPU_ID" \
    $EXTRA_ARGS

echo "Pi0.5 benchmark complete."
