#!/bin/bash
# Run perturbed benchmark for Motus policy
#
# Usage:
#   bash scripts/run_motus.sh [GPU_ID] [SETTINGS] [TASKS]
#
# Examples:
#   bash scripts/run_motus.sh 0                          # all settings, all tasks
#   bash scripts/run_motus.sh 0 scale_lm_always_on       # single setting
#   bash scripts/run_motus.sh 0 "" adjust_bottle,click_bell  # specific tasks

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

echo "Running Motus benchmark on GPU $GPU_ID"
echo "Settings filter: ${SETTINGS:-all}"
echo "Tasks filter: ${TASKS:-all}"

$PYTHON -m benchmark.eval_runner \
    --policy motus \
    --policy-config configs/motus.yml \
    --spec benchmark/benchmark_spec.json \
    --output results \
    --task-config demo_clean \
    --gpu "$GPU_ID" \
    $EXTRA_ARGS

echo "Motus benchmark complete."
