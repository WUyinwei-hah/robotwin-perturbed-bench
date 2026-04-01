#!/bin/bash
# Run perturbed benchmark for LingbotVA policy (scale-only settings).
#
# IMPORTANT: The LingbotVA server must be launched separately before running this.
# See: lingbot-va/evaluation/robotwin/launch_server.sh
#
# Usage:
#   bash scripts/run_lingbot_va_scale.sh [GPU_ID] [TASKS] [EXP_ID]
#
# Examples:
#   bash scripts/run_lingbot_va_scale.sh
#   bash scripts/run_lingbot_va_scale.sh 0 adjust_bottle,click_bell exp_20260304_scale_baseline_v1
#
# Environment variables:
#   SKIP_EXPERT_CHECK=1  Skip expert demo verification (use when CuroboPlanner unavailable)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$BENCH_ROOT"

GPU_ID="${1:-0}"
TASKS="${2:-}"
EXP_ID="${3:-}"

PYTHON="${PYTHON:-/gemini/code/envs/robotwin/bin/python}"
SPEC_PATH="${SPEC_PATH:-benchmark/benchmark_spec.json}"

# Scale-only settings from benchmark spec.
SCALE_SETTINGS="scale_lm_always_on,scale_lm_onset_then_always,scale_high_always_on,scale_high_onset_then_always"

# Bypass proxy for local websocket server.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

OUTPUT_DIR="results"
if [ -n "$EXP_ID" ]; then
    OUTPUT_DIR="results/$EXP_ID"
fi

EXTRA_ARGS="--settings $SCALE_SETTINGS"
if [ -n "$TASKS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --tasks $TASKS"
fi
if [ "${SKIP_EXPERT_CHECK:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip-expert-check"
fi

echo "Running LingbotVA scale-only benchmark on GPU $GPU_ID"
echo "Spec: $SPEC_PATH"
echo "Settings: $SCALE_SETTINGS"
echo "Tasks filter: ${TASKS:-all}"
echo "Output dir: $OUTPUT_DIR"
echo "NOTE: Ensure the LingbotVA server is running!"

$PYTHON -m benchmark.eval_runner \
    --policy lingbot_va \
    --policy-config configs/lingbot_va.yml \
    --spec "$SPEC_PATH" \
    --output "$OUTPUT_DIR" \
    --task-config demo_clean \
    --gpu "$GPU_ID" \
    $EXTRA_ARGS

echo "LingbotVA scale-only benchmark complete."
