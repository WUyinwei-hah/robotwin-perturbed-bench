#!/bin/bash
# Run perturbed benchmark for MotusV2 policy (video-conditioned)
#
# Usage:
#   bash scripts/run_motus_v2.sh [GPU_ID] [SETTINGS] [TASKS]
#
# Examples:
#   bash scripts/run_motus_v2.sh 1                           # all settings, all tasks
#   bash scripts/run_motus_v2.sh 1 scale_lm_always_on        # single setting
#   bash scripts/run_motus_v2.sh 1 "" adjust_bottle          # single task
#
# To use a different checkpoint:
#   CHECKPOINT_PATH=/path/to/ckpt bash scripts/run_motus_v2.sh 1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$BENCH_ROOT"

GPU_ID="${1:-1}"
SETTINGS="${2:-}"
TASKS="${3:-}"

PYTHON="${PYTHON:-/gemini/code/envs/robotwin_motus/bin/python}"

# Allow checkpoint override via env var
if [ -n "$CHECKPOINT_PATH" ]; then
    # Write a temp config with the overridden checkpoint
    TMP_CONFIG=$(mktemp /tmp/motus_v2_config_XXXXXX.yml)
    cat configs/motus_v2.yml | sed "s|checkpoint_path:.*|checkpoint_path: ${CHECKPOINT_PATH}|" > "$TMP_CONFIG"
    CONFIG="$TMP_CONFIG"
    trap "rm -f $TMP_CONFIG" EXIT
else
    CONFIG="configs/motus_v2.yml"
fi

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

echo "Running MotusV2 benchmark on GPU $GPU_ID"
echo "Config: $CONFIG"
echo "Settings filter: ${SETTINGS:-all}"
echo "Tasks filter: ${TASKS:-all}"

CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m benchmark.eval_runner \
    --policy motus_v2 \
    --policy-config "$CONFIG" \
    --spec benchmark/benchmark_spec.json \
    --output results \
    --task-config demo_clean \
    --gpu "$GPU_ID" \
    $EXTRA_ARGS

echo "MotusV2 benchmark complete."
