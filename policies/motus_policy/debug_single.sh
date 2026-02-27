#!/bin/bash
# Single-GPU debug script for Motus evaluation
# Usage: bash debug_single.sh [task_name] [gpu_id]
#   task_name: e.g. click_alarmclock (default: click_alarmclock)
#   gpu_id:    e.g. 0 (default: 0)

TASK_NAME="${1:-click_alarmclock}"
GPU_ID="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="/gemini/code/robotwin"
CHECKPOINT_PATH="/gemini/code/models/motus_pretrained_models/Motus_robotwin2"
WAN_PATH="/gemini/code/models/motus_pretrained_models/Wan2.2-TI2V-5B"
VLM_PATH="/gemini/code/models/motus_pretrained_models/Qwen3-VL-2B-Instruct"
CONDA_ENV="/gemini/code/envs/robotwin"
POLICY_NAME="Motus"
TASK_CONFIG="demo_randomized"
SEED="42"

echo "=== Motus Single-GPU Debug ==="
echo "Task:       $TASK_NAME"
echo "GPU:        $GPU_ID"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "=============================="

cd "$ROBOTWIN_ROOT" || exit 1

# Conda
export PATH="/gemini/code/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Log to both file and terminal
LOG_DIR="${SCRIPT_DIR}/logs_debug"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TASK_NAME}.log"

echo "Log file: $LOG_FILE"
echo ""

python script/eval_policy.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${CHECKPOINT_PATH}" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}"
