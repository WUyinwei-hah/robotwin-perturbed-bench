#!/bin/bash
# Experiment 1: Baseline Motus evaluation on RoboTwin
# - 2 GPUs, 2 episodes per task
# - Saves real env video + model predicted video + combined video

echo "========================================"
echo "  Experiment: Motus BASELINE"
echo "  GPUs: 2 | Episodes per task: 2"
echo "  Started at $(date)"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="/gemini/code/robotwin"
CHECKPOINT_PATH="/gemini/code/models/motus_pretrained_models/Motus_robotwin2"
WAN_PATH="/gemini/code/models/motus_pretrained_models/Wan2.2-TI2V-5B"
VLM_PATH="/gemini/code/models/motus_pretrained_models/Qwen3-VL-2B-Instruct"
CONDA_ENV="/gemini/code/envs/robotwin"
POLICY_NAME="Motus"
TASK_CONFIG="demo_randomized"
SEED="42"
TEST_NUM=2

# GPU configuration - use exactly 2 GPUs
GPU_IDS=(0 1)

# Experiment output directory
EXP_NAME="baseline"
LOG_DIR="${SCRIPT_DIR}/experiments/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"

# Setup environment
cd "$ROBOTWIN_ROOT" || exit 1
export PATH="/gemini/code/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8

# Load tasks
TASKS_PATH="${SCRIPT_DIR}/tasks_all.txt"
if [ ! -f "$TASKS_PATH" ]; then
    echo "Error: Tasks file not found: $TASKS_PATH"
    exit 1
fi
mapfile -t tasks < "$TASKS_PATH"
total=${#tasks[@]}
echo "Total tasks: $total"
echo "GPUs: ${GPU_IDS[*]}"
echo ""

# GPU management
declare -A gpu_pid
for gpu_id in "${GPU_IDS[@]}"; do
    gpu_pid[$gpu_id]=""
done

is_running() {
    [ -n "$1" ] && kill -0 "$1" 2>/dev/null
}

get_free_gpu() {
    while true; do
        for gpu_id in "${GPU_IDS[@]}"; do
            if ! is_running "${gpu_pid[$gpu_id]}"; then
                echo "$gpu_id"
                return 0
            fi
        done
        sleep 2
    done
}

# Launch tasks
pids=()
completed=0

for task in "${tasks[@]}"; do
    gpu_id=$(get_free_gpu)
    log_file="${LOG_DIR}/${task}.log"

    echo -e "\033[36m→ [${EXP_NAME}] Task: $task | GPU: $gpu_id\033[0m"

    (
        export CUDA_VISIBLE_DEVICES=$gpu_id

        PYTHONWARNINGS=ignore::UserWarning \
        python script/eval_policy.py \
            --config "policy/${POLICY_NAME}/deploy_policy.yml" \
            --overrides \
            --task_name "${task}" \
            --task_config "${TASK_CONFIG}" \
            --ckpt_setting "${CHECKPOINT_PATH}" \
            --exp_name "${EXP_NAME}" \
            --seed "${SEED}" \
            --policy_name "${POLICY_NAME}" \
            --log_dir "${LOG_DIR}" \
            --wan_path "${WAN_PATH}" \
            --vlm_path "${VLM_PATH}" \
            --test_num "${TEST_NUM}" \
            > "$log_file" 2>&1

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "DONE Task $task completed successfully" >> "$log_file"
        else
            echo "FAIL Task $task failed with exit code $exit_code" >> "$log_file"
        fi
    ) &

    pid=$!
    gpu_pid[$gpu_id]=$pid
    pids+=($pid)
    sleep 1
done

echo -e "\n\033[33mWaiting for all tasks to complete...\033[0m"
for pid in "${pids[@]}"; do
    wait "$pid"
    ((completed++))
    echo -e "  Completed: $completed / $total"
done

echo -e "\n\033[32m=== All tasks completed ===\033[0m"

# Combine videos (eval results are saved under eval_result/ in robotwin root)
echo -e "\n\033[33mCombining real + predicted videos...\033[0m"
python "${SCRIPT_DIR}/combine_videos.py" --results_dir "${ROBOTWIN_ROOT}/eval_result" 2>&1 | tail -10

# Generate summary
summary="${LOG_DIR}/summary.txt"
cat > "$summary" << EOF
Motus Baseline Experiment Summary
==================================
Date: $(date)
Experiment: ${EXP_NAME}
Checkpoint: ${CHECKPOINT_PATH}
Test Episodes: ${TEST_NUM}
GPUs: ${GPU_IDS[*]}
Seed: ${SEED}

Task Results:
EOF

success=0
failed=0
for task in "${tasks[@]}"; do
    log_file="${LOG_DIR}/${task}.log"
    if [ ! -f "$log_file" ]; then
        echo "  [MISSING] $task" >> "$summary"
        ((failed++))
    elif grep -q "DONE.*completed successfully" "$log_file" 2>/dev/null; then
        # Extract success rate from log
        rate=$(grep -oP 'Success rate:.*?(\d+\.\d+%)' "$log_file" | tail -1 | grep -oP '\d+\.\d+%')
        echo "  [OK] $task  (${rate:-N/A})" >> "$summary"
        ((success++))
    else
        echo "  [FAIL] $task" >> "$summary"
        ((failed++))
    fi
done

cat >> "$summary" << EOF

Summary: ${success} OK / ${failed} FAIL / ${total} Total
Logs: ${LOG_DIR}
EOF

echo ""
cat "$summary"
echo ""
echo "Experiment finished at $(date)"
