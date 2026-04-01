#!/bin/bash
# Baseline: MotusV2 checkpoint (step_110000, video-condition, NO scale training)
# scale perturbation × high severity, scenario A
# Task: adjust_bottle only, 2 episodes, GPU 1

echo "================================================================"
echo "  Baseline: MotusV2 step_110000 (no scale training)"
echo "  Perturbation: scale × high | Scenario: A"
echo "  Task: adjust_bottle | Episodes: 2"
echo "  Started at $(date)"
echo "================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="/gemini/code/robotwin"
CHECKPOINT_PATH="/gemini/code/Motus/checkpoints/robotwin_full_v2/robotwin_full_v2/checkpoint_step_110000"
WAN_PATH="/gemini/code/models/motus_pretrained_models/Wan2.2-TI2V-5B"
VLM_PATH="/gemini/code/models/motus_pretrained_models/Qwen3-VL-2B-Instruct"
CONDA_ENV="/gemini/code/envs/robotwin"
POLICY_NAME="MotusV2"
TASK_CONFIG="demo_randomized"
SEED="42"
TEST_NUM=2

PERTURB_SCENARIO="A"
PERTURB_T_ON_LO=1
PERTURB_T_ON_HI=3
PERTURB_TYPES=("scale")
SEVERITIES=("high")

GPU_IDS=(1)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TOP_DIR="${SCRIPT_DIR}/experiments/perturbed_scale_baseline_v2_${TIMESTAMP}"
mkdir -p "$TOP_DIR"
echo "Log directory: $TOP_DIR"

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

tasks=("adjust_bottle")
total_tasks=${#tasks[@]}
echo "Tasks: ${tasks[*]} | GPUs: ${GPU_IDS[*]}"
echo ""

declare -A gpu_pid
for gpu_id in "${GPU_IDS[@]}"; do gpu_pid[$gpu_id]=""; done

is_running() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }

get_free_gpu() {
    while true; do
        for gpu_id in "${GPU_IDS[@]}"; do
            if ! is_running "${gpu_pid[$gpu_id]}"; then echo "$gpu_id"; return 0; fi
        done
        sleep 5
    done
}

for sev in "${SEVERITIES[@]}"; do
    for ptype in "${PERTURB_TYPES[@]}"; do
        EXP_NAME="perturbed_${ptype}_${sev}"
        COND_DIR="${TOP_DIR}/${EXP_NAME}"
        mkdir -p "$COND_DIR"

        echo "  Condition: ${EXP_NAME} | Type: ${ptype} | Severity: ${sev}"

        pids=()
        for task in "${tasks[@]}"; do
            gpu_id=$(get_free_gpu)
            log_file="${COND_DIR}/${task}.log"
            echo "  → Task: $task | GPU: $gpu_id"

            (
                export CUDA_VISIBLE_DEVICES=$gpu_id
                PYTHONWARNINGS=ignore::UserWarning \
                python script/eval_policy_perturbed.py \
                    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
                    --overrides \
                    --task_name "${task}" \
                    --task_config "${TASK_CONFIG}" \
                    --ckpt_setting "${CHECKPOINT_PATH}" \
                    --exp_name "${EXP_NAME}" \
                    --seed "${SEED}" \
                    --policy_name "${POLICY_NAME}" \
                    --log_dir "${COND_DIR}" \
                    --wan_path "${WAN_PATH}" \
                    --vlm_path "${VLM_PATH}" \
                    --test_num "${TEST_NUM}" \
                    --perturb_severity "${sev}" \
                    --perturb_scenario "${PERTURB_SCENARIO}" \
                    --perturb_t_on_chunk_lo "${PERTURB_T_ON_LO}" \
                    --perturb_t_on_chunk_hi "${PERTURB_T_ON_HI}" \
                    --perturb_types "${ptype}" \
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

        for pid in "${pids[@]}"; do wait "$pid"; done

        # Print result
        log_file="${COND_DIR}/adjust_bottle.log"
        if grep -q "DONE.*completed successfully" "$log_file" 2>/dev/null; then
            rate=$(grep -oP 'Success rate:.*?\K\d+/\d+' "$log_file" | tail -1)
            echo "  Result: ${rate:-see log}"
        else
            echo "  Result: FAILED (check ${log_file})"
        fi
    done
done

echo ""
echo "================================================================"
echo "  Baseline (V2 step_110000) scale+high done: $(date)"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Logs: ${TOP_DIR}"
echo "================================================================"
