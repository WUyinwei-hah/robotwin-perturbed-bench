#!/bin/bash
# Experiment: Motus V1 Perturbation Robustness Evaluation
# - Bias perturbation only × 2 severity levels = 2 conditions
# - Scenario A (always-on from start)
# - 2 episodes per task, 50 tasks
# - GPU: 4

echo "================================================================"
echo "  Experiment: Motus V1 PERTURBATION ROBUSTNESS"
echo "  Conditions: bias × 2 severities = 2"
echo "  Episodes per task: 2 | Tasks: 50"
echo "  Started at $(date)"
echo "================================================================"

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

# Perturbation: scenario A (always-on from start)
PERTURB_SCENARIO="A"
PERTURB_T_ON_LO=1
PERTURB_T_ON_HI=3

# GPU pool
GPU_IDS=(4)

# Top-level experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TOP_DIR="${SCRIPT_DIR}/experiments/perturbed_bias_sweep_${TIMESTAMP}"
mkdir -p "$TOP_DIR"
echo "Top-level log directory: $TOP_DIR"

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
total_tasks=${#tasks[@]}
echo "Total tasks: $total_tasks"
echo "GPUs: ${GPU_IDS[*]}"
echo ""

# ============================================================
# Perturbation conditions: bias × 2 severities = 2
# ============================================================
PERTURB_TYPES=("bias")
SEVERITIES=("lm" "high")

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
        sleep 5
    done
}

# ============================================================
# Run all conditions
# ============================================================
total_conditions=$(( ${#PERTURB_TYPES[@]} * ${#SEVERITIES[@]} ))
condition_idx=0

for ptype in "${PERTURB_TYPES[@]}"; do
    for sev in "${SEVERITIES[@]}"; do
        ((condition_idx++))
        EXP_NAME="perturbed_${ptype}_${sev}"
        COND_DIR="${TOP_DIR}/${EXP_NAME}"
        mkdir -p "$COND_DIR"

        echo ""
        echo "================================================================"
        echo "  Condition ${condition_idx}/${total_conditions}: ${EXP_NAME}"
        echo "  Type: ${ptype} | Severity: ${sev} | Scenario: ${PERTURB_SCENARIO}"
        echo "================================================================"

        # Launch all tasks for this condition
        pids=()
        completed=0

        for task in "${tasks[@]}"; do
            gpu_id=$(get_free_gpu)
            log_file="${COND_DIR}/${task}.log"

            echo -e "\033[36m  → [${EXP_NAME}] Task: $task | GPU: $gpu_id\033[0m"

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

        # Wait for all tasks in this condition
        echo -e "\n\033[33m  Waiting for condition ${EXP_NAME}...\033[0m"
        for pid in "${pids[@]}"; do
            wait "$pid"
            ((completed++))
        done
        echo -e "\033[32m  Condition ${EXP_NAME} done (${completed}/${total_tasks} tasks)\033[0m"

        # Per-condition summary
        cond_summary="${COND_DIR}/summary.txt"
        cat > "$cond_summary" << EOF
Condition: ${EXP_NAME}
Type: ${ptype} | Severity: ${sev} | Scenario: ${PERTURB_SCENARIO}
t_on_chunk_range: [${PERTURB_T_ON_LO}, ${PERTURB_T_ON_HI}]
Test Episodes: ${TEST_NUM}

Task Results:
EOF
        cond_ok=0
        cond_fail=0
        cond_total_suc=0
        for task in "${tasks[@]}"; do
            log_file="${COND_DIR}/${task}.log"
            if [ ! -f "$log_file" ]; then
                echo "  [MISSING] $task" >> "$cond_summary"
                ((cond_fail++))
            elif grep -q "DONE.*completed successfully" "$log_file" 2>/dev/null; then
                rate=$(grep -oP 'Success rate:.*?(\d+\.\d+%)' "$log_file" | tail -1 | grep -oP '\d+\.\d+%')
                suc_count=$(grep -oP 'Success rate:.*?\K\d+(?=/\d+)' "$log_file" | tail -1)
                echo "  [OK] $task  (${rate:-N/A})" >> "$cond_summary"
                ((cond_ok++))
                if [ -n "$suc_count" ]; then
                    ((cond_total_suc += suc_count))
                fi
            else
                echo "  [FAIL] $task" >> "$cond_summary"
                ((cond_fail++))
            fi
        done
        cond_total_ep=$((cond_ok * TEST_NUM))
        if [ $cond_total_ep -gt 0 ]; then
            cond_rate=$(echo "scale=1; $cond_total_suc * 100 / $cond_total_ep" | bc)
        else
            cond_rate="N/A"
        fi
        cat >> "$cond_summary" << EOF

Summary: ${cond_ok} OK / ${cond_fail} FAIL / ${total_tasks} Total
Aggregate success rate: ${cond_total_suc}/${cond_total_ep} = ${cond_rate}%
EOF
        echo "  Condition summary saved: $cond_summary"
    done
done

# ============================================================
# Overall summary
# ============================================================
overall="${TOP_DIR}/overall_summary.txt"
cat > "$overall" << EOF
Motus V1 Perturbation Robustness — Overall Summary
====================================================
Date: $(date)
Checkpoint: ${CHECKPOINT_PATH}
Test Episodes per task: ${TEST_NUM}
Tasks: ${total_tasks}
GPUs: ${GPU_IDS[*]}
Seed: ${SEED}
Scenario: ${PERTURB_SCENARIO}
t_on_chunk_range: [${PERTURB_T_ON_LO}, ${PERTURB_T_ON_HI}]

Condition Results:
EOF

printf "%-25s %10s %10s %10s\n" "Condition" "OK/Total" "SuccEps" "Rate" >> "$overall"
printf "%-25s %10s %10s %10s\n" "-------------------------" "----------" "----------" "----------" >> "$overall"

for ptype in "${PERTURB_TYPES[@]}"; do
    for sev in "${SEVERITIES[@]}"; do
        EXP_NAME="perturbed_${ptype}_${sev}"
        COND_DIR="${TOP_DIR}/${EXP_NAME}"

        cond_ok=0
        cond_total_suc=0
        for task in "${tasks[@]}"; do
            log_file="${COND_DIR}/${task}.log"
            if [ -f "$log_file" ] && grep -q "DONE.*completed successfully" "$log_file" 2>/dev/null; then
                ((cond_ok++))
                suc_count=$(grep -oP 'Success rate:.*?\K\d+(?=/\d+)' "$log_file" | tail -1)
                if [ -n "$suc_count" ]; then
                    ((cond_total_suc += suc_count))
                fi
            fi
        done
        cond_total_ep=$((cond_ok * TEST_NUM))
        if [ $cond_total_ep -gt 0 ]; then
            cond_rate=$(echo "scale=1; $cond_total_suc * 100 / $cond_total_ep" | bc)
        else
            cond_rate="N/A"
        fi
        printf "%-25s %10s %10s %10s\n" "${EXP_NAME}" "${cond_ok}/${total_tasks}" "${cond_total_suc}/${cond_total_ep}" "${cond_rate}%" >> "$overall"
    done
done

echo "" >> "$overall"
echo "Logs: ${TOP_DIR}" >> "$overall"

echo ""
echo "================================================================"
cat "$overall"
echo "================================================================"
echo ""
echo "Experiment finished at $(date)"
