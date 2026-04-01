#!/bin/bash
# Overfit10: MotusV3 step_500, adjust_bottle, scale+high+scenarioA, 10 episodes, GPU 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="/gemini/code/robotwin"
CHECKPOINT_PATH="/gemini/code/Motus/checkpoints_v3/robotwin_full_v3_overfit10/robotwin_full_v3_overfit10/checkpoint_step_500"
WAN_PATH="/gemini/code/models/motus_pretrained_models/Wan2.2-TI2V-5B"
VLM_PATH="/gemini/code/models/motus_pretrained_models/Qwen3-VL-2B-Instruct"
CONDA_ENV="/gemini/code/envs/robotwin_motus"
POLICY_NAME="MotusV2"
TASK_CONFIG="demo_randomized"
SEED="42"
TEST_NUM=10

PERTURB_SCENARIO="A"
PERTURB_T_ON_LO=1
PERTURB_T_ON_HI=3

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/experiments/scale_overfit10_step500_10ep_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  Overfit10: V3 step_500 | adjust_bottle | scale+high | 10ep"
echo "  GPU: 5 | Log: $LOG_DIR"
echo "  Started at $(date)"
echo "================================================================"

cd "$ROBOTWIN_ROOT" || exit 1
export PATH="/gemini/code/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=5

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_perturbed.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "adjust_bottle" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${CHECKPOINT_PATH}" \
    --exp_name "scale_overfit10_step500" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    --test_num "${TEST_NUM}" \
    --perturb_severity "high" \
    --perturb_scenario "${PERTURB_SCENARIO}" \
    --perturb_t_on_chunk_lo "${PERTURB_T_ON_LO}" \
    --perturb_t_on_chunk_hi "${PERTURB_T_ON_HI}" \
    --perturb_types "scale" \
    2>&1 | tee "${LOG_DIR}/adjust_bottle.log"

echo "================================================================"
echo "  Done at $(date)"
echo "================================================================"
