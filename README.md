# Perturbed RoboTwin Benchmark

A reproducible benchmark for evaluating robotic manipulation policies under fixed perturbation conditions on the RoboTwin platform.

## Overview

- **50 tasks** from the RoboTwin task suite
- **5 perturbation types**: Scale, Coupling, Low-pass IIR, Low-pass FIR, Bias (Drop excluded)
- **2 severity levels**: LM (low-medium), HIGH
- **2 timing modes**: `always_on` (from step 0), `onset_then_always` (from fixed early onset)
- **50 repeats** per (setting × task) combination
- **20 settings** × 50 tasks × 50 repeats = **50,000 episodes per policy**
- **3 policies**: Motus, Pi0.5, LingbotVA

All perturbation parameters, seeds, and timing are pre-generated in `benchmark_spec.json` for full reproducibility.
Each task uses one verified environment seed (reused across all settings and repeats), while perturbation parameters are resampled per repeat.

## Directory Structure

```
robotwin-perturbed-bench/
├── benchmark/
│   ├── perturbation_engine.py   # Forked perturbation engine (no Drop)
│   ├── spec_generator.py        # Generates benchmark_spec.json
│   ├── perturbed_env.py         # Wraps TASK_ENV with qpos-level perturbation
│   ├── eval_runner.py           # Core evaluation loop
│   ├── aggregate_results.py     # Result aggregation & CSV export
│   └── benchmark_spec.json      # Generated spec (after running generate_spec.sh)
├── policies/                    # *** SELF-CONTAINED policy code ***
│   ├── base_adapter.py          # Abstract policy interface
│   ├── motus_adapter.py         # Motus adapter
│   ├── pi05_adapter.py          # Pi0.5 adapter
│   ├── lingbot_va_adapter.py    # LingbotVA adapter (websocket client)
│   ├── motus_policy/            # Full Motus inference code (models/, utils/, bak/)
│   ├── pi05_policy/             # Full Pi0.5 inference code (src/, packages/)
│   └── lingbot_va/              # LingbotVA websocket client + server launch script
├── configs/
│   ├── motus.yml                # Motus model paths
│   ├── pi05_robotwin2.yml       # Pi0.5 (HF pi0.5_robotwin2) model paths
│   └── lingbot_va.yml           # LingbotVA server config
├── scripts/
│   ├── generate_spec.sh         # Step 1: generate benchmark spec
│   ├── run_motus.sh             # Run Motus evaluation
│   ├── run_pi05.sh              # Run Pi0.5 evaluation
│   ├── run_lingbot_va.sh        # Run LingbotVA evaluation
│   └── aggregate.sh             # Aggregate results
├── tasks_all.txt                # 50 benchmark tasks
├── results/                     # Episode results (JSON per episode)
├── summary/                     # Aggregated CSV/JSON summaries
├── envs -> ../robotwin/envs     # Symlink to RoboTwin envs
├── task_config -> ../robotwin/task_config
├── assets -> ../robotwin/assets
├── description -> ../robotwin/description
└── script -> ../robotwin/script
```

## Self-Contained Design

All policy inference code is bundled inside this repository:

- **Motus** → `policies/motus_policy/` (models, WAN backbone, Qwen VLM, utils)
- **Pi0.5** → `policies/pi05_policy/` (full Pi0.5 inference stack)
- **LingbotVA** → `policies/lingbot_va/` (websocket client + msgpack serialization)

You do **not** need to clone the original `Motus` or `lingbot-va` repositories to run the benchmark. The only external dependency is the **RoboTwin simulation environment** (`robotwin` repo), which provides task definitions, SAPIEN scenes, and robot assets.

> **Exception**: LingbotVA uses a client-server architecture. The *client* is self-contained, but the *server* (model inference) still requires the `lingbot-va` repo to be installed separately. A launch script template is provided at `policies/lingbot_va/server/launch_server.sh`.

## Run on Another Machine (Full Setup Guide)

This section is the recommended migration path after cloning from GitHub.

### A. Recommended workspace layout

Use the same parent folder for all related repos:

```text
<WORK_ROOT>/
  robotwin/                    # RoboTwin simulator (required for all policies)
  robotwin-perturbed-bench/    # This repo (self-contained policy code)
  lingbot-va/                  # Only needed for LingbotVA SERVER
  models/                      # Model weights (downloaded separately)
```

Example below uses `<WORK_ROOT>=/data/code`.

### B. Clone required repositories

```bash
mkdir -p /data/code
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /data/code/robotwin
git clone https://github.com/WUyinwei-hah/robotwin-perturbed-bench.git /data/code/robotwin-perturbed-bench

# Optional: only if you need LingbotVA evaluation
git clone <YOUR_LINGBOT_VA_REPO_URL> /data/code/lingbot-va
```

### C. Python environment and base dependencies

Follow RoboTwin official install document first:
- https://robotwin-platform.github.io/doc/usage/robotwin-install.html

Then ensure your Python env can import RoboTwin env + sapien.

If you maintain a dedicated env path, export it before running scripts:

```bash
export PYTHON=/path/to/your/python
```

> In this benchmark repo, shell scripts default to `/gemini/code/envs/robotwin/bin/python`.
> On a new machine, you should override via `PYTHON=... bash scripts/xxx.sh`.

### D. Recreate local symlink dependencies (critical)

This repository contains symlinks that target local RoboTwin paths. After cloning,
you must recreate or fix them.

Run in benchmark root:

```bash
cd /data/code/robotwin-perturbed-bench
rm -rf envs task_config assets description script
ln -s ../robotwin/envs envs
ln -s ../robotwin/task_config task_config
ln -s ../robotwin/assets assets
ln -s ../robotwin/description description
ln -s ../robotwin/script script
```

Verify:

```bash
ls -l envs task_config assets description script
```

### E. Models and checkpoints

This benchmark does **evaluation only** (no retraining), but each policy still needs
its runtime checkpoints/models.

1. **Motus** — update `configs/motus.yml`:
   - `robotwin_root`: path to your RoboTwin repo
   - `checkpoint_path`: Motus model checkpoint
   - `wan_path`: Wan2.2-TI2V-5B weights
   - `vlm_path`: Qwen3-VL-2B-Instruct weights
   - *(No `policy_dir` needed — code is self-contained in `policies/motus_policy/`)*
2. **Pi0.5** — use `configs/pi05_robotwin2.yml` (default):
   - Pre-trained checkpoint: [motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2)
   - Download and setup (see [Pi0.5 Setup](#pi05-setup) below)
   - `checkpoint_dir`: absolute path to the downloaded checkpoint
   - `robotwin_root`: path to your RoboTwin repo
   - *(No `policy_dir` needed — code is self-contained in `policies/pi05_policy/`)*
3. **LingbotVA** — update `configs/lingbot_va.yml`:
   - `robotwin_root`: path to your RoboTwin repo
   - `host`, `port`: websocket server address
   - Start LingbotVA server before running client-side eval
   - *(No `lingbot_va_root` needed — client is self-contained in `policies/lingbot_va/`)*

> If you need exact model download links, use each policy repo/release page corresponding
> to your internal setup. This benchmark intentionally reads paths from `configs/*.yml`.

### F. Dataset (usually NOT needed)

**For benchmark evaluation, no dataset download is required.**
The `benchmark_spec.json` is pre-generated and committed to this repo with all seeds and perturbation parameters.

Dataset is only needed if you want to **regenerate** the spec from scratch.
In that case, `spec_generator.py` reads seed candidates from a clean expert-demo dataset with this structure:

```text
<dataset_root>/
  <task_name>/
    qpos/
      0.pt
      1.pt
      ...
```

The official RoboTwin dataset on HuggingFace has a **different structure** — per-embodiment zip files:

```text
https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset
  <task_name>/
    aloha-agilex_clean_50.zip   # ← zipped, organized by embodiment
    aloha-agilex_randomized_500.zip
    ur5_clean_50.zip
    ...
```

To use HuggingFace data for spec regeneration, you need to:
1. Download the `aloha-agilex_clean_50.zip` for each task
2. Unzip into `<dataset_root>/<task_name>/` so that `qpos/*.pt` files are directly accessible
3. Pass `--dataset-root <dataset_root>` to `spec_generator.py`

> **In practice, you should just use the pre-generated `benchmark_spec.json`** — regeneration requires SAPIEN + CuroboPlanner for seed verification.

### G. CuroboPlanner / IK solver compatibility (important)

The eval runner includes a **Phase 1 expert check** that verifies each seed is solvable
before running the policy. This requires CuroboPlanner (GPU-based IK solver).

If CuroboPlanner crashes on your machine (e.g., `CUDA driver version is insufficient
for CUDA runtime version`), you have two options:

**Option 1: Fix CuroboPlanner** (recommended)

```bash
# Reinstall curobo compiled against your PyTorch's CUDA version
pip uninstall curobo
export CUDA_HOME=/usr/local/cuda-12.1  # match your PyTorch CUDA
pip install git+https://github.com/NVlabs/curobo.git --no-build-isolation
```

**Option 2: Skip expert check** (seeds are already pre-verified in the spec)

```bash
SKIP_EXPERT_CHECK=1 bash scripts/run_motus.sh 0
```

This is safe because all seeds in `benchmark_spec.json` were verified during spec
generation. The expert check is a redundant safety net.

### H. Machine validation checklist (run before long experiments)

```bash
cd /data/code/robotwin-perturbed-bench

# 1) renderer/sapien health
${PYTHON:-python} -c "from script.test_render import Sapien_TEST; Sapien_TEST()"

# 2) ffmpeg availability
ffmpeg -version

# 3) quick smoke eval (small subset)
PYTHON=${PYTHON:-python} bash scripts/run_motus.sh 0 "scale_lm_always_on" "adjust_bottle"

# If expert check fails (IK errors), try with skip:
SKIP_EXPERT_CHECK=1 PYTHON=${PYTHON:-python} bash scripts/run_motus.sh 0 "scale_lm_always_on" "adjust_bottle"
```

### I. Cross-machine path adaptation summary

On new machines, update these files first:

- `scripts/*.sh`: optionally pass `PYTHON=/path/to/python`
- `configs/motus.yml`: `robotwin_root` + model checkpoint paths
- `configs/pi05_robotwin2.yml`: `checkpoint_dir` + `robotwin_root`
- `configs/lingbot_va.yml`: `robotwin_root` + server host/port

No `policy_dir` or `lingbot_va_root` paths needed — all policy code is self-contained.
Do this once, then the same run commands are portable.

## Quick Start

### 0. Prerequisites (must pass before running)

1. RoboTwin base code exists at `/gemini/code/robotwin`
2. Python env exists at `/gemini/code/envs/robotwin/bin/python`
3. Symlinks are valid:

```bash
ls -l envs task_config assets description script
```

4. `ffmpeg` is available:

```bash
ffmpeg -version
```

5. (For LingbotVA) policy server is running before evaluation.

### 1. Benchmark Spec (pre-generated)

`benchmark/benchmark_spec.json` is already committed with deterministic perturbation configs
for all 50,000 episodes. **You do not need to regenerate it.**

To regenerate from scratch (requires SAPIEN + CuroboPlanner + clean dataset):

```bash
bash scripts/generate_spec.sh
```

### 2. Update Policy Configs (critical)

Edit these files to your real paths:

- `configs/motus.yml` — `robotwin_root`, `checkpoint_path`, `wan_path`, `vlm_path`
- `configs/pi05_robotwin2.yml` — `checkpoint_dir`, `robotwin_root` (see [Pi0.5 Setup](#pi05-setup))
- `configs/lingbot_va.yml` — `robotwin_root`, `host`, `port`

No external `policy_dir` or `lingbot_va_root` needed. If model paths are wrong, loading will fail before the first episode.

#### <a name="pi05-setup"></a>Pi0.5 Setup (motus-robotics/pi0.5_robotwin2)

**Step 1: Download the checkpoint from HuggingFace**

```bash
# Install huggingface-cli if not available
pip install huggingface_hub

# Download model weights + norm stats (skip optimizer.pt — not needed for inference)
huggingface-cli download motus-robotics/pi0.5_robotwin2 \
    model.safetensors \
    assets/pi0.5_clean_randomize_joint_training/norm_stats.json \
    --local-dir /gemini/code/models/pi05_robotwin2
```

**Step 2: Verify directory structure**

After download, you should have:

```text
/gemini/code/models/pi05_robotwin2/
  model.safetensors                                          # 7.47 GB (PyTorch weights)
  assets/
    pi0.5_clean_randomize_joint_training/
      norm_stats.json                                        # Normalization statistics
```

**Step 3: Update config** (if using a different path)

Edit `configs/pi05_robotwin2.yml`:

```yaml
checkpoint_dir: /path/to/your/pi05_robotwin2   # must contain model.safetensors + assets/
robotwin_root: /path/to/your/robotwin           # RoboTwin simulator repo
pi0_step: 32                                    # action horizon (model generates 32 actions/step)
```

**Step 4: Run**

```bash
# Full benchmark
bash scripts/run_pi05.sh 0

# With specific settings/tasks
bash scripts/run_pi05.sh 0 "scale_lm_always_on" "adjust_bottle"

# Skip expert check (if CuroboPlanner unavailable)
SKIP_EXPERT_CHECK=1 bash scripts/run_pi05.sh 0

# Use a different config file
POLICY_CONFIG=configs/pi05_robotwin2.yml bash scripts/run_pi05.sh 0
```

**Model details:**
- Base: pi0.5 official base model (JAX → PyTorch converted)
- Fine-tuned on: 50 RoboTwin 2.0 tasks (clean + randomized, joint training)
- Action space: 14D qpos (delta joint actions, `adapt_to_pi=True`)
- Action horizon: 32 steps per inference call
- Config name: `pi05_robotwin2` (registered in `policies/pi05_policy/src/openpi/training/config.py`)

### 3. Run Evaluation

```bash
# Full benchmark (all settings, all tasks)
bash scripts/run_motus.sh 0

# Specific settings/tasks
bash scripts/run_motus.sh 0 "scale_lm_always_on,bias_high_always_on"
bash scripts/run_motus.sh 0 "" "adjust_bottle,click_bell"

# Pi0.5
bash scripts/run_pi05.sh 1

# LingbotVA (server must be running separately)
bash scripts/run_lingbot_va.sh 2

# Skip expert check (when CuroboPlanner is unavailable)
SKIP_EXPERT_CHECK=1 bash scripts/run_motus.sh 0
```

#### Practical run strategy (recommended)

Run by setting/task subsets first, then scale up:

```bash
# 1 setting + 2 tasks smoke run
bash scripts/run_motus.sh 0 "scale_lm_always_on" "adjust_bottle,click_bell"

# all settings + 1 task (stress perturbation coverage)
bash scripts/run_motus.sh 0 "" "adjust_bottle"
```

Evaluation supports **resume**: existing `episode_*.json` files are skipped automatically.

### 3.1 Expected output structure during eval

Each episode writes one JSON:

```text
results/
  Motus/
    scale_lm_always_on/
      adjust_bottle/
        episode_0.json
        episode_1.json
        ...
        videos/
```

Episode JSON includes:
- `success`
- `steps`
- `error` (null if normal)
- `perturbation_log`
- `setting_id` / `repeat_idx` / `policy`

### 4. Aggregate Results

```bash
bash scripts/aggregate.sh
```

This produces:
- `summary/overall.csv` — per-policy success rates
- `summary/by_setting.csv` — per-setting breakdown
- `summary/by_task.csv` — per-task breakdown
- `summary/by_perturb_type.csv` — by perturbation type
- `summary/by_severity.csv` — by severity
- `summary/by_timing.csv` — by timing mode
- `summary/aggregate.json` — full JSON summary

## Perturbation Injection

All perturbations are injected at the **14D qpos execution layer**:

- **qpos policies** (Motus, Pi0.5): perturbation applied directly to the 14D action before `take_action(action, action_type='qpos')`
- **EE policies** (LingbotVA): IK/path planning runs normally, then joint targets are offset by the perturbation at each waypoint

This ensures all policies experience equivalent actuator-level disturbances regardless of their action space.

## Timing Modes

| Mode | Description | `t_on_raw` |
|---|---|---|
| `always_on` | Perturbation active from step 0 | 0 |
| `onset_then_always` | Perturbation starts at early onset, stays until end | 16% × step_limit, clipped to [32, 192] |

## Reproducibility contract

Reproducibility is guaranteed by:

1. fixed benchmark generation seed (`master_seed`)
2. fixed task list (`tasks_all.txt`)
3. fixed repeats (`50`)
4. fixed perturbation config per (setting, task, repeat)
5. fixed environment seed per task (same seed reused for all settings/repeats)

Do **not** regenerate `benchmark_spec.json` with different arguments if you want comparable results.

## Troubleshooting

### 1) `not a valid checkpoint/path not found`
Check `configs/*.yml` model paths.

### 2) `Render Error` / SAPIEN init fail
Validate display/driver environment and run:

```bash
${PYTHON:-python} -c "from script.test_render import Sapien_TEST; Sapien_TEST()"
```

### 3) `expert_failed` / `IK Failed! Cannot find valid solution`
The expert demo planner (CuroboPlanner) cannot solve the task on this machine.
This is typically caused by CUDA version incompatibility with `curobo`.

**Quick fix**: skip expert check (seeds are pre-verified):
```bash
SKIP_EXPERT_CHECK=1 bash scripts/run_motus.sh 0
```

**Permanent fix**: reinstall curobo matching your PyTorch CUDA version:
```bash
pip uninstall curobo
export CUDA_HOME=/usr/local/cuda-12.1  # match: python -c "import torch; print(torch.version.cuda)"
pip install git+https://github.com/NVlabs/curobo.git --no-build-isolation
```

### 4) LingbotVA blocks or times out
- confirm server is started
- confirm `port` in `configs/lingbot_va.yml` matches server

### 5) Interrupted runs
Re-run the same command. Existing episodes are skipped (resume behavior).

## Upload to GitHub

This directory is currently not a git repository by default. Use:

```bash
# in /gemini/code/robotwin-perturbed-bench
git init
git add .
git commit -m "Initial perturbed Robotwin benchmark"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

If using HTTPS + PAT:
- create PAT with `repo` scope
- use `https://<TOKEN>@github.com/<user>/<repo>.git` or credential manager

If using SSH:
- ensure `ssh -T git@github.com` succeeds first.

## Benchmark Spec Variants

| File | Settings | Tasks | Repeats | Total/policy | Description |
|------|----------|-------|---------|-------------|-------------|
| `benchmark/benchmark_spec.json` | 20 (all) | 50 | 50 | 50,000 | Full benchmark (all severities) |
| `benchmark/benchmark_spec_lm20.json` | 10 (lm only) | 50 | 20 | 10,000 | Compact benchmark for quick multi-machine eval |
| `benchmark/benchmark_spec_verified.json` | 10 (lm) | 50 | 20 | 10,000 | Verified perturbations (generated via `generate_verified_spec.py`) |

## Settings (20 total)

5 types × 2 severities × 2 timings = 20 settings:

| Type | Severity | Timing |
|---|---|---|
| scale | lm, high | always_on, onset_then_always |
| coupling | lm, high | always_on, onset_then_always |
| iir | lm, high | always_on, onset_then_always |
| fir | lm, high | always_on, onset_then_always |
| bias | lm, high | always_on, onset_then_always |

## Multi-Machine Experiment Workflow

For running experiments across multiple machines and syncing via GitHub, see
**[docs/multi_machine_sync.md](docs/multi_machine_sync.md)**.

Key points:
- Each machine runs a partition of experiments using `--worker-id` / `--num-workers`
- Results are synced via `git push/pull` (episode JSONs have unique paths, no conflicts)
- Use `report_generation.py` to generate unified reports from merged results

## Report Generation

Unified progress and success-rate reporting across any number of policies:

```bash
# Auto-detect all policies, print to terminal
python report_generation.py --results-dir results

# Export portable Markdown + JSON
python report_generation.py --results-dir results \
    --export-md report.md --export-json report.json

# Use lm-only spec for expected counts
python report_generation.py --results-dir results \
    --spec benchmark/benchmark_spec_lm20.json
```

Full documentation: **[docs/report_generation.md](docs/report_generation.md)**

## Environment

- Python: `/gemini/code/envs/robotwin/bin/python`
- CWD: `/gemini/code/robotwin-perturbed-bench/`
- Task config: `demo_clean` (no domain randomization)
