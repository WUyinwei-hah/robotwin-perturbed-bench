# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproducible benchmark for evaluating robotic manipulation policies under fixed perturbation conditions on the RoboTwin simulation platform.

- **3 policies**: Motus (qpos), Pi0.5 (qpos), LingbotVA (end-effector, websocket client-server)
- **50 RoboTwin tasks**, **5 perturbation types** (Scale, Coupling, IIR, FIR, Bias), **2 severities** (LM, HIGH), **2 timing modes** (always_on, onset_then_always)
- **50 repeats** per (setting × task) = 50,000 episodes per policy
- All perturbation parameters are pre-generated in `benchmark/benchmark_spec.json` for reproducibility

## Common Commands

### Running evaluations
```bash
# Run a specific policy (GPU_ID, optional SETTINGS filter, optional TASKS filter)
bash scripts/run_motus.sh 0
bash scripts/run_pi05.sh 1
bash scripts/run_lingbot_va.sh 2

# With filters
bash scripts/run_motus.sh 0 "scale_lm_always_on" "adjust_bottle,click_bell"

# Skip expert check (recommended — seeds are pre-verified in spec)
SKIP_EXPERT_CHECK=1 bash scripts/run_motus.sh 0

# Override Python interpreter
PYTHON=/path/to/python bash scripts/run_motus.sh 0

# Override Pi0.5 config
POLICY_CONFIG=configs/pi05_robotwin2.yml bash scripts/run_pi05.sh 0
```

### Direct eval_runner invocation
```bash
python -m benchmark.eval_runner \
    --policy motus \
    --policy-config configs/motus.yml \
    --spec benchmark/benchmark_spec.json \
    --output results \
    --task-config demo_clean \
    --gpu 0 \
    --skip-expert-check \
    --settings scale_lm_always_on \
    --tasks adjust_bottle,click_bell
```

### Aggregating results
```bash
bash scripts/aggregate.sh                    # results/ → summary/
python -m benchmark.aggregate_results --results-dir results --output summary
```

### Report generation
```bash
python report_generation.py --results-dir results
python report_generation.py --results-dir results --export-md report.md --export-json report.json
python report_generation.py --results-dir results --spec benchmark/benchmark_spec_lm20.json
```

### Monitoring
```bash
python scripts/watch_benchmark.py            # auto-refresh every 30s
python scripts/watch_benchmark.py --once     # single snapshot
```

### Spec generation (rarely needed — spec is pre-generated)
```bash
bash scripts/generate_spec.sh
```

## Architecture

### Three-layer design

1. **Benchmark framework** (`benchmark/`): Evaluation loop, perturbation engine, spec generation
2. **Policy adapters** (`policies/`): Unified `PolicyAdapter` interface wrapping each policy
3. **Configuration** (`configs/`): YAML files mapping model paths per policy

### Key files

| File | Purpose |
|------|---------|
| `benchmark/eval_runner.py` | Core two-phase evaluation loop (expert check → policy execution) |
| `benchmark/perturbation_engine.py` | 5 perturbation types with state management, timing modes |
| `benchmark/perturbed_env.py` | `PerturbedEnvWrapper` — monkey-patches `take_action()` for perturbation injection |
| `benchmark/spec_generator.py` | Generates `benchmark_spec.json` with deterministic perturbation configs |
| `benchmark/aggregate_results.py` | Result aggregation & CSV export |
| `policies/base_adapter.py` | Abstract `PolicyAdapter` — defines `load()`, `reset()`, `step()`, `action_type`, `name` |
| `policies/motus_adapter.py` | Motus adapter (qpos actions) |
| `policies/pi05_adapter.py` | Pi0.5 adapter (qpos actions, 32-step action horizon) |
| `policies/lingbot_va_adapter.py` | LingbotVA adapter (end-effector actions via websocket) |
| `report_generation.py` | Unified cross-policy reporting (Markdown + JSON export) |
| `tasks_all.txt` | List of all 50 benchmark task names |

### Policy adapter pattern

All policies implement `PolicyAdapter` (in `policies/base_adapter.py`):
- `load(config)` — load model weights
- `reset(task_env, instruction)` — reset for new episode
- `step(task_env, observation)` — run one step, **internally calls `task_env.take_action()`**
- `action_type` — `"qpos"` or `"ee"`
- `name` — human-readable name (used as result directory name)

### Perturbation injection

All perturbations are injected at the **14D qpos layer** via `PerturbedEnvWrapper`, which monkey-patches `task_env.take_action()`:
- **qpos policies** (Motus, Pi0.5): perturbation applied directly to the 14D action vector
- **EE policies** (LingbotVA): IK runs normally, then joint targets are perturbed at each waypoint
- **Gripper dimensions (indices 6, 13) are never perturbed**
- Action dim layout: `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`

### Self-contained policy code

All policy inference code is bundled in the repo:
- `policies/motus_policy/` — Motus (WAN backbone + Qwen VLM)
- `policies/pi05_policy/` — Pi0.5 (full openpi inference stack)
- `policies/lingbot_va/` — LingbotVA websocket client (server requires separate `lingbot-va` repo)

### External dependencies (symlinks)

The repo requires symlinks to a sibling `robotwin/` directory:
```
envs -> ../robotwin/envs
task_config -> ../robotwin/task_config
assets -> ../robotwin/assets
description -> ../robotwin/description
script -> ../robotwin/script
```

### Result structure

Each episode produces a JSON file:
```
results/<PolicyName>/<setting_id>/<task_name>/episode_<repeat_idx>.json
```
Episode JSON contains: `success`, `steps`, `error`, `perturbation_log`, `setting_id`, `repeat_idx`, `policy`.

**Resume behavior**: existing episode JSON files are automatically skipped on re-runs.

## Important Implementation Details

- **`--skip-expert-check`** (or `SKIP_EXPERT_CHECK=1`): Always recommended. Skips Phase 1 expert demo verification since seeds are pre-verified in the spec. Avoids CuroboPlanner/IK failures.
- **LingbotVA multi-GPU**: Each GPU needs its own config file pointing to a dedicated server port. Shared configs cause websocket 1011 crashes.
- **Retry logic**: Transient eval errors (websocket failures) are retried up to 3 times.
- **Python envs**: Motus/LingbotVA default to `/gemini/code/envs/robotwin/bin/python`; Pi0.5 defaults to `/gemini/code/envs/robotwin_pi05/bin/python`. Override with `PYTHON=`.
- **No test suite**: The benchmark itself is the test — there are no unit tests for the framework code. Tests in `policies/pi05_policy/` belong to the upstream openpi project.

## Configuration

Policy configs are YAML files in `configs/`. Key fields:
- `robotwin_root`: path to RoboTwin repo
- Policy-specific model paths (checkpoint, weights, etc.)
- LingbotVA: `host`, `port` for websocket server

## RoboTwin Upstream Patches

Two `arm_tag` AttributeError bugs in RoboTwin must be patched when using `--skip-expert-check`:
- `envs/open_laptop.py` — `check_success()` needs `hasattr(self, 'arm_tag')` guard
- `envs/place_object_scale.py` — same pattern

See README.md "RoboTwin Upstream Bug Fixes" section for exact patches.
