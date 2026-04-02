# LM20 / Verified Spec Workflow

This note records the benchmark-spec generation workflow used in
`robotwin-perturbed-bench`, including the new mixed-perturbation path.

## Existing Files

- Base benchmark spec generator:
  [benchmark/spec_generator.py](/gemini/code/robotwin-perturbed-bench/benchmark/spec_generator.py)
- Verified perturbation generator:
  [benchmark/generate_verified_spec.py](/gemini/code/robotwin-perturbed-bench/benchmark/generate_verified_spec.py)
- Default shell wrapper for base spec:
  [scripts/generate_spec.sh](/gemini/code/robotwin-perturbed-bench/scripts/generate_spec.sh)

## Existing Generated Specs

- Full spec:
  [benchmark/benchmark_spec.json](/gemini/code/robotwin-perturbed-bench/benchmark/benchmark_spec.json)
- Compact lm-only 20-repeat spec:
  [benchmark/benchmark_spec_lm20.json](/gemini/code/robotwin-perturbed-bench/benchmark/benchmark_spec_lm20.json)
- Verified spec:
  [benchmark/benchmark_spec_verified.json](/gemini/code/robotwin-perturbed-bench/benchmark/benchmark_spec_verified.json)

## What Each Generator Does

### 1. `spec_generator.py`

Purpose:
- generate deterministic benchmark configs
- sample perturbation parameters
- assign one stable env seed per task

By default it emits the full benchmark spec.

### 2. `generate_verified_spec.py`

Purpose:
- take an existing spec as input
- replay clean expert trajectories under sampled perturbations
- keep perturbation configs that actually cause expert failure
- label every collected config with `caused_failure`
- support multi-GPU multi-worker generation and later merge

Important parameters:
- `--num-effective 20`: collect 20 verified configs per `(setting, task)`
- `--severity lm`: restrict to lm settings
- `--merge`: merge worker partial results into one final spec

## How `benchmark_spec_lm20.json` Is Interpreted

`benchmark_spec_lm20.json` is the compact lm-only spec used for quick
evaluation:

- 10 settings
- 50 tasks
- 20 repeats per setting
- 10,000 episodes per policy

Historically it is treated as a pre-verified compact spec in evaluation
workflows, and benchmark runs usually pair it with `--skip-expert-check`.

## New Mixed Support

The benchmark perturbation engine now supports:

- `scale_coupling_bias` as the true composite mixed perturbation type
- setting IDs rendered as `mixed_*`

Examples:
- `mixed_lm_always_on`
- `mixed_lm_onset_then_always`
- `mixed_high_always_on`
- `mixed_high_onset_then_always`

## Recommended Workflow For `mixed lm20 verified`

### Step 1. Build a temporary mixed-only lm20 base spec

Run from repo root:

```bash
/gemini/code/envs/robotwin_motus/bin/python - <<'PY'
from pathlib import Path
import json
from benchmark.spec_generator import generate_benchmark_spec

tasks = [line.strip() for line in Path("tasks_all.txt").read_text().splitlines() if line.strip()]
spec = generate_benchmark_spec(
    tasks=tasks,
    repeats_per_setting=20,
    master_seed=42,
    ensure_stable_seeds=True,
    dataset_root="/gemini/code/datasets/robotwin_dataset/clean",
    verify_seeds=False,
)

mixed_settings = [s for s in spec["settings"] if s["perturb_type"] == "scale_coupling_bias" and s["severity"] == "lm"]
mixed_ids = {s["id"] for s in mixed_settings}

spec["settings"] = mixed_settings
spec["num_settings"] = len(mixed_settings)
spec["total_episodes_per_policy"] = len(mixed_settings) * len(spec["tasks"]) * spec["repeats_per_setting"]
spec["perturbation_configs"] = {k: v for k, v in spec["perturbation_configs"].items() if k in mixed_ids}
spec["env_seeds"] = {k: v for k, v in spec["env_seeds"].items() if k in mixed_ids}
spec["verified"] = False

Path("benchmark/benchmark_spec_mixed_lm20.json").write_text(json.dumps(spec, indent=2))
print("Wrote benchmark/benchmark_spec_mixed_lm20.json")
print("settings:", [s["id"] for s in mixed_settings])
PY
```

Expected settings:
- `mixed_lm_always_on`
- `mixed_lm_onset_then_always`

### Step 2. Launch 8 verification workers

```bash
mkdir -p logs/mixed_lm20_verified verified_spec_parts_mixed_lm20

for gpu in $(seq 0 7); do
  /gemini/code/envs/robotwin_motus/bin/python -m benchmark.generate_verified_spec \
    --spec benchmark/benchmark_spec_mixed_lm20.json \
    --dataset-root /gemini/code/datasets/robotwin_dataset/clean \
    --gpu $gpu \
    --worker-id $gpu \
    --num-workers 8 \
    --num-effective 20 \
    --max-tries 100 \
    --severity lm \
    --output verified_spec_parts_mixed_lm20/part_${gpu}.json \
    > logs/mixed_lm20_verified/worker_${gpu}.log 2>&1 &
done
```

Notes:
- Do not wrap this command with an outer `CUDA_VISIBLE_DEVICES=...` plus
  `--gpu 0`.
- `generate_verified_spec.py` already sets `CUDA_VISIBLE_DEVICES` internally
  from `--gpu`, so the correct multi-GPU launch is simply `--gpu $gpu`.

### Step 3. Merge worker outputs

```bash
/gemini/code/envs/robotwin_motus/bin/python -m benchmark.generate_verified_spec \
  --merge \
  --parts-dir verified_spec_parts_mixed_lm20 \
  --spec benchmark/benchmark_spec_mixed_lm20.json \
  --output benchmark/benchmark_spec_mixed_lm20_verified.json
```

## Output Files For Mixed

- Base mixed lm20 spec:
  [benchmark/benchmark_spec_mixed_lm20.json](/gemini/code/robotwin-perturbed-bench/benchmark/benchmark_spec_mixed_lm20.json)
- Verified mixed lm20 spec:
  [benchmark/benchmark_spec_mixed_lm20_verified.json](/gemini/code/robotwin-perturbed-bench/benchmark/benchmark_spec_mixed_lm20_verified.json)
- Worker parts:
  `/gemini/code/robotwin-perturbed-bench/verified_spec_parts_mixed_lm20/`
- Worker logs:
  `/gemini/code/robotwin-perturbed-bench/logs/mixed_lm20_verified/`

## Practical Recommendation

When running policy evaluation on the new mixed setting:

- prefer the verified mixed spec once it is available
- keep `--skip-expert-check`
- use `mixed_lm_always_on` first as the smallest smoke target
