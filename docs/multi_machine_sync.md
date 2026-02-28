# Multi-Machine Experiment Sync Guide

This document is intended for **AI agents** and **human operators** running
benchmark experiments on multiple machines. It describes the standard workflow
for launching experiments, syncing results via GitHub, and generating unified
reports.

---

## Overview

```
Machine A (this machine)          GitHub repo            Machine B (remote)
  run experiments  ──push──>   results/<policy>/   <──pull── run experiments
  report_generation.py          benchmark specs           report_generation.py
```

All experiment results live under `results/` with the layout:

```
results/<policy>/<setting_id>/<task>/episode_<N>.json
```

Each machine runs a subset of experiments and pushes results to GitHub.
Other machines pull to get a complete picture.

---

## Prerequisites

1. Clone the repo on each machine:
   ```bash
   git clone <REPO_URL> robotwin-perturbed-bench
   cd robotwin-perturbed-bench
   ```

2. Ensure the benchmark spec exists:
   - Full spec (20 settings × 50 tasks × 50 repeats): `benchmark/benchmark_spec.json`
   - **LM-only compact spec** (10 settings × 50 tasks × 20 repeats): `benchmark/benchmark_spec_lm20.json`

3. Ensure the dataset is available at the expected path, or override with
   `--dataset-root`.

---

## Step 1: Run Experiments

Use the eval runner to run a specific policy on a specific spec:

```bash
# Example: run Motus evaluation with lm-only 20-repeat spec
python -m benchmark.eval_runner \
    --spec benchmark/benchmark_spec_lm20.json \
    --policy Motus \
    --gpu 0 \
    --worker-id 0 --num-workers 4 \
    --results-dir results
```

### Multi-GPU parallel launch

```bash
NUM_GPUS=8
POLICY=Motus
SPEC=benchmark/benchmark_spec_lm20.json

for gpu in $(seq 0 $((NUM_GPUS-1))); do
  nohup python -m benchmark.eval_runner \
    --spec $SPEC \
    --policy $POLICY \
    --gpu $gpu \
    --worker-id $gpu --num-workers $NUM_GPUS \
    --results-dir results \
    > logs/eval_${POLICY}_gpu${gpu}.log 2>&1 &
done
```

### Current experiment plan

| Machine | Policies | Spec | Repeats | Severity |
|---------|----------|------|---------|----------|
| Machine A | (verified spec generation) | benchmark_spec.json | 50 | lm |
| Machine B | Motus, lingbot | benchmark_spec_lm20.json | 20 | lm only |

---

## Step 2: Sync Results to GitHub

After experiments finish (or periodically during long runs):

```bash
cd robotwin-perturbed-bench

# Stage only the results directory
git add results/

# Commit with descriptive message
git commit -m "results: <policy> on <machine> - <N> episodes done"

# Pull latest from remote first (merge other machines' results)
git pull --rebase origin main

# Push
git push origin main
```

### Important rules

- **Never force-push** — results from other machines would be lost.
- **Always pull before push** to merge results from other machines.
- **Commit frequently** (e.g., every few hours or after each task completes)
  to minimize merge conflicts.
- Episode JSON files have unique paths (`<policy>/<setting>/<task>/episode_N.json`),
  so **merge conflicts should be rare** as long as different machines run
  different worker partitions.

### Avoiding conflicts

If two machines run the **same** (policy, setting, task, episode) pair, the
second push will conflict. To prevent this:

- Use `--worker-id` / `--num-workers` to partition work.
- Or assign different policies to different machines.
- Or assign different setting subsets to different machines.

---

## Step 3: Generate Report

After pulling the latest results from all machines:

```bash
git pull origin main

# Terminal report (auto-detects all policies)
python report_generation.py --results-dir results --spec benchmark/benchmark_spec_lm20.json

# Export portable JSON + Markdown
python report_generation.py \
    --results-dir results \
    --spec benchmark/benchmark_spec_lm20.json \
    --include-all-tasks \
    --export-json reports/report_$(date +%Y%m%d).json \
    --export-md reports/report_$(date +%Y%m%d).md
```

### Report features

- **Per-policy**: overall progress, success rate, error breakdown
- **Per-setting**: each of the 10 lm settings individually
- **Per-dimension**: grouped by `perturb_type`, `severity`, `timing`
- **Per-task**: worst/best tasks by progress and success
- **Cross-policy**: side-by-side comparison table

### Sharing reports

```bash
git add reports/
git commit -m "reports: update $(date +%Y%m%d)"
git push origin main
```

The JSON report is self-contained and can be loaded on any machine without
access to the raw episode files.

---

## Step 4: Monitor Progress

Quick progress check without full report:

```bash
# Count episodes per policy
for pol in results/*/; do
    name=$(basename $pol)
    count=$(find $pol -name 'episode_*.json' | wc -l)
    echo "$name: $count episodes"
done
```

Or use the report script with `--quiet` to just write JSON:

```bash
python report_generation.py --results-dir results \
    --spec benchmark/benchmark_spec_lm20.json \
    --quiet --export-json /tmp/progress.json

# Parse with jq
cat /tmp/progress.json | python -c "
import json, sys
r = json.load(sys.stdin)
for p in r['policies']:
    ov = r['per_policy'][p]['overall']
    print(f\"{p}: {ov['completed']}/{ov['expected']} ({ov['progress']*100:.1f}%) success={ov['success_rate']*100:.1f}%\")
"
```

---

## Spec Files Reference

| File | Settings | Tasks | Repeats | Total/policy | Use case |
|------|----------|-------|---------|-------------|----------|
| `benchmark_spec.json` | 20 (all) | 50 | 50 | 50,000 | Full benchmark |
| `benchmark_spec_lm20.json` | 10 (lm only) | 50 | 20 | 10,000 | Quick multi-machine eval |
| `benchmark_spec_verified.json` | 10 (lm) | 50 | 20 | 10,000 | Verified perturbations (after generation) |

---

## Troubleshooting

### Merge conflict on pull
```bash
# Episode files should not conflict. If they do:
git pull --rebase origin main
# If rebase fails, prefer the remote version for results:
git checkout --theirs results/
git add results/
git rebase --continue
```

### Missing episodes after sync
Check that the remote machine committed and pushed:
```bash
git log --oneline -5 origin/main
```

### Different spec versions
Always ensure all machines use the **same spec file**. The spec file is
committed to the repo, so `git pull` will sync it. Use `--spec` to
explicitly point to the correct spec.

### Environment differences
The eval runner and report script use only Python stdlib. Results are
plain JSON files. No environment-specific dependencies affect the report.
