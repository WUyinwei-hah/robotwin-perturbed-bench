# Benchmark Report Generation

`report_generation.py` is a unified CLI tool that scans experiment results and
produces progress / success-rate statistics broken down by **policy**, **setting**,
**perturbation dimension**, and **task**.

It supports any number of policies (auto-detected or explicitly listed) and
outputs consistent reports to the **terminal**, **Markdown**, or **JSON** —
ensuring identical formatting across different machines and environments.

---

## Prerequisites

- Python ≥ 3.9 (no extra packages needed; uses only stdlib)
- A `benchmark_spec.json` (or `verified_spec.json`) describing settings, tasks,
  and repeats
- A results directory with the standard layout:

```
<results_root>/
  <policy_A>/
    <setting_id>/
      <task>/
        episode_0.json
        episode_1.json
        ...
  <policy_B>/
    ...
```

Each `episode_*.json` must contain at least:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the episode succeeded |
| `error` | `str \| null` | Error message, or `null` if valid |
| `setting_id` | `str` | Setting identifier (also inferred from path) |
| `task` | `str` | Task name (also inferred from path) |
| `repeat_idx` | `int` | Repeat index (also inferred from filename) |
| `policy` | `str` | Policy name (also inferred from path) |

---

## Quick Start

```bash
cd /path/to/robotwin-perturbed-bench

# 1. Auto-detect all policies, print to terminal
python report_generation.py --results-dir results

# 2. Export to Markdown + JSON (portable across devices)
python report_generation.py --results-dir results \
    --export-md report.md \
    --export-json report.json
```

---

## CLI Reference

```
python report_generation.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--spec PATH` | `benchmark/benchmark_spec.json` | Path to benchmark spec JSON |
| `--results-dir DIR` | `results` | Root results directory |
| `--policy NAME` | *(auto-detect)* | Policy to include (repeatable for multiple policies) |
| `--top-k N` | `8` | Number of top/bottom rows in task and error summaries |
| `--include-all-tasks` | off | Use ALL tasks from spec for expected counts, even if not yet observed |
| `--max-repeat N` | *(none)* | Only count episodes with `repeat_idx < N` |
| `--export-json PATH` | *(none)* | Write full report to a JSON file |
| `--export-md PATH` | *(none)* | Write full report to a Markdown file |
| `--quiet` | off | Suppress terminal output (only write export files) |

---

## Usage Examples

### 1. All policies, terminal only

```bash
python report_generation.py --results-dir results
```

### 2. Specific policies

```bash
python report_generation.py --results-dir results \
    --policy Motus --policy lingbot
```

### 3. Use verified spec instead of default

```bash
python report_generation.py --results-dir results \
    --spec benchmark/verified_spec.json
```

### 4. Only first 20 repeats

```bash
python report_generation.py --results-dir results --max-repeat 20
```

### 5. Full export (reproducible across devices)

```bash
python report_generation.py --results-dir results \
    --include-all-tasks \
    --export-json reports/report_$(date +%Y%m%d).json \
    --export-md reports/report_$(date +%Y%m%d).md
```

### 6. Silent export (CI / scripted use)

```bash
python report_generation.py --results-dir results \
    --quiet --export-json report.json
```

---

## Report Sections

The report is structured into the following sections (both in terminal and
exported formats):

### Overall (per policy)

| Metric | Description |
|--------|-------------|
| Progress | completed / expected episodes |
| Valid | episodes without errors |
| Errors | episodes with errors |
| Success | successes / valid episodes (valid-only rate) |

### Top Errors

Most common error types, normalized to a short label (e.g. `eval_error`,
`expert_check_error`).

### By Perturbation Dimension

Three sub-tables grouping results along each axis of the setting space:

- **`perturb_type`**: scale, coupling, lowpass_iir, lowpass_fir, bias
- **`severity`**: lm, high
- **`timing`**: always_on, onset_then_always

### Per Setting (20 rows)

Full breakdown for each of the 20 setting combinations, showing progress and
success rate.

### Effective-Only Success Rate

When the benchmark spec contains verified perturbation data (i.e.,
`caused_failure` fields in `perturbation_configs`), the report includes an
additional **effective-only success rate** metric at every level.

**Concept**: Some perturbation configs do not actually cause the expert
(clean trajectory replay) to fail. These (setting, task, repeat) combinations
are "perturbation-insensitive" — even a trivial policy could succeed on them.
The effective-only metric **excludes** these easy episodes and computes success
rate only over episodes where the perturbation was verified to break the expert.

This gives a more meaningful measure of a policy's robustness because it
answers: *"When the perturbation actually matters, how often does the policy
still succeed?"*

| Column | Meaning |
|--------|---------|
| `eff_valid` | Valid episodes where `caused_failure=True` |
| `eff_successes` | Successes among `eff_valid` |
| `eff_success_rate` | `eff_successes / eff_valid` |

The metric appears in: overall summary, per-dimension tables, per-setting rows,
per-task rows, and in Markdown/JSON exports. If the spec does not contain
`caused_failure` data, these columns are omitted or show "N/A".

To get effective-only data, use the **verified spec**:

```bash
python report_generation.py --results-dir results \
    --spec benchmark/benchmark_spec_lm20.json
```

### Per Task

- **Worst by completion**: tasks with the least progress
- **Best by success**: tasks with the highest success rate (among those with
  valid episodes)

### Cross-Policy Comparison

When multiple policies are present, a comparison table shows the success rate of
each policy for each setting side by side.

---

## JSON Schema

The exported JSON has the following top-level structure:

```json
{
  "generated_at": "2026-02-28T23:51:54",
  "spec_summary": {
    "num_settings": 20,
    "num_tasks": 50,
    "repeats_per_setting": 50,
    "total_per_policy": 50000
  },
  "policies": ["Motus", "Pi05", "lingbot"],
  "io_errors": {},
  "global_error_types": {"eval_error": 50},
  "per_policy": {
    "<policy_name>": {
      "overall": { "completed", "expected", "progress", "valid", "errors", "successes", "success_rate" },
      "error_types": { "<error_label>": <count> },
      "per_setting": [ { "setting_id", "perturb_type", "severity", "timing", ... } ],
      "per_dimension": {
        "perturb_type": { "<type>": { "completed", "expected", ... } },
        "severity":     { "<sev>":  { ... } },
        "timing":       { "<tim>":  { ... } }
      },
      "per_task": [ { "task", "completed", "expected", "progress", "successes", "valid", "success_rate" } ]
    }
  },
  "cross_policy_by_setting": [
    { "setting_id", "perturb_type", "severity", "timing", "<policy_A>": {...}, "<policy_B>": {...} }
  ]
}
```

This JSON is self-contained and can be loaded on any machine to regenerate
tables or plots without access to the original episode files.

---

## Cross-Device Consistency

To ensure identical reports across different devices:

1. **Use `--export-json`** to save the full report data. JSON output is
   deterministic and portable.
2. **Share the JSON file** (e.g. via git or cloud storage) rather than
   re-scanning results on each device.
3. **Use `--include-all-tasks`** so that expected counts are always based on the
   spec, not on which tasks happen to have results locally.
4. **Pin `--max-repeat`** if comparing partial runs at different stages.

The Markdown export is also deterministic and can be committed to the repo for
version-controlled reporting.
