"""
Aggregate Results for the Perturbed RoboTwin Benchmark.

Parses episode JSON results and produces summary tables (per-setting, per-task,
per-policy) and CSV exports.

Usage:
    python -m benchmark.aggregate_results --results-dir results --output summary
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_all_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all episode result JSONs from the results directory."""
    results = []
    results_path = Path(results_dir)

    for policy_dir in sorted(results_path.iterdir()):
        if not policy_dir.is_dir():
            continue
        policy_name = policy_dir.name

        for setting_dir in sorted(policy_dir.iterdir()):
            if not setting_dir.is_dir():
                continue
            setting_id = setting_dir.name

            for task_dir in sorted(setting_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                task_name = task_dir.name

                for ep_file in sorted(task_dir.glob("episode_*.json")):
                    try:
                        with open(ep_file) as f:
                            result = json.load(f)
                        result.setdefault("policy", policy_name)
                        result.setdefault("setting_id", setting_id)
                        result.setdefault("task", task_name)
                        results.append(result)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"[WARN] Failed to load {ep_file}: {e}")

    return results


def compute_success_rate(results: List[Dict[str, Any]]) -> float:
    """Compute success rate from a list of episode results."""
    if not results:
        return 0.0
    valid = [r for r in results if r.get("error") is None]
    if not valid:
        return 0.0
    successes = sum(1 for r in valid if r.get("success", False))
    return successes / len(valid)


def parse_setting_id(setting_id: str) -> Dict[str, str]:
    """Parse setting_id into components: perturb_type, severity, timing."""
    parts = setting_id.split("_")
    # Format: {type}_{severity}_{timing_part1}[_{timing_part2}]
    # e.g. scale_lm_always_on, bias_high_onset_then_always
    if len(parts) >= 4 and parts[-2] == "always" and parts[-1] == "on":
        timing = "always_on"
        severity = parts[-3]
        perturb_type = "_".join(parts[:-3])
    elif len(parts) >= 5 and parts[-3] == "onset" and parts[-2] == "then" and parts[-1] == "always":
        timing = "onset_then_always"
        severity = parts[-4]
        perturb_type = "_".join(parts[:-4])
    else:
        perturb_type = parts[0] if parts else "unknown"
        severity = parts[1] if len(parts) > 1 else "unknown"
        timing = "_".join(parts[2:]) if len(parts) > 2 else "unknown"

    return {
        "perturb_type": perturb_type,
        "severity": severity,
        "timing": timing,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregated statistics.

    Returns:
        Dict with keys:
            - by_policy: {policy: {success_rate, total, successes}}
            - by_policy_setting: {policy: {setting_id: {success_rate, ...}}}
            - by_policy_task: {policy: {task: {success_rate, ...}}}
            - by_policy_type: {policy: {perturb_type: {success_rate, ...}}}
            - by_policy_severity: {policy: {severity: {success_rate, ...}}}
            - by_policy_timing: {policy: {timing: {success_rate, ...}}}
            - by_policy_setting_task: {policy: {setting_id: {task: {success_rate, ...}}}}
    """
    # Group results
    by_policy = defaultdict(list)
    by_policy_setting = defaultdict(lambda: defaultdict(list))
    by_policy_task = defaultdict(lambda: defaultdict(list))
    by_policy_type = defaultdict(lambda: defaultdict(list))
    by_policy_severity = defaultdict(lambda: defaultdict(list))
    by_policy_timing = defaultdict(lambda: defaultdict(list))
    by_policy_setting_task = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in results:
        policy = r.get("policy", "unknown")
        setting = r.get("setting_id", "unknown")
        task = r.get("task", "unknown")

        by_policy[policy].append(r)
        by_policy_setting[policy][setting].append(r)
        by_policy_task[policy][task].append(r)
        by_policy_setting_task[policy][setting][task].append(r)

        parsed = parse_setting_id(setting)
        by_policy_type[policy][parsed["perturb_type"]].append(r)
        by_policy_severity[policy][parsed["severity"]].append(r)
        by_policy_timing[policy][parsed["timing"]].append(r)

    def make_stats(result_list):
        valid = [r for r in result_list if r.get("error") is None]
        total = len(result_list)
        valid_count = len(valid)
        successes = sum(1 for r in valid if r.get("success", False))
        errors = total - valid_count
        return {
            "total": total,
            "valid": valid_count,
            "successes": successes,
            "errors": errors,
            "success_rate": successes / valid_count if valid_count > 0 else 0.0,
        }

    agg = {}

    # Overall by policy
    agg["by_policy"] = {p: make_stats(rs) for p, rs in by_policy.items()}

    # By policy + setting
    agg["by_policy_setting"] = {
        p: {s: make_stats(rs) for s, rs in settings.items()}
        for p, settings in by_policy_setting.items()
    }

    # By policy + task
    agg["by_policy_task"] = {
        p: {t: make_stats(rs) for t, rs in tasks.items()}
        for p, tasks in by_policy_task.items()
    }

    # By policy + perturb_type
    agg["by_policy_type"] = {
        p: {t: make_stats(rs) for t, rs in types.items()}
        for p, types in by_policy_type.items()
    }

    # By policy + severity
    agg["by_policy_severity"] = {
        p: {s: make_stats(rs) for s, rs in sevs.items()}
        for p, sevs in by_policy_severity.items()
    }

    # By policy + timing
    agg["by_policy_timing"] = {
        p: {t: make_stats(rs) for t, rs in timings.items()}
        for p, timings in by_policy_timing.items()
    }

    return agg


def print_summary(agg: Dict[str, Any]) -> None:
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 80)
    print("PERTURBED ROBOTWIN BENCHMARK — AGGREGATE RESULTS")
    print("=" * 80)

    # Overall
    print("\n## Overall Success Rate by Policy")
    print(f"{'Policy':<20} {'Success Rate':>12} {'Successes':>10} {'Valid':>8} {'Errors':>8}")
    print("-" * 60)
    for policy, stats in sorted(agg["by_policy"].items()):
        print(f"{policy:<20} {stats['success_rate']*100:>11.1f}% {stats['successes']:>10} {stats['valid']:>8} {stats['errors']:>8}")

    # By perturbation type
    print("\n## Success Rate by Perturbation Type")
    for policy, types in sorted(agg["by_policy_type"].items()):
        print(f"\n  Policy: {policy}")
        print(f"  {'Type':<15} {'Success Rate':>12} {'N':>6}")
        print(f"  {'-'*35}")
        for ptype, stats in sorted(types.items()):
            print(f"  {ptype:<15} {stats['success_rate']*100:>11.1f}% {stats['valid']:>6}")

    # By severity
    print("\n## Success Rate by Severity")
    for policy, sevs in sorted(agg["by_policy_severity"].items()):
        print(f"\n  Policy: {policy}")
        print(f"  {'Severity':<15} {'Success Rate':>12} {'N':>6}")
        print(f"  {'-'*35}")
        for sev, stats in sorted(sevs.items()):
            print(f"  {sev:<15} {stats['success_rate']*100:>11.1f}% {stats['valid']:>6}")

    # By timing
    print("\n## Success Rate by Timing")
    for policy, timings in sorted(agg["by_policy_timing"].items()):
        print(f"\n  Policy: {policy}")
        print(f"  {'Timing':<20} {'Success Rate':>12} {'N':>6}")
        print(f"  {'-'*40}")
        for timing, stats in sorted(timings.items()):
            print(f"  {timing:<20} {stats['success_rate']*100:>11.1f}% {stats['valid']:>6}")

    # By setting (top-level)
    print("\n## Success Rate by Setting")
    for policy, settings in sorted(agg["by_policy_setting"].items()):
        print(f"\n  Policy: {policy}")
        print(f"  {'Setting':<40} {'Success Rate':>12} {'N':>6}")
        print(f"  {'-'*60}")
        for setting, stats in sorted(settings.items()):
            print(f"  {setting:<40} {stats['success_rate']*100:>11.1f}% {stats['valid']:>6}")


def export_csv(agg: Dict[str, Any], output_dir: str) -> None:
    """Export aggregated results to CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Overall
    with open(out / "overall.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "success_rate", "successes", "valid", "errors", "total"])
        for policy, stats in sorted(agg["by_policy"].items()):
            w.writerow([policy, f"{stats['success_rate']:.4f}", stats["successes"],
                        stats["valid"], stats["errors"], stats["total"]])

    # By setting
    with open(out / "by_setting.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "setting_id", "perturb_type", "severity", "timing",
                     "success_rate", "successes", "valid", "errors"])
        for policy, settings in sorted(agg["by_policy_setting"].items()):
            for setting_id, stats in sorted(settings.items()):
                parsed = parse_setting_id(setting_id)
                w.writerow([
                    policy, setting_id, parsed["perturb_type"], parsed["severity"],
                    parsed["timing"], f"{stats['success_rate']:.4f}",
                    stats["successes"], stats["valid"], stats["errors"],
                ])

    # By task
    with open(out / "by_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "task", "success_rate", "successes", "valid", "errors"])
        for policy, tasks in sorted(agg["by_policy_task"].items()):
            for task, stats in sorted(tasks.items()):
                w.writerow([
                    policy, task, f"{stats['success_rate']:.4f}",
                    stats["successes"], stats["valid"], stats["errors"],
                ])

    # By perturbation type
    with open(out / "by_perturb_type.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "perturb_type", "success_rate", "successes", "valid"])
        for policy, types in sorted(agg["by_policy_type"].items()):
            for ptype, stats in sorted(types.items()):
                w.writerow([policy, ptype, f"{stats['success_rate']:.4f}",
                            stats["successes"], stats["valid"]])

    # By severity
    with open(out / "by_severity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "severity", "success_rate", "successes", "valid"])
        for policy, sevs in sorted(agg["by_policy_severity"].items()):
            for sev, stats in sorted(sevs.items()):
                w.writerow([policy, sev, f"{stats['success_rate']:.4f}",
                            stats["successes"], stats["valid"]])

    # By timing
    with open(out / "by_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "timing", "success_rate", "successes", "valid"])
        for policy, timings in sorted(agg["by_policy_timing"].items()):
            for timing, stats in sorted(timings.items()):
                w.writerow([policy, timing, f"{stats['success_rate']:.4f}",
                            stats["successes"], stats["valid"]])

    print(f"\nCSV files exported to {out}/")


def export_json(agg: Dict[str, Any], output_dir: str) -> None:
    """Export aggregated results as JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"JSON exported to {out}/aggregate.json")


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root results directory")
    parser.add_argument("--output", type=str, default="summary",
                        help="Output directory for CSV/JSON summaries")
    parser.add_argument("--policies", type=str, default=None,
                        help="Comma-separated policy names to include (default: all)")
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    results = load_all_results(args.results_dir)

    if args.policies:
        filter_policies = set(args.policies.split(","))
        results = [r for r in results if r.get("policy") in filter_policies]

    print(f"Loaded {len(results)} episode results")

    if not results:
        print("No results found. Exiting.")
        return

    agg = aggregate(results)
    print_summary(agg)
    export_csv(agg, args.output)
    export_json(agg, args.output)


if __name__ == "__main__":
    main()
