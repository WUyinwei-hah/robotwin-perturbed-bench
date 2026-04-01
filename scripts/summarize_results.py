#!/usr/bin/env python3
"""Summarize perturbed benchmark results for one policy run.

Expected layout:
  <results_root>/<policy_name>/<setting_id>/<task_name>/episode_*.json

Example:
  python scripts/summarize_results.py \
      --results-root results/exp_20260304_scale_baseline_v1 \
      --policy lingbot \
      --output-json /tmp/summary.json \
      --output-md /tmp/summary.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _safe_rate(success: int, total: int) -> float:
    return float(success) / float(total) if total > 0 else 0.0


def load_episode_results(policy_dir: Path) -> List[dict]:
    episodes: List[dict] = []
    for ep_json in policy_dir.glob("*/*/episode_*.json"):
        try:
            with ep_json.open("r", encoding="utf-8") as f:
                item = json.load(f)
            item["_path"] = str(ep_json)
            episodes.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    return episodes


def summarize(episodes: List[dict]) -> Dict:
    by_setting: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})
    by_task: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})
    by_setting_task: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

    error_count = 0
    for ep in episodes:
        setting_id = str(ep.get("setting_id", "unknown_setting"))
        task = str(ep.get("task", "unknown_task"))
        success = bool(ep.get("success", False))
        has_error = bool(ep.get("error"))

        by_setting[setting_id]["total"] += 1
        by_task[task]["total"] += 1
        by_setting_task[(setting_id, task)]["total"] += 1

        if success:
            by_setting[setting_id]["success"] += 1
            by_task[task]["success"] += 1
            by_setting_task[(setting_id, task)]["success"] += 1

        if has_error:
            error_count += 1

    total = len(episodes)
    success_total = sum(1 for e in episodes if bool(e.get("success", False)))

    out = {
        "overall": {
            "episodes": total,
            "success": success_total,
            "success_rate": _safe_rate(success_total, total),
            "error_episodes": error_count,
        },
        "by_setting": {},
        "by_task": {},
        "by_setting_task": {},
    }

    for setting_id, stats in sorted(by_setting.items()):
        out["by_setting"][setting_id] = {
            "episodes": stats["total"],
            "success": stats["success"],
            "success_rate": _safe_rate(stats["success"], stats["total"]),
        }

    for task, stats in sorted(by_task.items()):
        out["by_task"][task] = {
            "episodes": stats["total"],
            "success": stats["success"],
            "success_rate": _safe_rate(stats["success"], stats["total"]),
        }

    for (setting_id, task), stats in sorted(by_setting_task.items()):
        key = f"{setting_id}::{task}"
        out["by_setting_task"][key] = {
            "setting_id": setting_id,
            "task": task,
            "episodes": stats["total"],
            "success": stats["success"],
            "success_rate": _safe_rate(stats["success"], stats["total"]),
        }

    return out


def to_markdown(summary: Dict) -> str:
    lines: List[str] = []

    o = summary["overall"]
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| episodes | success | success_rate | error_episodes |")
    lines.append("|---:|---:|---:|---:|")
    lines.append(
        f"| {o['episodes']} | {o['success']} | {o['success_rate']:.4f} | {o['error_episodes']} |"
    )
    lines.append("")

    lines.append("## By Setting")
    lines.append("")
    lines.append("| setting_id | episodes | success | success_rate |")
    lines.append("|---|---:|---:|---:|")
    for setting_id, stats in summary["by_setting"].items():
        lines.append(
            f"| {setting_id} | {stats['episodes']} | {stats['success']} | {stats['success_rate']:.4f} |"
        )
    lines.append("")

    lines.append("## By Task")
    lines.append("")
    lines.append("| task | episodes | success | success_rate |")
    lines.append("|---|---:|---:|---:|")
    for task, stats in summary["by_task"].items():
        lines.append(
            f"| {task} | {stats['episodes']} | {stats['success']} | {stats['success_rate']:.4f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark episode_*.json results")
    parser.add_argument("--results-root", type=str, required=True, help="Root containing policy result folders")
    parser.add_argument("--policy", type=str, default="lingbot", help="Policy folder name under results root")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save summary JSON")
    parser.add_argument("--output-md", type=str, default=None, help="Optional path to save markdown summary")
    args = parser.parse_args()

    policy_dir = Path(args.results_root) / args.policy
    if not policy_dir.exists():
        raise FileNotFoundError(f"Policy result directory not found: {policy_dir}")

    episodes = load_episode_results(policy_dir)
    summary = summarize(episodes)

    print(json.dumps(summary["overall"], indent=2))

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(to_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
