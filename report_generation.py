#!/usr/bin/env python3
"""Unified benchmark report generator.

Scans a results directory tree with layout:
    <results_root>/<policy>/<setting_id>/<task>/episode_<N>.json

Reads benchmark_spec.json to compute expected episode counts and reports:
  - Overall progress / success rate per policy
  - Per-setting breakdown (perturb_type, severity, timing)
  - Per-task breakdown
  - Cross-policy comparison table
  - Top error types
  - **Effective-only success rate**: success rate computed only on episodes
    whose perturbation was verified to cause expert failure (caused_failure=True).
    Episodes where the expert still succeeds despite perturbation indicate
    perturbation-insensitive (setting, task) pairs and are excluded from this metric.

Supports any number of policies (auto-detected or explicitly listed).
Outputs to terminal (human-readable) and optionally to JSON/Markdown for
reproducible, cross-device consistent reporting.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_spec(spec_path: Path) -> Dict[str, Any]:
    with spec_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_rate(num: int, den: int) -> float:
    return (num / den) if den > 0 else 0.0


def normalize_error(err: Any) -> str:
    if err is None:
        return ""
    text = str(err).strip()
    if not text:
        return "unknown_error"
    prefixes = [
        "expert_check_error",
        "eval_error",
        "unstable_seed",
        "expert_failed",
    ]
    for p in prefixes:
        if text.startswith(p):
            return p
    if ":" in text:
        return text.split(":", 1)[0].strip()
    return text


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def discover_policies(results_root: Path) -> List[str]:
    """Auto-detect policy names from first-level subdirectories."""
    if not results_root.is_dir():
        return []
    return sorted(
        d.name for d in results_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def collect_records(
    results_root: Path,
    policies: List[str],
    max_repeat: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Collect all episode_*.json records under results_root for given policies."""
    records: List[Dict[str, Any]] = []
    io_errors: Counter = Counter()

    for policy in policies:
        policy_dir = results_root / policy
        if not policy_dir.is_dir():
            continue
        for ep_file in policy_dir.rglob("episode_*.json"):
            try:
                with ep_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                io_errors["json_decode_error"] += 1
                continue
            except OSError:
                io_errors["os_error"] += 1
                continue

            # Infer repeat_idx from data or filename
            repeat_idx = data.get("repeat_idx")
            if repeat_idx is None:
                try:
                    repeat_idx = int(ep_file.stem.split("_")[1])
                except (ValueError, IndexError):
                    repeat_idx = None

            if max_repeat is not None and repeat_idx is not None and int(repeat_idx) >= max_repeat:
                continue

            data.setdefault("repeat_idx", repeat_idx)

            # Fill missing fields from path: <policy>/<setting_id>/<task>/episode_N.json
            parts = ep_file.relative_to(results_root).parts
            if len(parts) >= 4:
                data.setdefault("policy", parts[0])
                data.setdefault("setting_id", parts[1])
                data.setdefault("task", parts[2])
            elif len(parts) >= 3:
                data.setdefault("policy", policy)
                data.setdefault("setting_id", parts[0])
                data.setdefault("task", parts[1])
            data.setdefault("policy", policy)
            records.append(data)

    return records, io_errors


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _group_stats(records: List[Dict[str, Any]], expected: int) -> Dict[str, Any]:
    """Compute stats for an arbitrary group of episode records.

    Includes an *effective-only* success rate that counts only episodes whose
    perturbation was verified as effective (caused_failure == True).  Episodes
    where the perturbation did NOT cause the expert to fail are excluded from
    this metric because the task is inherently insensitive to that perturbation.
    """
    total = len(records)
    valid = [r for r in records if r.get("error") is None]
    errors = [r for r in records if r.get("error") is not None]
    successes = sum(1 for r in valid if bool(r.get("success", False)))

    # Effective-only: perturbation actually caused expert failure
    eff_valid = [r for r in valid if r.get("_caused_failure") is True]
    eff_successes = sum(1 for r in eff_valid if bool(r.get("success", False)))
    # Count how many effective episodes exist (including errored ones)
    eff_total = sum(1 for r in records if r.get("_caused_failure") is True)

    return {
        "completed": total,
        "expected": expected,
        "progress": _safe_rate(total, expected),
        "valid": len(valid),
        "errors": len(errors),
        "successes": successes,
        "success_rate": _safe_rate(successes, len(valid)),
        "eff_total": eff_total,
        "eff_valid": len(eff_valid),
        "eff_successes": eff_successes,
        "eff_success_rate": _safe_rate(eff_successes, len(eff_valid)),
    }


def build_setting_map(spec: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Map setting_id -> {perturb_type, severity, timing}."""
    return {s["id"]: s for s in spec["settings"]}


def _build_caused_failure_lookup(spec: Dict[str, Any]) -> Dict[str, bool]:
    """Build a (setting_id, task, repeat_idx) -> caused_failure lookup.

    Returns a dict keyed by ``"<setting_id>/<task>/<repeat_idx>"`` -> bool.
    If the spec does not contain ``caused_failure`` fields the dict is empty,
    and the effective-only columns will simply show "N/A".
    """
    pc = spec.get("perturbation_configs", {})
    lookup: Dict[str, bool] = {}
    for sid, task_map in pc.items():
        if not isinstance(task_map, dict):
            continue
        for task, configs in task_map.items():
            if not isinstance(configs, list):
                continue
            for idx, cfg in enumerate(configs):
                if "caused_failure" in cfg:
                    lookup[f"{sid}/{task}/{idx}"] = bool(cfg["caused_failure"])
    return lookup


def compute_full_report(
    records: List[Dict[str, Any]],
    io_errors: Counter,
    spec: Dict[str, Any],
    include_all_tasks: bool,
) -> Dict[str, Any]:
    """Build the complete report data structure."""
    settings = spec["settings"]
    setting_ids = [s["id"] for s in settings]
    setting_map = build_setting_map(spec)
    all_tasks = spec["tasks"]
    repeats = int(spec["repeats_per_setting"])

    # Build caused_failure lookup and tag each record
    cf_lookup = _build_caused_failure_lookup(spec)
    has_cf_data = len(cf_lookup) > 0
    for r in records:
        sid = r.get("setting_id", "")
        task = r.get("task", "")
        # repeat_idx from the record itself or inferred earlier
        ridx = r.get("repeat_idx")
        if ridx is None:
            ridx = r.get("_repeat_idx")
        key = f"{sid}/{task}/{ridx}"
        r["_caused_failure"] = cf_lookup.get(key)  # True / False / None

    # Group records by policy
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_policy[r.get("policy", "unknown")].append(r)
    policy_names = sorted(by_policy.keys())

    observed_tasks = sorted({r.get("task", "unknown") for r in records})
    tasks_list = all_tasks if include_all_tasks else observed_tasks

    report: Dict[str, Any] = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "spec_summary": {
            "num_settings": len(settings),
            "num_tasks": len(all_tasks),
            "repeats_per_setting": repeats,
            "total_per_policy": len(settings) * len(all_tasks) * repeats,
        },
        "has_caused_failure_data": has_cf_data,
        "policies": policy_names,
        "io_errors": dict(io_errors),
        "per_policy": {},
    }

    # Error counter for all records combined
    all_error_records = [r for r in records if r.get("error") is not None]
    report["global_error_types"] = dict(
        Counter(normalize_error(r.get("error")) for r in all_error_records).most_common(20)
    )

    for pol in policy_names:
        pol_records = by_policy[pol]
        expected_total = len(settings) * len(tasks_list) * repeats

        # Overall
        overall = _group_stats(pol_records, expected_total)

        # Error types for this policy
        pol_errors = [r for r in pol_records if r.get("error") is not None]
        error_types = dict(
            Counter(normalize_error(r.get("error")) for r in pol_errors).most_common(20)
        )

        # Per-setting
        by_setting: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in pol_records:
            by_setting[r.get("setting_id", "unknown")].append(r)

        per_setting: List[Dict[str, Any]] = []
        for sid in setting_ids:
            s_recs = by_setting.get(sid, [])
            expected_s = len(tasks_list) * repeats
            row = _group_stats(s_recs, expected_s)
            row["setting_id"] = sid
            meta = setting_map.get(sid, {})
            row["perturb_type"] = meta.get("perturb_type", "")
            row["severity"] = meta.get("severity", "")
            row["timing"] = meta.get("timing", "")
            per_setting.append(row)

        # Per-setting grouped by dimension (perturb_type, severity, timing)
        per_dim: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for dim_key in ("perturb_type", "severity", "timing"):
            groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in pol_records:
                sid = r.get("setting_id", "unknown")
                meta = setting_map.get(sid, {})
                groups[meta.get(dim_key, "unknown")].append(r)
            dim_stats: Dict[str, Dict[str, Any]] = {}
            for gname, grecs in sorted(groups.items()):
                n_settings_in_group = sum(
                    1 for s in settings if s.get(dim_key) == gname
                )
                exp = n_settings_in_group * len(tasks_list) * repeats
                dim_stats[gname] = _group_stats(grecs, exp)
            per_dim[dim_key] = dim_stats

        # Per-task
        by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in pol_records:
            by_task[r.get("task", "unknown")].append(r)

        expected_per_task = len(settings) * repeats
        per_task: List[Dict[str, Any]] = []
        for t in tasks_list:
            row = _group_stats(by_task.get(t, []), expected_per_task)
            row["task"] = t
            per_task.append(row)
        per_task.sort(key=lambda x: (x["progress"], x["success_rate"], x["task"]))

        report["per_policy"][pol] = {
            "overall": overall,
            "error_types": error_types,
            "per_setting": per_setting,
            "per_dimension": per_dim,
            "per_task": per_task,
        }

    # Cross-policy comparison (one row per setting)
    cross_policy: List[Dict[str, Any]] = []
    for sid in setting_ids:
        row: Dict[str, Any] = {"setting_id": sid}
        meta = setting_map.get(sid, {})
        row["perturb_type"] = meta.get("perturb_type", "")
        row["severity"] = meta.get("severity", "")
        row["timing"] = meta.get("timing", "")
        for pol in policy_names:
            pol_records = by_policy[pol]
            s_recs = [r for r in pol_records if r.get("setting_id") == sid]
            expected_s = len(tasks_list) * repeats
            row[pol] = _group_stats(s_recs, expected_s)
        cross_policy.append(row)
    report["cross_policy_by_setting"] = cross_policy

    return report


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_terminal_report(report: Dict[str, Any], top_k: int) -> None:
    spec = report["spec_summary"]
    print(f"\nGenerated: {report['generated_at']}")
    print(
        f"Spec: {spec['num_settings']} settings × "
        f"{spec['num_tasks']} tasks × "
        f"{spec['repeats_per_setting']} repeats = "
        f"{spec['total_per_policy']} episodes/policy"
    )
    print(f"Policies detected: {', '.join(report['policies'])}")

    if report["io_errors"]:
        print(f"\nI/O errors during scan: {dict(report['io_errors'])}")

    has_cf = report.get("has_caused_failure_data", False)

    for pol in report["policies"]:
        data = report["per_policy"][pol]
        ov = data["overall"]

        print("\n" + "=" * 96)
        print(f"  Policy: {pol}")
        print("=" * 96)
        print(
            f"  Progress : {ov['completed']:>6}/{ov['expected']}  ({_pct(ov['progress'])})"
        )
        print(
            f"  Valid    : {ov['valid']:>6}   Errors: {ov['errors']}"
        )
        print(
            f"  Success  : {ov['successes']:>6}/{ov['valid']}  ({_pct(ov['success_rate'])}  valid-only)"
        )
        if has_cf and ov["eff_valid"] > 0:
            print(
                f"  Effective: {ov['eff_successes']:>6}/{ov['eff_valid']}  "
                f"({_pct(ov['eff_success_rate'])}  effective-perturbation-only)"
            )

        # Error types
        if data["error_types"]:
            print(f"\n  Top errors (max {top_k}):")
            for i, (etype, cnt) in enumerate(data["error_types"].items()):
                if i >= top_k:
                    break
                print(f"    {etype}: {cnt}")

        # Per dimension
        print("\n  --- By Perturbation Dimension ---")
        for dim_key in ("perturb_type", "severity", "timing"):
            dim_data = data["per_dimension"].get(dim_key, {})
            if not dim_data:
                continue
            print(f"\n  [{dim_key}]")
            hdr = f"  {'group':<25} {'progress':>12} {'valid':>8} {'success':>14}"
            sep = f"  {'-'*25} {'-'*12} {'-'*8} {'-'*14}"
            if has_cf:
                hdr += f" {'eff_success':>18}"
                sep += f" {'-'*18}"
            print(hdr)
            print(sep)
            for gname, gs in dim_data.items():
                prog_str = f"{gs['completed']}/{gs['expected']}"
                succ_str = f"{gs['successes']}/{gs['valid']} ({_pct(gs['success_rate'])})"
                line = f"  {gname:<25} {prog_str:>12} {gs['valid']:>8} {succ_str:>14}"
                if has_cf:
                    if gs['eff_valid'] > 0:
                        eff_str = f"{gs['eff_successes']}/{gs['eff_valid']} ({_pct(gs['eff_success_rate'])})"
                    else:
                        eff_str = "N/A"
                    line += f" {eff_str:>18}"
                print(line)

        # Per setting
        print(f"\n  --- Per Setting ({len(data['per_setting'])} total) ---")
        hdr_s = (
            f"  {'setting_id':<40} {'type':<12} {'sev':<6} {'timing':<20} "
            f"{'progress':>12} {'success':>14}"
        )
        sep_s = f"  {'-'*40} {'-'*12} {'-'*6} {'-'*20} {'-'*12} {'-'*14}"
        if has_cf:
            hdr_s += f" {'eff_success':>18}"
            sep_s += f" {'-'*18}"
        print(hdr_s)
        print(sep_s)
        for row in data["per_setting"]:
            prog_str = f"{row['completed']}/{row['expected']}"
            succ_str = f"{row['successes']}/{row['valid']} ({_pct(row['success_rate'])})"
            line = (
                f"  {row['setting_id']:<40} {row['perturb_type']:<12} "
                f"{row['severity']:<6} {row['timing']:<20} "
                f"{prog_str:>12} {succ_str:>14}"
            )
            if has_cf:
                if row['eff_valid'] > 0:
                    eff_str = f"{row['eff_successes']}/{row['eff_valid']} ({_pct(row['eff_success_rate'])})"
                else:
                    eff_str = "N/A"
                line += f" {eff_str:>18}"
            print(line)

        # Per task (worst / best)
        print(f"\n  --- Task Progress (worst {top_k}) ---")
        for row in data["per_task"][:top_k]:
            prog_str = f"{row['completed']}/{row['expected']}"
            succ_str = f"{row['successes']}/{row['valid']} ({_pct(row['success_rate'])})"
            line = f"    {row['task']:<35} progress={prog_str:<12} success={succ_str}"
            if has_cf and row['eff_valid'] > 0:
                eff_str = f"{row['eff_successes']}/{row['eff_valid']} ({_pct(row['eff_success_rate'])})"
                line += f"  eff={eff_str}"
            print(line)

        best = sorted(
            [r for r in data["per_task"] if r["valid"] > 0],
            key=lambda x: (-x["success_rate"], -x["valid"], x["task"]),
        )
        if best:
            print(f"\n  --- Task Success (best {top_k}) ---")
            for row in best[:top_k]:
                prog_str = f"{row['completed']}/{row['expected']}"
                succ_str = f"{row['successes']}/{row['valid']} ({_pct(row['success_rate'])})"
                line = f"    {row['task']:<35} success={succ_str:<16} progress={prog_str}"
                if has_cf and row['eff_valid'] > 0:
                    eff_str = f"{row['eff_successes']}/{row['eff_valid']} ({_pct(row['eff_success_rate'])})"
                    line += f"  eff={eff_str}"
                print(line)

    # Cross-policy comparison
    if len(report["policies"]) > 1:
        print("\n" + "=" * 96)
        print("  Cross-Policy Comparison (success rate, valid-only)")
        print("=" * 96)
        pols = report["policies"]
        header = f"  {'setting_id':<40} " + " ".join(f"{p:>14}" for p in pols)
        print(header)
        print(f"  {'-'*40} " + " ".join(f"{'-'*14}" for _ in pols))
        for row in report["cross_policy_by_setting"]:
            cells = []
            for p in pols:
                gs = row.get(p, {})
                if gs.get("valid", 0) > 0:
                    cells.append(f"{gs['successes']}/{gs['valid']} {_pct(gs['success_rate'])}")
                else:
                    cells.append("--")
            line = f"  {row['setting_id']:<40} " + " ".join(f"{c:>14}" for c in cells)
            print(line)


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def export_markdown(report: Dict[str, Any], out_path: Path) -> None:
    lines: List[str] = []
    W = lines.append

    spec = report["spec_summary"]
    W(f"# Benchmark Report")
    W(f"")
    W(f"Generated: `{report['generated_at']}`")
    W(f"")
    W(f"| Item | Value |")
    W(f"|------|-------|")
    W(f"| Settings | {spec['num_settings']} |")
    W(f"| Tasks | {spec['num_tasks']} |")
    W(f"| Repeats/setting | {spec['repeats_per_setting']} |")
    W(f"| Episodes/policy | {spec['total_per_policy']} |")
    W(f"| Policies | {', '.join(report['policies'])} |")
    W(f"")

    has_cf = report.get("has_caused_failure_data", False)

    for pol in report["policies"]:
        data = report["per_policy"][pol]
        ov = data["overall"]
        W(f"## {pol}")
        W(f"")
        W(f"| Metric | Value |")
        W(f"|--------|-------|")
        W(f"| Progress | {ov['completed']}/{ov['expected']} ({_pct(ov['progress'])}) |")
        W(f"| Valid | {ov['valid']} |")
        W(f"| Errors | {ov['errors']} |")
        W(f"| Success (valid) | {ov['successes']}/{ov['valid']} ({_pct(ov['success_rate'])}) |")
        if has_cf and ov["eff_valid"] > 0:
            W(f"| **Effective-only success** | {ov['eff_successes']}/{ov['eff_valid']} ({_pct(ov['eff_success_rate'])}) |")
        W(f"")

        # Per dimension tables
        for dim_key in ("perturb_type", "severity", "timing"):
            dim_data = data["per_dimension"].get(dim_key, {})
            if not dim_data:
                continue
            W(f"### By {dim_key}")
            W(f"")
            if has_cf:
                W(f"| {dim_key} | completed | valid | success | rate | eff_success | eff_rate |")
                W(f"|---|---|---|---|---|---|---|")
                for gname, gs in dim_data.items():
                    eff_s = f"{gs['eff_successes']}/{gs['eff_valid']}" if gs['eff_valid'] > 0 else "N/A"
                    eff_r = _pct(gs['eff_success_rate']) if gs['eff_valid'] > 0 else "N/A"
                    W(
                        f"| {gname} | {gs['completed']}/{gs['expected']} "
                        f"| {gs['valid']} | {gs['successes']} | {_pct(gs['success_rate'])} "
                        f"| {eff_s} | {eff_r} |"
                    )
            else:
                W(f"| {dim_key} | completed | valid | success | rate |")
                W(f"|---|---|---|---|---|")
                for gname, gs in dim_data.items():
                    W(
                        f"| {gname} | {gs['completed']}/{gs['expected']} "
                        f"| {gs['valid']} | {gs['successes']} | {_pct(gs['success_rate'])} |"
                    )
            W(f"")

        # Per setting
        W(f"### Per Setting")
        W(f"")
        if has_cf:
            W(f"| setting_id | type | severity | timing | progress | success_rate | eff_success_rate |")
            W(f"|---|---|---|---|---|---|---|")
            for row in data["per_setting"]:
                eff_r = _pct(row['eff_success_rate']) if row['eff_valid'] > 0 else "N/A"
                W(
                    f"| {row['setting_id']} | {row['perturb_type']} | {row['severity']} "
                    f"| {row['timing']} | {row['completed']}/{row['expected']} "
                    f"| {_pct(row['success_rate'])} | {eff_r} |"
                )
        else:
            W(f"| setting_id | type | severity | timing | progress | success_rate |")
            W(f"|---|---|---|---|---|---|")
            for row in data["per_setting"]:
                W(
                    f"| {row['setting_id']} | {row['perturb_type']} | {row['severity']} "
                    f"| {row['timing']} | {row['completed']}/{row['expected']} "
                    f"| {_pct(row['success_rate'])} |"
                )
        W(f"")

        # Per task
        W(f"### Per Task")
        W(f"")
        if has_cf:
            W(f"| task | completed | expected | progress | successes | valid | success_rate | eff_success | eff_rate |")
            W(f"|---|---|---|---|---|---|---|---|---|")
        else:
            W(f"| task | completed | expected | progress | successes | valid | success_rate |")
            W(f"|---|---|---|---|---|---|---|")
        sorted_tasks = sorted(data["per_task"], key=lambda x: x["task"])
        for row in sorted_tasks:
            base = (
                f"| {row['task']} | {row['completed']} | {row['expected']} "
                f"| {_pct(row['progress'])} | {row['successes']} | {row['valid']} "
                f"| {_pct(row['success_rate'])}"
            )
            if has_cf:
                eff_s = f"{row['eff_successes']}/{row['eff_valid']}" if row['eff_valid'] > 0 else "N/A"
                eff_r = _pct(row['eff_success_rate']) if row['eff_valid'] > 0 else "N/A"
                base += f" | {eff_s} | {eff_r}"
            base += " |"
            W(base)
        W(f"")

    # Cross-policy
    if len(report["policies"]) > 1:
        pols = report["policies"]
        W(f"## Cross-Policy Comparison")
        W(f"")
        W(f"| setting_id | " + " | ".join(pols) + " |")
        W(f"|---" + "|---" * len(pols) + "|")
        for row in report["cross_policy_by_setting"]:
            cells = []
            for p in pols:
                gs = row.get(p, {})
                if gs.get("valid", 0) > 0:
                    cells.append(f"{_pct(gs['success_rate'])}")
                else:
                    cells.append("--")
            W(f"| {row['setting_id']} | " + " | ".join(cells) + " |")
        W(f"")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown report saved to: {out_path}")


# ---------------------------------------------------------------------------
# JSON export (machine-readable, cross-device portable)
# ---------------------------------------------------------------------------

def export_json(report: Dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON report saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified benchmark report: progress, success rate, per-setting / per-task / cross-policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect all policies under results/
  python report_generation.py --results-dir results

  # Explicit policies
  python report_generation.py --results-dir results --policy Motus --policy lingbot

  # Export markdown + JSON
  python report_generation.py --results-dir results --export-md report.md --export-json report.json

  # Only count first 20 repeats
  python report_generation.py --results-dir results --max-repeat 20

  # Use a different spec
  python report_generation.py --results-dir results --spec benchmark/verified_spec.json
""",
    )
    parser.add_argument(
        "--spec",
        default="benchmark/benchmark_spec.json",
        help="Path to benchmark spec JSON (default: benchmark/benchmark_spec.json)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root results directory containing <policy>/<setting>/<task>/episode_*.json",
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        dest="policies",
        help="Policy name to include (repeatable). If omitted, auto-detect from results-dir.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of rows for top/bottom task and error summaries (default: 8)",
    )
    parser.add_argument(
        "--include-all-tasks",
        action="store_true",
        help="Use ALL tasks in spec for expected counts (even if not yet observed).",
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=None,
        help="Only count episodes with repeat_idx < N.",
    )
    parser.add_argument(
        "--export-json",
        default=None,
        help="Export full report to JSON file.",
    )
    parser.add_argument(
        "--export-md",
        default=None,
        help="Export full report to Markdown file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal output (only write export files).",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"ERROR: spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)
    spec = load_spec(spec_path)

    results_root = Path(args.results_dir)
    if not results_root.is_dir():
        print(f"ERROR: results dir not found: {results_root}", file=sys.stderr)
        sys.exit(1)

    policies = args.policies if args.policies else discover_policies(results_root)
    if not policies:
        print(f"ERROR: no policies found under {results_root}", file=sys.stderr)
        sys.exit(1)

    records, io_errors = collect_records(results_root, policies, max_repeat=args.max_repeat)
    report = compute_full_report(
        records=records,
        io_errors=io_errors,
        spec=spec,
        include_all_tasks=args.include_all_tasks,
    )

    if not args.quiet:
        print_terminal_report(report, top_k=args.top_k)

    if args.export_json:
        export_json(report, Path(args.export_json))
    if args.export_md:
        export_markdown(report, Path(args.export_md))


if __name__ == "__main__":
    main()
