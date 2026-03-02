#!/usr/bin/env python3
"""Real-time benchmark monitor for Motus + LingbotVA (5 always_on settings).

Usage:
    python scripts/watch_benchmark.py                # auto-refresh every 30s
    python scripts/watch_benchmark.py --once          # single snapshot
    python scripts/watch_benchmark.py --interval 10   # refresh every 10s
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Settings ───────────────────────────────────────────────────────────
ALWAYS_ON_SETTINGS = [
    "scale_lm_always_on",
    "bias_lm_always_on",
    "coupling_lm_always_on",
    "iir_lm_always_on",
    "fir_lm_always_on",
]
SPEC_PATH = "benchmark/benchmark_spec_lm20.json"
RESULTS_DIR = "results"

# ── Colours ────────────────────────────────────────────────────────────
C = sys.stdout.isatty()
def _c(code: str, t: str) -> str: return f"\033[{code}m{t}\033[0m" if C else t
def G(t): return _c("32", t)
def R(t): return _c("31", t)
def Y(t): return _c("33", t)
def B(t): return _c("1", t)
def D(t): return _c("2", t)
def CY(t): return _c("36", t)

# ── Helpers ────────────────────────────────────────────────────────────
def pct(n: int, d: int) -> str:
    return f"{n/d*100:.1f}%" if d else "—"

def bar(n: int, d: int, w: int = 20) -> str:
    if d == 0: return " " * w
    f = int(n / d * w)
    return G("█" * f) + D("░" * (w - f))

def fmt_td(s: float) -> str:
    d = int(s)
    h, rem = divmod(d, 3600)
    m, sec = divmod(rem, 60)
    if h > 24:
        days = h // 24
        h = h % 24
        return f"{days}d {h:02d}:{m:02d}:{sec:02d}"
    return f"{h:02d}:{m:02d}:{sec:02d}"

# ── Data loading ───────────────────────────────────────────────────────
def load_spec(path: str) -> Dict[str, Any]:
    with open(path) as f:
        spec = json.load(f)
    return {
        "settings": [s["id"] for s in spec["settings"]],
        "tasks": spec["tasks"],
        "repeats": spec["repeats_per_setting"],
    }

def scan_policy(policy_dir: Path, target_settings: List[str]) -> List[Dict[str, Any]]:
    records = []
    for ep in policy_dir.rglob("episode_*.json"):
        parts = ep.parts
        if len(parts) < 4:
            continue
        setting = parts[-3]
        if setting not in target_settings:
            continue
        try:
            d = json.loads(ep.read_text())
            d["_mtime"] = ep.stat().st_mtime
            d.setdefault("setting_id", setting)
            d.setdefault("task", parts[-2])
            records.append(d)
        except Exception:
            pass
    return records

# ── Stats ──────────────────────────────────────────────────────────────
def compute(records, n_tasks, repeats, target_settings):
    now = time.time()
    expected = len(target_settings) * n_tasks * repeats
    done = len(records)
    ok = sum(1 for r in records if r.get("success"))

    # effective (exclude expert_failed / expert_check_error)
    eff = [r for r in records if (r.get("error") or "") not in ("expert_failed",) and not (r.get("error") or "").startswith("expert_check_error")]
    eff_ok = sum(1 for r in eff if r.get("success"))

    # errors
    errors = Counter()
    for r in records:
        e = r.get("error") or ""
        if not e: continue
        tag = e.split("\n")[0][:80]
        errors[tag] += 1

    # per-setting
    per_setting = {}
    exp_per = n_tasks * repeats
    for sid in target_settings:
        recs = [r for r in records if r.get("setting_id") == sid]
        ok_s = sum(1 for r in recs if r.get("success"))
        per_setting[sid] = {"done": len(recs), "ok": ok_s, "expected": exp_per}

    # per-task (top/bottom)
    per_task = defaultdict(lambda: {"done": 0, "ok": 0})
    for r in records:
        t = r.get("task", "?")
        per_task[t]["done"] += 1
        if r.get("success"):
            per_task[t]["ok"] += 1

    # timing
    mtimes = [r["_mtime"] for r in records if "_mtime" in r]
    t_first = min(mtimes) if mtimes else None
    t_last = max(mtimes) if mtimes else None

    # active setting detection (latest write in last 30 min)
    active = []
    for sid in target_settings:
        recs = [r for r in records if r.get("setting_id") == sid]
        if not recs: continue
        latest = max(r["_mtime"] for r in recs)
        d = per_setting[sid]["done"]
        e = per_setting[sid]["expected"]
        if d < e and (now - latest) < 1800:
            active.append((sid, d, e, latest))
    active.sort(key=lambda x: x[3], reverse=True)

    # speed & ETA
    elapsed = (now - t_first) if t_first else 0
    speed = done / elapsed if elapsed > 1 else 0
    remaining = expected - done
    eta_sec = remaining / speed if speed > 0 else 0

    return {
        "expected": expected, "done": done, "ok": ok,
        "eff": len(eff), "eff_ok": eff_ok,
        "errors": errors,
        "per_setting": per_setting,
        "per_task": dict(per_task),
        "active": active,
        "elapsed": elapsed, "speed": speed, "eta_sec": eta_sec,
        "t_first": t_first, "t_last": t_last,
    }

# ── Display ────────────────────────────────────────────────────────────
def print_policy(name: str, s: Dict[str, Any], target_settings: List[str]):
    done, exp = s["done"], s["expected"]
    sr = pct(s["ok"], done)
    eff_sr = pct(s["eff_ok"], s["eff"]) if s["eff"] else "—"
    speed_h = s["speed"] * 3600
    eta_dt = (datetime.now() + timedelta(seconds=s["eta_sec"])).strftime("%m-%d %H:%M") if s["eta_sec"] > 0 else "—"

    print(f"\n {B(CY(name))}")
    print(f"   {bar(done, exp, 30)} {done}/{exp} ({pct(done, exp)})")
    print(f"   SR: {G(sr) if s['ok'] else R(sr)} ({s['ok']}/{done})  |  Eff-SR: {eff_sr} ({s['eff_ok']}/{s['eff']})")
    print(f"   Elapsed: {fmt_td(s['elapsed'])}  |  Speed: {speed_h:.1f} ep/h  |  ETA: {fmt_td(s['eta_sec'])} ({eta_dt})")

    # Active settings
    if s["active"]:
        print(f"   {B('Currently running')}:")
        for sid, d, e, ts in s["active"]:
            ago = time.time() - ts
            sr_s = pct(s["per_setting"][sid]["ok"], d)
            print(f"     → {Y(sid):<40s} {d}/{e}  SR={sr_s}  (last {int(ago)}s ago)")

    # Per-setting table
    print(f"   {B('Per-setting')}:")
    print(f"     {'Setting':<36s} {'Progress':>12s}  {'%':>6s}  {'SR':>6s}")
    print(f"     {'─'*36} {'─'*12}  {'─'*6}  {'─'*6}")
    for sid in target_settings:
        ps = s["per_setting"].get(sid, {"done": 0, "ok": 0, "expected": 1000})
        d, e, o = ps["done"], ps["expected"], ps["ok"]
        colour = G if d >= e else Y if d > 0 else D
        prog = colour(f"{d}/{e}")
        print(f"     {sid:<36s} {prog:>22s}  {pct(d, e):>6s}  {pct(o, d):>6s}")

    # Top errors
    errs = [(k, v) for k, v in s["errors"].most_common() if v > 0]
    if errs:
        print(f"   {B('Errors')}:")
        for k, v in errs[:5]:
            print(f"     {v:>4d}  {k}")

    # Top/bottom 3 tasks
    task_items = sorted(s["per_task"].items(), key=lambda x: x[1]["ok"] / max(x[1]["done"], 1), reverse=True)
    if len(task_items) >= 6:
        print(f"   {B('Best tasks')}:")
        for t, v in task_items[:3]:
            print(f"     {t:<36s} {v['ok']:>3d}/{v['done']:<3d}  {pct(v['ok'], v['done'])}")
        print(f"   {B('Worst tasks')}:")
        for t, v in task_items[-3:]:
            print(f"     {t:<36s} {v['ok']:>3d}/{v['done']:<3d}  {pct(v['ok'], v['done'])}")


def print_comparison(all_stats: Dict[str, Dict[str, Any]]):
    if len(all_stats) < 2:
        return
    print(f"\n {B('Cross-Policy Comparison')}")
    print(f"   {'Policy':<14s} {'Progress':>12s}  {'SR':>7s} {'Eff-SR':>7s} {'Speed':>9s}  {'ETA':>18s}")
    print(f"   {'─'*14} {'─'*12}  {'─'*7} {'─'*7} {'─'*9}  {'─'*18}")
    for name, s in all_stats.items():
        d, e = s["done"], s["expected"]
        sr = pct(s["ok"], d)
        eff_sr = pct(s["eff_ok"], s["eff"]) if s["eff"] else "—"
        sp = f"{s['speed']*3600:.0f} ep/h"
        eta_dt = (datetime.now() + timedelta(seconds=s["eta_sec"])).strftime("%m-%d %H:%M") if s["eta_sec"] > 0 else "DONE"
        eta_s = fmt_td(s["eta_sec"])
        print(f"   {name:<14s} {d:>5d}/{e:<5d}  {sr:>7s} {eff_sr:>7s} {sp:>9s}  {eta_s} ({eta_dt})")


def render(spec, results_dir, target_settings):
    n_tasks = len(spec["tasks"])
    repeats = spec["repeats"]
    exp_per_policy = len(target_settings) * n_tasks * repeats

    base = Path(results_dir)
    policies = sorted([p.name for p in base.iterdir() if p.is_dir()]) if base.exists() else []

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f" {D(ts)}  |  {len(target_settings)} settings × {n_tasks} tasks × {repeats} repeats = {exp_per_policy} ep/policy")
    print(f" {D('Settings')}: {', '.join(target_settings)}")

    all_stats = {}
    for pname in policies:
        pdir = base / pname
        records = scan_policy(pdir, target_settings)
        s = compute(records, n_tasks, repeats, target_settings)
        all_stats[pname] = s
        print_policy(pname, s, target_settings)

    print_comparison(all_stats)
    print()


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Real-time benchmark monitor")
    parser.add_argument("--once", action="store_true", help="Single snapshot, no refresh")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--spec", default=SPEC_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--settings", default=None,
                        help="Comma-separated settings (default: 5 always_on)")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    target = args.settings.split(",") if args.settings else ALWAYS_ON_SETTINGS

    if args.once:
        render(spec, args.results_dir, target)
        return

    while True:
        os.system("clear")
        render(spec, args.results_dir, target)
        print(D(f" ─── refreshing in {args.interval}s (Ctrl+C to stop) ───"))
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            break


if __name__ == "__main__":
    main()
