#!/usr/bin/env python3
import os
import re
import sys
import time
import json
import glob
import argparse
from datetime import datetime, timedelta

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "Pi05")

SETTINGS_TARGET = {
    "scale_lm_always_on": 1000,
    "bias_lm_always_on": 1000,
    "coupling_lm_always_on": 1000,
    "iir_lm_always_on": 1000,
    "fir_lm_always_on": 1000,
}
GRAND_TOTAL = sum(SETTINGS_TARGET.values())

GPU_SETTINGS = {
    0: "bias_lm_always_on",
    1: "coupling→bias",
    2: "scale→bias",
    3: "iir→bias",
    4: "fir→bias",
    5: "bias_lm_always_on",
    6: "bias_lm_always_on",
    7: "bias_lm_always_on",
}

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

START_TIME = datetime(2026, 3, 2, 11, 55, 0)


def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def parse_log(gpu_id):
    log_path = os.path.join(LOG_DIR, f"pi05_lm20_g{gpu_id}.log")
    if not os.path.exists(log_path):
        return None

    result = {
        "gpu": gpu_id,
        "done": 0,
        "total": 0,
        "eta_min": None,
        "current_task": "",
        "succ_count": 0,
        "fail_count": 0,
        "skip_count": 0,
        "mtime": 0,
        "alive": False,
    }

    try:
        result["mtime"] = os.path.getmtime(log_path)
        result["alive"] = (time.time() - result["mtime"]) < 120

        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(8192, size)
            f.seek(-read_size, 2)
            tail = strip_ansi(f.read().decode("utf-8", errors="replace"))

        ep_pat = re.compile(
            r'\[(\d+)/(\d+)\]\s+(\S+)\s+(SUCC|FAIL)\s+steps=\d+(?:\s+rate=\S+)?\s+ETA=(\d+)min'
        )
        matches = list(ep_pat.finditer(tail))
        if matches:
            m = matches[-1]
            result["done"]         = int(m.group(1))
            result["total"]        = int(m.group(2))
            result["current_task"] = m.group(3)
            result["eta_min"]      = int(m.group(5))

        with open(log_path, "r", errors="replace") as f:
            content = strip_ansi(f.read())
        result["succ_count"] = len(re.findall(r'\bSUCC\b', content))
        result["fail_count"] = len(re.findall(r'\bFAIL\b', content))
        result["skip_count"] = len(re.findall(r'\bSKIP\b', content))

    except Exception as e:
        result["error"] = str(e)

    return result


def get_results_stats():
    stats = {}
    for setting, target in SETTINGS_TARGET.items():
        path = os.path.join(RESULTS_DIR, setting)
        if not os.path.isdir(path):
            stats[setting] = {"success": 0, "fail": 0, "error": 0, "total": 0, "target": target}
            continue
        eps = glob.glob(os.path.join(path, "**", "episode_*.json"), recursive=True)
        success = fail = error = 0
        for ep in eps:
            try:
                with open(ep) as f:
                    d = json.load(f)
                if d.get("success"):
                    success += 1
                elif d.get("error") and d["error"] not in ("expert_failed", "unstable_seed"):
                    error += 1
                else:
                    fail += 1
            except:
                pass
        stats[setting] = {"success": success, "fail": fail, "error": error,
                          "total": success + fail + error, "target": target}
    return stats


def fmt_eta(minutes):
    if minutes is None:
        return "N/A"
    h, m = divmod(int(minutes), 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def fmt_bar(done, total, width=25):
    if total == 0:
        return "[" + "?" * width + "]"
    pct = min(done / total, 1.0)
    filled = int(pct * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct * 100:5.1f}%"


def eta_color(minutes):
    if minutes is None:
        return DIM
    if minutes > 120:
        return RED
    if minutes > 60:
        return YELLOW
    return GREEN


def status_icon(alive):
    return f"{GREEN}●{RESET}" if alive else f"{RED}✖{RESET}"


def render():
    now = datetime.now()
    elapsed = now - START_TIME
    elapsed_h = elapsed.total_seconds() / 3600

    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  Pi0.5 Benchmark Monitor (always_on only)   {DIM}{now.strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"  Elapsed: {CYAN}{elapsed_h:.1f}h{RESET}  (restarted {START_TIME.strftime('%m/%d %H:%M')})")
    print()

    # GPU status
    print(f"  {BOLD}{'GPU':<4} {'St':>2} {'Done':>12} {'ETA':<8} {'S/F/K':>12} Assignment / Last Task{RESET}")
    print(f"  {'─' * 74}")

    max_eta = 0
    for gpu_id in range(8):
        data = parse_log(gpu_id)
        if data is None:
            print(f"  {gpu_id:<4} {RED}✖{RESET}  no log")
            continue

        done, total = data["done"], data["total"]
        eta = data["eta_min"] or 0
        max_eta = max(max_eta, eta)
        ec = eta_color(data["eta_min"])
        task = (data["current_task"] or "")
        assignment = GPU_SETTINGS.get(gpu_id, "?")
        sfk = f"{data['succ_count']}/{data['fail_count']}/{data['skip_count']}"

        if total > 0:
            progress = f"{done:>5}/{total:<5}"
        else:
            progress = "loading... "

        label = task if task else assignment
        print(f"  {gpu_id:<4} {status_icon(data['alive'])} {progress}  {ec}{fmt_eta(data['eta_min']):<8}{RESET} {sfk:>12}  {DIM}{label}{RESET}")

    # Results breakdown by setting
    print()
    print(f"  {BOLD}{'─' * 74}{RESET}")
    print(f"  {BOLD}Results by setting:{RESET}")
    print(f"  {'Setting':<30} {'Done':>6} {'Target':>6} {'Bar':<33} {'SR':>7} {'Err':>4}")
    print(f"  {'─' * 74}")

    stats = get_results_stats()
    total_done = 0
    total_s = 0
    total_t = 0

    for setting in ["scale_lm_always_on", "bias_lm_always_on", "coupling_lm_always_on",
                     "iir_lm_always_on", "fir_lm_always_on"]:
        v = stats.get(setting, {"success": 0, "fail": 0, "error": 0, "total": 0, "target": 1000})
        target = v["target"]
        done = v["total"]
        total_done += done
        total_s += v["success"]
        total_t += v["total"]
        sr = v["success"] / v["total"] * 100 if v["total"] else 0
        sc = GREEN if sr > 30 else (YELLOW if sr > 10 else RED)
        err_s = f"{RED}{v['error']}{RESET}" if v["error"] else f"{DIM}0{RESET}"
        short = setting.replace("_lm_always_on", "")
        bar = fmt_bar(done, target, 20)
        print(f"  {short:<30} {done:>6} {target:>6} {bar}  {sc}{sr:>6.1f}%{RESET} {err_s:>4}")

    overall_sr = total_s / total_t * 100 if total_t else 0
    print(f"  {'─' * 74}")
    osc = GREEN if overall_sr > 25 else YELLOW
    print(f"  {BOLD}{'TOTAL':<30} {total_done:>6} {GRAND_TOTAL:>6} {fmt_bar(total_done, GRAND_TOTAL, 20)}  {osc}{overall_sr:>6.1f}%{RESET}")

    # ETA
    print()
    if max_eta > 0:
        eta_finish = now + timedelta(minutes=max_eta)
        print(f"  {BOLD}ETA (bottleneck GPU): {YELLOW}{fmt_eta(max_eta)}{RESET}{BOLD} → ~{eta_finish.strftime('%m/%d %H:%M')}{RESET}")

    if elapsed_h > 0.05 and total_done > 0:
        eps_per_h = total_done / elapsed_h
        remaining = GRAND_TOTAL - total_done
        if eps_per_h > 0:
            eta_overall_h = remaining / eps_per_h
            eta_overall_finish = now + timedelta(hours=eta_overall_h)
            print(f"  Throughput: {CYAN}{eps_per_h:.1f} ep/h{RESET}  ETA (avg): {YELLOW}{eta_overall_h:.1f}h{RESET} → ~{eta_overall_finish.strftime('%m/%d %H:%M')}")

    print()
    print(f"  {BOLD}{'=' * 78}{RESET}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        if not args.once:
            os.system("clear")
        render()
        if args.once:
            break
        print(f"\n  {DIM}Refresh every {args.interval}s — Ctrl+C to exit{RESET}\n")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)


if __name__ == "__main__":
    main()
