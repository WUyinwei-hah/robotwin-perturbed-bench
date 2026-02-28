"""
Generate Verified Benchmark Spec

For each (setting, task), samples perturbation parameters and replays the clean
expert trajectory with perturbation in SAPIEN. If the task still succeeds
(perturbation ineffective), resamples up to --max-tries times. If max tries are
exhausted for a slot, it falls back to direct one-shot sampling for remaining
slots of that (setting, task). Every collected sample is labeled with
`caused_failure` to indicate whether perturbation caused task failure.

Designed for multi-GPU parallelism: each worker handles a partition of tasks,
writes a partial spec. A merge step combines all partials into the final spec.

Usage (single worker):
    python -m benchmark.generate_verified_spec \
        --spec benchmark/benchmark_spec.json \
        --dataset-root /gemini/code/datasets/robotwin_dataset/clean \
        --gpu 0 --worker-id 0 --num-workers 8 \
        --num-effective 20 --max-tries 100 --severity lm \
        --output verified_spec_parts/part_0.json

Merge all parts:
    python -m benchmark.generate_verified_spec --merge \
        --parts-dir verified_spec_parts \
        --spec benchmark/benchmark_spec.json \
        --output benchmark/benchmark_spec_verified.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from benchmark.perturbation_engine import (
    PerturbationConfig,
    PerturbationType,
    Severity,
    TimingMode,
    create_perturbation,
    init_state,
    sample_perturbation_params,
    _numpy_to_list,
)
from benchmark.spec_generator import (
    _class_decorator,
    _load_task_config,
    _setup_env_args,
    compute_t_on_raw,
)


def _replay_check(task_env, task_args, seed, traj, perturb_cfg):
    """Replay clean trajectory with perturbation, return True if task FAILS (effective)."""
    try:
        task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
        obs = task_env.get_obs()
        q_0 = np.array(obs["joint_action"]["vector"], dtype=np.float32)
        state = init_state(u_prev=q_0, a_prev=q_0)
        perturb = create_perturbation(perturb_cfg, state)

        T = traj.shape[0]
        for t in range(T):
            a_t = perturb.apply(traj[t].astype(np.float32))
            task_env.take_action(a_t.tolist(), action_type="qpos")

        success = bool(task_env.check_success())
        return not success  # effective = task failed
    except Exception as e:
        # Env error → treat as effective (don't waste retries)
        return True


def generate_for_worker(
    spec_path: str,
    dataset_root: str,
    task_config_name: str,
    worker_id: int,
    num_workers: int,
    num_effective: int,
    max_tries: int,
    severity_filter: Optional[str],
    output_path: str,
):
    with open(spec_path) as f:
        spec = json.load(f)

    all_tasks = spec["tasks"]
    settings = spec["settings"]
    master_seed = spec["master_seed"]

    if severity_filter:
        settings = [s for s in settings if s["severity"] == severity_filter]

    my_tasks = [t for i, t in enumerate(all_tasks) if i % num_workers == worker_id]
    print(f"[Worker {worker_id}] {len(my_tasks)} tasks, {len(settings)} settings, "
          f"{num_effective} effective per (setting,task), max_tries={max_tries}")

    base_args = _setup_env_args(_load_task_config(task_config_name))
    base_args["eval_mode"] = True
    base_args["render_freq"] = 0
    base_args["task_config"] = task_config_name
    base_args["policy_name"] = "gen_verified"
    base_args["ckpt_setting"] = "gen_verified"

    # Output structure: {setting_id: {task_name: [list of verified PerturbationConfig dicts]}}
    verified_configs = {}
    verified_seeds = {}
    stats = {"total_tries": 0, "effective": 0, "non_effective": 0, "exhausted": 0}

    total_jobs = len(settings) * len(my_tasks)
    done_jobs = 0
    start_time = time.time()

    for task_name in my_tasks:
        # Get seed (same across all settings)
        first_sid = settings[0]["id"]
        seed = spec["env_seeds"][first_sid][task_name][0]

        qpos_path = os.path.join(dataset_root, task_name, "qpos", f"{seed}.pt")
        if not os.path.exists(qpos_path):
            print(f"[Worker {worker_id}] WARNING: {qpos_path} not found, skip {task_name}")
            continue

        traj = torch.load(qpos_path, map_location="cpu", weights_only=True).numpy()
        print(f"\n[Worker {worker_id}] {task_name}: seed={seed}, traj={traj.shape}")

        task_env = _class_decorator(task_name)
        task_args = copy.deepcopy(base_args)
        task_args["task_name"] = task_name

        for setting in settings:
            setting_id = setting["id"]
            perturb_type = PerturbationType(setting["perturb_type"])
            severity = Severity(setting["severity"])
            timing = TimingMode(setting["timing"])

            if timing == TimingMode.ALWAYS_ON:
                t_on_raw = 0
            else:
                t_on_raw = compute_t_on_raw(task_name)

            # Deterministic RNG per (setting, task)
            rng_seed = hash((master_seed, setting_id, task_name)) & 0x7FFFFFFF
            rng = np.random.default_rng(rng_seed)

            collected = []
            total_attempts = 0
            direct_sample_mode = False

            for slot in range(num_effective):
                if direct_sample_mode:
                    total_attempts += 1
                    stats["total_tries"] += 1

                    params = sample_perturbation_params(perturb_type, severity, rng)
                    perturb_seed = int(rng.integers(0, 2**31))

                    cfg = PerturbationConfig(
                        perturb_type=perturb_type,
                        severity=severity,
                        timing=timing,
                        t_on_raw=t_on_raw,
                        params=params,
                        seed=perturb_seed,
                    )

                    effective = _replay_check(task_env, task_args, seed, traj, cfg)
                    cfg_dict = cfg.to_dict()
                    cfg_dict["caused_failure"] = bool(effective)
                    collected.append(cfg_dict)

                    if effective:
                        stats["effective"] += 1
                    else:
                        stats["non_effective"] += 1
                    continue

                found = False
                last_cfg_dict = None
                for attempt in range(max_tries):
                    total_attempts += 1
                    stats["total_tries"] += 1

                    params = sample_perturbation_params(perturb_type, severity, rng)
                    perturb_seed = int(rng.integers(0, 2**31))

                    cfg = PerturbationConfig(
                        perturb_type=perturb_type,
                        severity=severity,
                        timing=timing,
                        t_on_raw=t_on_raw,
                        params=params,
                        seed=perturb_seed,
                    )

                    effective = _replay_check(task_env, task_args, seed, traj, cfg)
                    cfg_dict = cfg.to_dict()
                    cfg_dict["caused_failure"] = bool(effective)
                    last_cfg_dict = cfg_dict

                    if effective:
                        collected.append(cfg_dict)
                        stats["effective"] += 1
                        found = True
                        break

                if not found:
                    # Exhausted max_tries: keep the last sampled config and switch
                    # to direct one-shot sampling for remaining slots.
                    collected.append(last_cfg_dict)
                    stats["exhausted"] += 1
                    if last_cfg_dict and last_cfg_dict.get("caused_failure", False):
                        stats["effective"] += 1
                    else:
                        stats["non_effective"] += 1
                    direct_sample_mode = True
                    print(f"    WARNING: {setting_id}/{task_name} slot {slot}: "
                          f"exhausted {max_tries} tries, using last sample; "
                          f"switching to direct sampling for remaining slots")

            # Store results
            if setting_id not in verified_configs:
                verified_configs[setting_id] = {}
                verified_seeds[setting_id] = {}
            verified_configs[setting_id][task_name] = collected
            verified_seeds[setting_id][task_name] = [seed] * num_effective

            fail_count = sum(1 for c in collected if c.get("caused_failure", False))
            avg_tries = total_attempts / num_effective if num_effective > 0 else 0

            done_jobs += 1
            elapsed = time.time() - start_time
            rate = done_jobs / elapsed if elapsed > 0 else 0
            eta = (total_jobs - done_jobs) / rate if rate > 0 else 0

            print(f"  [{done_jobs}/{total_jobs}] {setting_id}: "
                  f"{num_effective} collected, failures={fail_count}/{num_effective}, "
                  f"avg_tries={avg_tries:.1f}, "
                  f"ETA={eta/60:.0f}min")

        # Close env for this task
        try:
            task_env.close_env(clear_cache=True)
        except TypeError:
            try:
                task_env.close_env()
            except Exception:
                pass
        except Exception:
            pass

    # Save partial result
    output = {
        "worker_id": worker_id,
        "num_workers": num_workers,
        "severity_filter": severity_filter,
        "num_effective": num_effective,
        "max_tries": max_tries,
        "master_seed": master_seed,
        "tasks": my_tasks,
        "settings": settings,
        "stats": stats,
        "verified_configs": verified_configs,
        "verified_seeds": verified_seeds,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: _numpy_to_list(o))

    print(f"\n{'='*60}")
    print(f"[Worker {worker_id}] DONE")
    print(f"  Total tries: {stats['total_tries']}")
    print(f"  Effective: {stats['effective']}")
    print(f"  Exhausted (max_tries hit): {stats['exhausted']}")
    print(f"  Saved to: {output_path}")


def merge_parts(parts_dir: str, spec_path: str, output_path: str):
    """Merge worker partial results into a single verified benchmark spec."""
    parts_dir = Path(parts_dir)
    part_files = sorted(parts_dir.glob("part_*.json"))

    if not part_files:
        print(f"No part files found in {parts_dir}")
        return

    print(f"Merging {len(part_files)} part files...")

    # Load original spec as template
    with open(spec_path) as f:
        orig_spec = json.load(f)

    # Merge verified configs and seeds from all parts
    merged_configs = {}
    merged_seeds = {}
    total_stats = {"total_tries": 0, "effective": 0, "exhausted": 0}
    num_effective = None
    severity_filter = None
    merged_settings = None

    for pf in part_files:
        with open(pf) as f:
            part = json.load(f)

        if num_effective is None:
            num_effective = part["num_effective"]
            severity_filter = part.get("severity_filter")
            merged_settings = part["settings"]

        for k in total_stats:
            total_stats[k] += part["stats"].get(k, 0)

        for sid, task_data in part["verified_configs"].items():
            if sid not in merged_configs:
                merged_configs[sid] = {}
                merged_seeds[sid] = {}
            merged_configs[sid].update(task_data)
        for sid, task_data in part["verified_seeds"].items():
            if sid not in merged_seeds:
                merged_seeds[sid] = {}
            merged_seeds[sid].update(task_data)

    # Build new spec
    new_spec = {
        "master_seed": orig_spec["master_seed"],
        "repeats_per_setting": num_effective,
        "num_settings": len(merged_settings),
        "num_tasks": orig_spec["num_tasks"],
        "total_episodes_per_policy": len(merged_settings) * orig_spec["num_tasks"] * num_effective,
        "settings": merged_settings,
        "tasks": orig_spec["tasks"],
        "step_limits": orig_spec["step_limits"],
        "perturbation_configs": merged_configs,
        "env_seeds": merged_seeds,
        "verified": True,
        "verification_stats": total_stats,
    }

    # For settings NOT in the verified set, keep original configs
    orig_settings_ids = {s["id"] for s in orig_spec["settings"]}
    verified_sids = {s["id"] for s in merged_settings}
    unverified_sids = orig_settings_ids - verified_sids

    if unverified_sids:
        print(f"  {len(unverified_sids)} settings not verified, keeping original configs")
        # Include all settings in the new spec
        all_settings = merged_settings + [
            s for s in orig_spec["settings"] if s["id"] in unverified_sids
        ]
        new_spec["settings"] = all_settings
        new_spec["num_settings"] = len(all_settings)

        for sid in unverified_sids:
            if sid in orig_spec["perturbation_configs"]:
                # Trim to num_effective repeats
                merged_configs[sid] = {}
                merged_seeds[sid] = {}
                for task in orig_spec["tasks"]:
                    merged_configs[sid][task] = orig_spec["perturbation_configs"][sid][task][:num_effective]
                    merged_seeds[sid][task] = orig_spec["env_seeds"][sid][task][:num_effective]

        new_spec["perturbation_configs"] = merged_configs
        new_spec["env_seeds"] = merged_seeds
        new_spec["total_episodes_per_policy"] = (
            len(all_settings) * orig_spec["num_tasks"] * num_effective
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(new_spec, f, indent=2, default=lambda o: _numpy_to_list(o))

    print(f"\n{'='*60}")
    print(f"VERIFIED SPEC GENERATED")
    print(f"  Settings: {new_spec['num_settings']}")
    print(f"  Tasks: {new_spec['num_tasks']}")
    print(f"  Repeats: {new_spec['repeats_per_setting']}")
    print(f"  Total episodes: {new_spec['total_episodes_per_policy']}")
    print(f"  Verification stats: {total_stats}")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate verified benchmark spec")
    parser.add_argument("--merge", action="store_true", help="Merge worker parts")
    parser.add_argument("--parts-dir", type=str, default="verified_spec_parts")
    parser.add_argument("--spec", type=str, default="benchmark/benchmark_spec.json")
    parser.add_argument("--dataset-root", type=str,
                        default="/gemini/code/datasets/robotwin_dataset/clean")
    parser.add_argument("--task-config", type=str, default="demo_clean")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-effective", type=int, default=20)
    parser.add_argument("--max-tries", type=int, default=100)
    parser.add_argument("--severity", type=str, default="lm",
                        choices=["lm", "high", "all"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.merge:
        merge_parts(args.parts_dir, args.spec,
                    args.output or "benchmark/benchmark_spec_verified.json")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    severity_filter = args.severity if args.severity != "all" else None
    output_path = args.output or f"verified_spec_parts/part_{args.worker_id}.json"

    print(f"[Worker {args.worker_id}] GPU={args.gpu}, severity={args.severity}, "
          f"num_effective={args.num_effective}, max_tries={args.max_tries}")

    generate_for_worker(
        spec_path=args.spec,
        dataset_root=args.dataset_root,
        task_config_name=args.task_config,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        num_effective=args.num_effective,
        max_tries=args.max_tries,
        severity_filter=severity_filter,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
