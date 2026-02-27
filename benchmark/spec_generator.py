"""
Benchmark Spec Generator

Generates a deterministic benchmark_spec.json with fixed perturbation configs
for all (setting, task, repeat) combinations.

Usage:
    python -m benchmark.spec_generator --output benchmark/benchmark_spec.json
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from benchmark.perturbation_engine import (
    BENCHMARK_PERTURB_TYPES,
    PerturbationConfig,
    PerturbationType,
    Severity,
    TimingMode,
    _numpy_to_list,
    sample_perturbation_params,
)

# ============================================================
# Step limit lookup (from task_config/_eval_step_limit.yml)
# ============================================================

STEP_LIMITS = {
    "adjust_bottle": 400,
    "beat_block_hammer": 400,
    "blocks_ranking_rgb": 1200,
    "blocks_ranking_size": 1200,
    "click_alarmclock": 400,
    "click_bell": 400,
    "dump_bin_bigbin": 600,
    "grab_roller": 400,
    "handover_block": 800,
    "handover_mic": 600,
    "hanging_mug": 900,
    "lift_pot": 400,
    "move_can_pot": 400,
    "move_pillbottle_pad": 400,
    "move_playingcard_away": 400,
    "move_stapler_pad": 400,
    "open_laptop": 700,
    "open_microwave": 1500,
    "pick_diverse_bottles": 400,
    "pick_dual_bottles": 400,
    "place_a2b_left": 400,
    "place_a2b_right": 400,
    "place_bread_basket": 700,
    "place_bread_skillet": 500,
    "place_burger_fries": 500,
    "place_can_basket": 700,
    "place_cans_plasticbox": 800,
    "place_container_plate": 400,
    "place_dual_shoes": 600,
    "place_empty_cup": 500,
    "place_fan": 400,
    "place_mouse_pad": 400,
    "place_object_basket": 700,
    "place_object_scale": 400,
    "place_object_stand": 400,
    "place_phone_stand": 400,
    "place_shoe": 500,
    "press_stapler": 400,
    "put_bottles_dustbin": 1700,
    "put_object_cabinet": 700,
    "rotate_qrcode": 400,
    "scan_object": 500,
    "shake_bottle": 700,
    "shake_bottle_horizontally": 700,
    "stack_blocks_three": 1200,
    "stack_blocks_two": 800,
    "stack_bowls_three": 1200,
    "stack_bowls_two": 900,
    "stamp_seal": 400,
    "turn_switch": 400,
}


# Ensure benchmark root is importable when run via -m
_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)


def _class_decorator(task_name: str):
    """Import and instantiate a task environment."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def _load_task_config(task_config_name: str) -> Dict[str, Any]:
    """Load task configuration YAML."""
    config_path = os.path.join(_BENCH_ROOT, "task_config", f"{task_config_name}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _setup_env_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Setup embodiment and camera configs for the environment."""
    from envs import CONFIGS_PATH

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(etype):
        robot_file = embodiment_types[etype]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def _is_seed_stable_and_solvable(task_name: str, env_args: Dict[str, Any], seed: int) -> bool:
    """Return True iff seed is stable and expert can solve it."""
    from envs._base_task import UnStableError

    task_env = _class_decorator(task_name)
    try:
        task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **env_args)
        task_env.play_once()
        return bool(task_env.plan_success and task_env.check_success())
    except UnStableError:
        return False
    except Exception:
        return False
    finally:
        try:
            task_env.close_env(clear_cache=True)
        except TypeError:
            try:
                task_env.close_env()
            except Exception:
                pass
        except Exception:
            pass


def _generate_stable_seeds_by_task(
    tasks: List[str],
    repeats_per_setting: int,
    master_seed: int,
    task_config_name: str,
    max_seed_trials_per_task: int,
) -> Dict[str, List[int]]:
    """Generate stable+solvable env seeds once per task, reused across all settings.

    WARNING: This launches SAPIEN for every candidate seed. Due to GPU resource
    leaks in SAPIEN, this may fail when run serially across many tasks in a
    single process. Prefer `_load_stable_seeds_from_dataset` when a clean
    expert-demo dataset is available.
    """
    print("Prechecking stable seeds (expert-solvable) ...")
    base_args = _setup_env_args(_load_task_config(task_config_name))
    base_args["eval_mode"] = True
    base_args["render_freq"] = 0
    base_args["task_config"] = task_config_name
    base_args["policy_name"] = "seed_check"
    base_args["ckpt_setting"] = "seed_check"

    seed_rng = np.random.default_rng(master_seed + 2027)
    stable_seeds_by_task: Dict[str, List[int]] = {}

    for task_name in tasks:
        task_args = copy.deepcopy(base_args)
        task_args["task_name"] = task_name

        accepted: List[int] = []
        tried = 0
        seen = set()
        while len(accepted) < repeats_per_setting and tried < max_seed_trials_per_task:
            candidate = int(seed_rng.integers(10000, 100000))
            tried += 1
            if candidate in seen:
                continue
            seen.add(candidate)

            if _is_seed_stable_and_solvable(task_name, task_args, candidate):
                accepted.append(candidate)

        if len(accepted) < repeats_per_setting:
            raise RuntimeError(
                f"Failed to find {repeats_per_setting} stable seeds for task '{task_name}' "
                f"within {max_seed_trials_per_task} trials."
            )

        stable_seeds_by_task[task_name] = accepted
        print(f"  {task_name}: {len(accepted)}/{repeats_per_setting} stable seeds found (trials={tried})")

    return stable_seeds_by_task


def _verify_seeds_subprocess(
    task_name: str,
    candidates: List[int],
    needed: int,
    task_config_name: str,
    strict: bool = True,
) -> List[int]:
    """Verify seeds for ONE task in a subprocess (avoids SAPIEN GPU leaks).

    If *strict* is True, requires both plan_success and check_success.
    If *strict* is False, only requires no UnStableError (env stability).
    Returns up to *needed* verified seeds from *candidates*.
    """
    import subprocess as _sp
    import textwrap

    script = textwrap.dedent(f"""\
        import sys, copy, json
        sys.path.insert(0, {repr(_BENCH_ROOT)})
        from benchmark.spec_generator import (
            _class_decorator, _load_task_config, _setup_env_args,
        )
        from envs._base_task import UnStableError
        import copy

        base_args = _setup_env_args(_load_task_config({repr(task_config_name)}))
        base_args["eval_mode"] = True
        base_args["render_freq"] = 0
        base_args["task_config"] = {repr(task_config_name)}
        base_args["policy_name"] = "seed_check"
        base_args["ckpt_setting"] = "seed_check"

        task_args = copy.deepcopy(base_args)
        task_args["task_name"] = {repr(task_name)}

        candidates = {candidates!r}
        needed = {needed}
        strict = {strict!r}
        accepted = []
        for seed in candidates:
            task_env = _class_decorator({repr(task_name)})
            ok = False
            try:
                task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
                task_env.play_once()
                if strict:
                    ok = bool(task_env.plan_success and task_env.check_success())
                else:
                    ok = True  # env didn't crash → stable
            except UnStableError:
                ok = False
            except Exception:
                ok = False
            finally:
                try: task_env.close_env(clear_cache=True)
                except:
                    try: task_env.close_env()
                    except: pass
            if ok:
                accepted.append(seed)
                if len(accepted) >= needed:
                    break

        print("SEEDS_RESULT:" + json.dumps(accepted))
    """)

    result = _sp.run(
        [sys.executable, "-u", "-c", script],
        capture_output=True,
        text=True,
        cwd=_BENCH_ROOT,
        timeout=7200,
    )

    for line in result.stdout.splitlines():
        if line.startswith("SEEDS_RESULT:"):
            return json.loads(line[len("SEEDS_RESULT:"):])

    # Debug: show stderr tail on failure
    err_tail = (result.stderr or "").strip().splitlines()[-5:]
    print(f"  [subprocess stderr tail for {task_name}]:")
    for el in err_tail:
        print(f"    {el}")
    return []


def _load_stable_seeds_from_dataset(
    tasks: List[str],
    master_seed: int,
    dataset_root: str,
    task_config_name: str = "demo_clean",
    verify: bool = True,
) -> Dict[str, int]:
    """Load ONE stable seed per task from an existing clean expert-demo dataset.

    Each episode file in ``<dataset_root>/<task>/qpos/<N>.pt`` indicates that
    seed *N* was successfully used to collect an expert demo, so it is
    *likely* stable and solvable.

    If *verify* is True (default), candidate seeds are validated via a
    SAPIEN expert-demo run in a **subprocess** (one per task, so GPU
    resources are released between tasks). Strict check first
    (plan_success AND check_success), with relaxed fallback (no crash).

    Returns ``{task_name: seed}``.
    """
    print(f"Loading stable seeds from dataset: {dataset_root}")
    print(f"  SAPIEN verification: {'ON' if verify else 'OFF'}")
    seed_rng = np.random.default_rng(master_seed + 2027)
    stable_seed_by_task: Dict[str, int] = {}

    for task_name in tasks:
        qpos_dir = os.path.join(dataset_root, task_name, "qpos")
        if not os.path.isdir(qpos_dir):
            raise FileNotFoundError(
                f"No qpos directory for task '{task_name}' at {qpos_dir}"
            )

        available = sorted(
            int(f.replace(".pt", ""))
            for f in os.listdir(qpos_dir)
            if f.endswith(".pt")
        )

        if not available:
            raise RuntimeError(f"Task '{task_name}' has no episodes in dataset.")

        # Deterministic shuffle to pick candidates
        shuffled = seed_rng.permutation(available).tolist()

        if not verify:
            stable_seed_by_task[task_name] = shuffled[0]
            print(f"  {task_name}: seed={shuffled[0]} (unverified, from {len(available)})")
            continue

        # Pass 1: strict (plan_success AND check_success)
        accepted = _verify_seeds_subprocess(
            task_name=task_name,
            candidates=shuffled,
            needed=1,
            task_config_name=task_config_name,
            strict=True,
        )

        if accepted:
            stable_seed_by_task[task_name] = accepted[0]
            print(f"  {task_name}: seed={accepted[0]} (strict, from {len(available)})")
            continue

        # Pass 2: relaxed (no crash)
        accepted = _verify_seeds_subprocess(
            task_name=task_name,
            candidates=shuffled,
            needed=1,
            task_config_name=task_config_name,
            strict=False,
        )

        if accepted:
            stable_seed_by_task[task_name] = accepted[0]
            print(f"  {task_name}: seed={accepted[0]} (relaxed, from {len(available)})")
        else:
            raise RuntimeError(
                f"Task '{task_name}': no stable seed found from "
                f"{len(available)} dataset candidates."
            )

    return stable_seed_by_task


def compute_t_on_raw(task_name: str) -> int:
    """Compute t_on_raw for onset_then_always timing.

    Rule: 16% of step_limit, clipped to [32, 192].
    """
    step_lim = STEP_LIMITS.get(task_name, 400)
    t_on = int(0.16 * step_lim)
    return max(32, min(192, t_on))


def make_setting_id(perturb_type: PerturbationType, severity: Severity, timing: TimingMode) -> str:
    """Create a human-readable setting ID."""
    type_short = {
        PerturbationType.SCALE: "scale",
        PerturbationType.COUPLING: "coupling",
        PerturbationType.LOWPASS_IIR: "iir",
        PerturbationType.LOWPASS_FIR: "fir",
        PerturbationType.BIAS: "bias",
    }
    return f"{type_short[perturb_type]}_{severity.value}_{timing.value}"


def generate_benchmark_spec(
    tasks: List[str],
    repeats_per_setting: int = 5,
    master_seed: int = 42,
    ensure_stable_seeds: bool = True,
    task_config_name: str = "demo_clean",
    max_seed_trials_per_task: int = 400,
    dataset_root: str | None = None,
    verify_seeds: bool = True,
) -> Dict[str, Any]:
    """Generate the full benchmark specification.

    Returns a dict with:
      - settings: list of setting descriptors
      - tasks: list of task names
      - repeats_per_setting: number of repeats
      - perturbation_configs: nested dict [setting_id][task_name] -> list of configs
      - env_seeds: nested dict [setting_id][task_name] -> list of starting seeds

    If *dataset_root* is provided, stable seeds are read from the clean dataset.
    If *verify_seeds* is True (default), each seed is validated via SAPIEN in a
    subprocess. Otherwise seeds are used as-is (fast but unverified).
    """
    master_rng = np.random.default_rng(master_seed)

    settings = []
    perturbation_configs = {}
    env_seeds = {}

    # Maps task_name -> single env seed
    stable_seed_by_task: Dict[str, int] = {}
    if ensure_stable_seeds:
        if dataset_root is not None:
            stable_seed_by_task = _load_stable_seeds_from_dataset(
                tasks=tasks,
                master_seed=master_seed,
                dataset_root=dataset_root,
                task_config_name=task_config_name,
                verify=verify_seeds,
            )
        else:
            # Legacy path: pick first seed from the old multi-seed method
            multi = _generate_stable_seeds_by_task(
                tasks=tasks,
                repeats_per_setting=1,
                master_seed=master_seed,
                task_config_name=task_config_name,
                max_seed_trials_per_task=max_seed_trials_per_task,
            )
            stable_seed_by_task = {t: seeds[0] for t, seeds in multi.items()}

    for perturb_type in BENCHMARK_PERTURB_TYPES:
        for severity in Severity:
            for timing in TimingMode:
                setting_id = make_setting_id(perturb_type, severity, timing)

                settings.append({
                    "id": setting_id,
                    "perturb_type": perturb_type.value,
                    "severity": severity.value,
                    "timing": timing.value,
                })

                perturbation_configs[setting_id] = {}
                env_seeds[setting_id] = {}

                for task_name in tasks:
                    task_configs = []
                    task_seeds = []

                    # Deterministic seed per (setting, task)
                    setting_task_seed = int(master_rng.integers(0, 2**31))
                    task_rng = np.random.default_rng(setting_task_seed)

                    for repeat_idx in range(repeats_per_setting):
                        # Deterministic perturbation params
                        params = sample_perturbation_params(perturb_type, severity, task_rng)
                        perturb_seed = int(task_rng.integers(0, 2**31))

                        if timing == TimingMode.ALWAYS_ON:
                            t_on_raw = 0
                        else:
                            t_on_raw = compute_t_on_raw(task_name)

                        cfg = PerturbationConfig(
                            perturb_type=perturb_type,
                            severity=severity,
                            timing=timing,
                            t_on_raw=t_on_raw,
                            params=params,
                            seed=perturb_seed,
                        )
                        task_configs.append(cfg.to_dict())

                        # Env seed: same seed for all repeats of this task
                        if ensure_stable_seeds:
                            env_seed = stable_seed_by_task[task_name]
                        else:
                            env_seed = int(task_rng.integers(10000, 100000))
                        task_seeds.append(env_seed)

                    perturbation_configs[setting_id][task_name] = task_configs
                    env_seeds[setting_id][task_name] = task_seeds

    spec = {
        "master_seed": master_seed,
        "repeats_per_setting": repeats_per_setting,
        "num_settings": len(settings),
        "num_tasks": len(tasks),
        "total_episodes_per_policy": len(settings) * len(tasks) * repeats_per_setting,
        "settings": settings,
        "tasks": tasks,
        "step_limits": {t: STEP_LIMITS.get(t, 400) for t in tasks},
        "perturbation_configs": perturbation_configs,
        "env_seeds": env_seeds,
    }

    return spec


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark specification")
    parser.add_argument("--output", type=str, default="benchmark/benchmark_spec.json")
    parser.add_argument("--tasks-file", type=str, default="tasks_all.txt")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--task-config", type=str, default="demo_clean")
    parser.add_argument("--max-seed-trials-per-task", type=int, default=400)
    parser.add_argument(
        "--no-stable-seed-check",
        action="store_true",
        help="Disable stability/expert precheck for env seeds (not recommended).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to clean expert-demo dataset to read stable seeds from "
             "(e.g. /path/to/robotwin_dataset/clean).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SAPIEN verification of dataset seeds (fast but unverified).",
    )
    args = parser.parse_args()

    # Load tasks
    tasks_path = Path(args.tasks_file)
    if not tasks_path.exists():
        print(f"Error: tasks file not found: {tasks_path}")
        sys.exit(1)

    tasks = [line.strip() for line in tasks_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(tasks)} tasks from {tasks_path}")

    # Generate spec
    spec = generate_benchmark_spec(
        tasks=tasks,
        repeats_per_setting=args.repeats,
        master_seed=args.master_seed,
        ensure_stable_seeds=not args.no_stable_seed_check,
        task_config_name=args.task_config,
        max_seed_trials_per_task=args.max_seed_trials_per_task,
        dataset_root=args.dataset_root,
        verify_seeds=not args.no_verify,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(spec, f, indent=2, default=lambda o: _numpy_to_list(o))

    print(f"Generated benchmark spec:")
    print(f"  Settings: {spec['num_settings']}")
    print(f"  Tasks: {spec['num_tasks']}")
    print(f"  Repeats per setting: {spec['repeats_per_setting']}")
    print(f"  Total episodes per policy: {spec['total_episodes_per_policy']}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
