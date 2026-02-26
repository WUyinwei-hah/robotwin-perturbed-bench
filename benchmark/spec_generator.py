"""
Benchmark Spec Generator

Generates a deterministic benchmark_spec.json with fixed perturbation configs
for all (setting, task, repeat) combinations.

Usage:
    python -m benchmark.spec_generator --output benchmark/benchmark_spec.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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
) -> Dict[str, Any]:
    """Generate the full benchmark specification.

    Returns a dict with:
      - settings: list of setting descriptors
      - tasks: list of task names
      - repeats_per_setting: number of repeats
      - perturbation_configs: nested dict [setting_id][task_name] -> list of configs
      - env_seeds: nested dict [setting_id][task_name] -> list of starting seeds
    """
    master_rng = np.random.default_rng(master_seed)

    settings = []
    perturbation_configs = {}
    env_seeds = {}

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

                        # Env seed for this episode
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
