"""
Core Evaluation Runner for the Perturbed RoboTwin Benchmark.

Runs policy evaluation under fixed perturbation configs from benchmark_spec.json.

Usage:
    python -m benchmark.eval_runner \
        --policy motus \
        --spec benchmark/benchmark_spec.json \
        --setting scale_lm_always_on \
        --tasks adjust_bottle,click_bell \
        --gpu 0
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Ensure robotwin-perturbed-bench root is importable
_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH_ROOT)
# Paths needed for env imports and instruction generation
sys.path.insert(0, os.path.join(_BENCH_ROOT, "description", "utils"))
sys.path.insert(0, os.path.join(_BENCH_ROOT, "script"))

from benchmark.perturbation_engine import PerturbationConfig, _numpy_to_list
from benchmark.perturbed_env import PerturbedEnvWrapper

logger = logging.getLogger(__name__)

# ============================================================
# Environment setup helpers (adapted from robotwin/script/eval_policy.py)
# ============================================================

def class_decorator(task_name):
    """Import and instantiate a task environment."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except Exception:
        raise SystemExit(f"No Task: {task_name}")


def load_task_config(task_config_name: str) -> Dict[str, Any]:
    """Load task configuration YAML."""
    config_path = f"./task_config/{task_config_name}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def setup_env_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Setup embodiment and camera configs for the environment."""
    from envs import CONFIGS_PATH

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(etype):
        robot_file = _embodiment_types[etype]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

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


def get_camera_config(camera_type: str) -> Dict[str, Any]:
    """Get camera resolution config."""
    from envs import CONFIGS_PATH
    camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")
    with open(camera_config_path, "r", encoding="utf-8") as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
    return configs[camera_type]


def generate_instruction(task_name: str, episode_info_list: list, test_num: int,
                         instruction_type: str = "seen") -> str:
    """Generate a language instruction for the task."""
    from generate_episode_instructions import generate_episode_descriptions, load_task_instructions
    results = generate_episode_descriptions(task_name, episode_info_list, test_num)
    if results and results[0].get(instruction_type):
        return np.random.choice(results[0][instruction_type])
    # Fallback: use the task's full_description when no templated instructions match
    task_data = load_task_instructions(task_name)
    return task_data.get("full_description", f"Complete the {task_name} task.")


# ============================================================
# Core evaluation loop
# ============================================================

def run_single_episode(
    task_name: str,
    task_env,
    env_args: Dict[str, Any],
    policy_adapter,
    seed: int,
    perturb_cfg: PerturbationConfig,
    video_size: Optional[str] = None,
    repeat_idx: int = 0,
    skip_expert_check: bool = False,
) -> Dict[str, Any]:
    """Run a single episode with perturbation.

    If *skip_expert_check* is True, Phase 1 (expert demo verification) is
    skipped entirely.  Use this when seeds are pre-verified or when the
    CuroboPlanner / IK solver is unavailable on the current machine.

    Returns:
        dict with keys: success, steps, seed, perturbation_log, error
    """
    from envs.utils.create_actor import UnStableError

    result = {
        "task": task_name,
        "seed": seed,
        "success": False,
        "steps": 0,
        "error": None,
        "perturbation_log": perturb_cfg.to_dict(),
    }

    episode_info = None

    if not skip_expert_check:
        # Phase 1: Expert check — verify seed is solvable
        render_freq = env_args["render_freq"]
        env_args["render_freq"] = 0

        try:
            task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **env_args)
            episode_info = task_env.play_once()
            task_env.close_env(clear_cache=True)
        except UnStableError:
            try:
                task_env.close_env(clear_cache=True)
            except Exception:
                pass
            result["error"] = "unstable_seed"
            env_args["render_freq"] = render_freq
            return result
        except Exception as e:
            try:
                task_env.close_env(clear_cache=True)
            except Exception:
                pass
            result["error"] = f"expert_check_error: {e}"
            env_args["render_freq"] = render_freq
            return result

        if not (task_env.plan_success and task_env.check_success()):
            result["error"] = "expert_failed"
            env_args["render_freq"] = render_freq
            return result

        env_args["render_freq"] = render_freq

    # Phase 2: Policy evaluation with perturbation
    try:
        task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **env_args)

        if episode_info is not None:
            episode_info_list = [episode_info["info"]]
            instruction = generate_instruction(task_name, episode_info_list, 1)
        else:
            # No expert run — generate instruction from task name only
            instruction = generate_instruction(task_name, [], 0)
        task_env.set_instruction(instruction=instruction)

        # Setup video recording
        ffmpeg_proc = None
        if task_env.eval_video_path is not None and video_size:
            ffmpeg_proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", video_size, "-framerate", "10",
                    "-i", "-", "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264", "-crf", "23",
                    f"{task_env.eval_video_path}/episode{repeat_idx}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            task_env._set_eval_video_ffmpeg(ffmpeg_proc)

        # Activate perturbation wrapper
        wrapper = PerturbedEnvWrapper(task_env, perturb_cfg)
        wrapper.activate()

        # Reset policy
        policy_adapter.reset(task_env, instruction)

        # Run policy loop
        succ = False
        while task_env.take_action_cnt < task_env.step_lim:
            observation = task_env.get_obs()
            policy_adapter.step(task_env, observation)
            if task_env.eval_success:
                succ = True
                break

        # Cleanup
        wrapper.deactivate()
        result["perturbation_log"] = wrapper.get_perturbation_log()

        if task_env.eval_video_path is not None:
            task_env._del_eval_video_ffmpeg()

        result["success"] = succ
        result["steps"] = task_env.take_action_cnt

        task_env.close_env(clear_cache=True)

    except Exception as e:
        result["error"] = f"eval_error: {e}\n{traceback.format_exc()}"
        try:
            task_env.close_env(clear_cache=True)
        except Exception:
            pass

    return result


def run_benchmark(
    policy_adapter,
    spec: Dict[str, Any],
    settings_filter: Optional[List[str]] = None,
    tasks_filter: Optional[List[str]] = None,
    output_dir: str = "results",
    task_config_name: str = "demo_clean",
    skip_expert_check: bool = False,
) -> None:
    """Run the full benchmark (or a subset).

    Args:
        policy_adapter: loaded PolicyAdapter instance
        spec: loaded benchmark_spec.json
        settings_filter: list of setting IDs to run (None = all)
        tasks_filter: list of task names to run (None = all)
        output_dir: root output directory
        task_config_name: task config to use
        skip_expert_check: skip Phase 1 expert demo verification per episode
    """
    policy_name = policy_adapter.name
    settings = spec["settings"]
    tasks = spec["tasks"]
    repeats = spec["repeats_per_setting"]
    step_limits = spec["step_limits"]

    if settings_filter:
        settings = [s for s in settings if s["id"] in settings_filter]
    if tasks_filter:
        tasks = [t for t in tasks if t in tasks_filter]

    # Load base task config
    base_args = load_task_config(task_config_name)
    base_args = setup_env_args(base_args)
    base_args["eval_mode"] = True
    base_args["render_freq"] = 0  # headless evaluation

    camera_config = get_camera_config(base_args["camera"]["head_camera_type"])
    video_size = f'{camera_config["w"]}x{camera_config["h"]}'

    total_settings = len(settings)
    total_tasks = len(tasks)
    total_episodes = total_settings * total_tasks * repeats

    print(f"\n{'='*60}")
    print(f"Perturbed RoboTwin Benchmark")
    print(f"{'='*60}")
    print(f"Policy:    {policy_name}")
    print(f"Settings:  {total_settings}")
    print(f"Tasks:     {total_tasks}")
    print(f"Repeats:   {repeats}")
    print(f"Total:     {total_episodes} episodes")
    print(f"Output:    {output_dir}")
    print(f"{'='*60}\n")

    episode_count = 0
    start_time = time.time()

    for si, setting in enumerate(settings):
        setting_id = setting["id"]
        setting_configs = spec["perturbation_configs"][setting_id]
        setting_seeds = spec["env_seeds"][setting_id]

        for ti, task_name in enumerate(tasks):
            task_configs = setting_configs[task_name]
            task_seeds = setting_seeds[task_name]

            # Deep-copy base_args so nested dicts are not shared
            env_args = copy.deepcopy(base_args)
            env_args["task_name"] = task_name
            env_args["task_config"] = task_config_name
            env_args["policy_name"] = policy_name
            env_args["ckpt_setting"] = policy_name  # used by _init_task_env_

            # Create task env
            TASK_ENV = class_decorator(task_name)
            TASK_ENV.suc = 0
            TASK_ENV.test_num = 0

            # Output directory for this setting + task
            result_dir = Path(output_dir) / policy_name / setting_id / task_name
            result_dir.mkdir(parents=True, exist_ok=True)

            # Check for existing results to allow resume
            existing = set()
            for f in result_dir.glob("episode_*.json"):
                try:
                    idx = int(f.stem.split("_")[1])
                    existing.add(idx)
                except ValueError:
                    pass

            task_successes = 0
            task_total = 0

            for ri in range(repeats):
                episode_count += 1
                result_file = result_dir / f"episode_{ri}.json"

                if ri in existing:
                    # Load existing result for progress tracking
                    with open(result_file) as f:
                        prev = json.load(f)
                    if prev.get("success"):
                        task_successes += 1
                    task_total += 1
                    print(f"[{episode_count}/{total_episodes}] SKIP {setting_id}/{task_name}/ep{ri} (exists)")
                    continue

                perturb_cfg = PerturbationConfig.from_dict(task_configs[ri])
                env_seed = task_seeds[ri]

                # Setup video path
                vid_dir = result_dir / "videos"
                vid_dir.mkdir(parents=True, exist_ok=True)
                env_args["eval_video_save_dir"] = str(vid_dir)

                # Deep-copy env_args per episode to avoid mutation
                ep_env_args = copy.deepcopy(env_args)

                result = run_single_episode(
                    task_name=task_name,
                    task_env=TASK_ENV,
                    env_args=ep_env_args,
                    policy_adapter=policy_adapter,
                    seed=env_seed,
                    perturb_cfg=perturb_cfg,
                    video_size=video_size,
                    repeat_idx=ri,
                    skip_expert_check=skip_expert_check,
                )

                # Add metadata
                result["setting_id"] = setting_id
                result["repeat_idx"] = ri
                result["policy"] = policy_name
                result["timestamp"] = datetime.now().isoformat()

                # Save result
                with open(result_file, "w") as f:
                    json.dump(_numpy_to_list(result), f, indent=2)

                if result["success"]:
                    task_successes += 1
                task_total += 1

                status = "\033[92mSUCC\033[0m" if result["success"] else "\033[91mFAIL\033[0m"
                if result.get("error"):
                    status = f"\033[93mERR: {result['error'][:40]}\033[0m"

                elapsed = time.time() - start_time
                eps = episode_count / elapsed if elapsed > 0 else 0
                eta = (total_episodes - episode_count) / eps if eps > 0 else 0

                print(
                    f"[{episode_count}/{total_episodes}] "
                    f"{setting_id}/{task_name}/ep{ri} "
                    f"{status} "
                    f"steps={result.get('steps', 0)} "
                    f"rate={task_successes}/{task_total} "
                    f"ETA={eta/60:.0f}min"
                )

            # Print task summary
            if task_total > 0:
                rate = task_successes / task_total * 100
                print(f"  >> {task_name}: {task_successes}/{task_total} = {rate:.1f}%")

    elapsed = time.time() - start_time
    print(f"\nBenchmark complete: {episode_count} episodes in {elapsed/60:.1f} min")


# ============================================================
# CLI
# ============================================================

def create_policy_adapter(policy_name: str, policy_config: Dict[str, Any]):
    """Factory: create and load a policy adapter."""
    if policy_name == "motus":
        from policies.motus_adapter import MotusAdapter
        adapter = MotusAdapter()
    elif policy_name == "pi05":
        from policies.pi05_adapter import Pi05Adapter
        adapter = Pi05Adapter()
    elif policy_name == "lingbot_va":
        from policies.lingbot_va_adapter import LingbotVAAdapter
        adapter = LingbotVAAdapter()
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    adapter.load(policy_config)
    return adapter


def main():
    parser = argparse.ArgumentParser(description="Perturbed RoboTwin Benchmark Runner")
    parser.add_argument("--policy", type=str, required=True,
                        choices=["motus", "pi05", "lingbot_va"],
                        help="Policy to evaluate")
    parser.add_argument("--policy-config", type=str, required=True,
                        help="Path to policy-specific YAML config")
    parser.add_argument("--spec", type=str, default="benchmark/benchmark_spec.json",
                        help="Path to benchmark spec JSON")
    parser.add_argument("--settings", type=str, default=None,
                        help="Comma-separated setting IDs to run (default: all)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names to run (default: all)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--task-config", type=str, default="demo_clean",
                        help="Task config name")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID")
    parser.add_argument("--skip-expert-check", action="store_true",
                        help="Skip Phase 1 expert demo verification. Use when "
                             "seeds are pre-verified or CuroboPlanner is unavailable.")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Verify SAPIEN renderer works
    from script.test_render import Sapien_TEST
    Sapien_TEST()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load benchmark spec
    with open(args.spec, "r") as f:
        spec = json.load(f)
    print(f"Loaded benchmark spec: {spec['num_settings']} settings, {spec['num_tasks']} tasks")

    # Load policy config
    with open(args.policy_config, "r") as f:
        policy_config = yaml.safe_load(f)

    # Create policy adapter
    print(f"Loading policy: {args.policy}")
    adapter = create_policy_adapter(args.policy, policy_config)
    print(f"Policy loaded: {adapter.name} (action_type={adapter.action_type})")

    # Parse filters
    settings_filter = args.settings.split(",") if args.settings else None
    tasks_filter = args.tasks.split(",") if args.tasks else None

    # Run benchmark
    run_benchmark(
        policy_adapter=adapter,
        spec=spec,
        settings_filter=settings_filter,
        tasks_filter=tasks_filter,
        output_dir=args.output,
        task_config_name=args.task_config,
        skip_expert_check=args.skip_expert_check,
    )


if __name__ == "__main__":
    main()
