#!/usr/bin/env python3
"""
Sample a (clean success, perturbed failure) paired demo with head-camera video.

Strategy:
  Phase A — Run play_once() with need_plan=True to get a guaranteed clean success.
            Save joint paths + record head-camera video.
  Phase B — Run play_once() with need_plan=False (replaying saved paths) in a new
            env with the same seed. Wrap robot.set_arm_joints to inject perturbation
            at the joint level + record video.

Usage:
    # From benchmark spec (auto-search for failure pair):
    CUDA_VISIBLE_DEVICES=0 python scripts/sample_paired_demo.py --gpu 0

    # From a specific perturbation JSON file:
    CUDA_VISIBLE_DEVICES=0 python scripts/sample_paired_demo.py --gpu 0 \
        --perturb-json /gemini/code/datasets/robotwin_dataset/perturbed/adjust_bottle/perturbation/194.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(description="Sample paired (clean, perturbed) demo")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--spec", type=str, default="benchmark/benchmark_spec.json")
    p.add_argument("--output", type=str, default="samples/")
    p.add_argument("--severity", type=str, default="high", choices=["lm", "high"])
    p.add_argument("--task", type=str, default="adjust_bottle")
    p.add_argument("--setting", type=str, default=None,
                   help="Specific setting ID (default: first matching)")
    p.add_argument("--perturb-json", type=str, default=None,
                   help="Path to a perturbation JSON file (overrides --spec/--setting)")
    p.add_argument("--use-rt", action="store_true",
                   help="Use ray-tracing shader (slower, prettier)")
    return p.parse_args()


def main():
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ---- Monkey-patch SAPIEN shader before any SAPIEN import ----
    if not args.use_rt:
        import sapien.render as _sr
        _orig = _sr.set_camera_shader_dir
        _sr.set_camera_shader_dir = lambda name: _orig("default")
        _sr.set_ray_tracing_samples_per_pixel = lambda *a, **kw: None
        _sr.set_ray_tracing_path_depth = lambda *a, **kw: None
        _sr.set_ray_tracing_denoiser = lambda *a, **kw: None

    _BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _BENCH_ROOT)
    sys.path.insert(0, os.path.join(_BENCH_ROOT, "description", "utils"))
    sys.path.insert(0, os.path.join(_BENCH_ROOT, "script"))

    import numpy as np
    from benchmark.eval_runner import (
        class_decorator, load_task_config, setup_env_args, get_camera_config,
    )
    from benchmark.perturbation_engine import (
        PerturbationConfig, PerturbationType, Severity, TimingMode,
        _numpy_to_list, create_perturbation, init_state,
    )

    # ---- Build perturbation config list ----
    # Each entry is (label, env_seed, PerturbationConfig)
    perturb_candidates = []

    if args.perturb_json:
        # Load from a single perturbation JSON file
        with open(args.perturb_json) as f:
            pj = json.load(f)

        # Map scenario field to TimingMode
        scenario_map = {"A": TimingMode.ALWAYS_ON, "B": TimingMode.ONSET_THEN_ALWAYS}
        timing = scenario_map.get(pj.get("scenario"), TimingMode.ALWAYS_ON)

        pcfg = PerturbationConfig(
            perturb_type=PerturbationType(pj["perturbation_type"]),
            severity=Severity(pj["severity"]),
            timing=timing,
            t_on_raw=pj["t_on_raw"],
            params=pj["params"],
            seed=pj.get("seed"),
        )
        env_seed = pj.get("source_episode", {}).get("seed")
        if env_seed is None:
            # Fallback: use spec seed
            with open(args.spec) as f:
                spec = json.load(f)
            first_sid = spec["settings"][0]["id"]
            env_seed = spec["env_seeds"][first_sid][args.task][0]

        task_name = pj.get("source_episode", {}).get("task", args.task)
        label = Path(args.perturb_json).stem
        perturb_candidates.append((label, env_seed, pcfg))

    else:
        # Load from benchmark spec
        with open(args.spec) as f:
            spec = json.load(f)

        task_name = args.task
        target_settings = [
            s for s in spec["settings"]
            if s["severity"] == args.severity and s["timing"] == "always_on"
        ]
        if args.setting:
            target_settings = [s for s in target_settings if s["id"] == args.setting]
        if not target_settings:
            print(f"ERROR: no settings for severity={args.severity}, timing=always_on")
            sys.exit(1)

        first_setting_id = target_settings[0]["id"]
        env_seed = spec["env_seeds"][first_setting_id][task_name][0]

        for setting in target_settings:
            sid = setting["id"]
            for ri, cfg_d in enumerate(spec["perturbation_configs"][sid][task_name]):
                pcfg = PerturbationConfig.from_dict(cfg_d)
                perturb_candidates.append((f"{sid}/repeat{ri}", env_seed, pcfg))

    # ---- Env args ----
    base_args = load_task_config("demo_clean")
    base_args = setup_env_args(base_args)
    base_args["eval_mode"] = True
    base_args["render_freq"] = 0
    base_args["task_name"] = task_name
    base_args["task_config"] = "demo_clean"
    base_args["policy_name"] = "expert_replay"
    base_args["ckpt_setting"] = "expert_replay"

    cam_cfg = get_camera_config(base_args["camera"]["head_camera_type"])
    video_size = f'{cam_cfg["w"]}x{cam_cfg["h"]}'

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Helpers ----
    def start_ffmpeg(output_path):
        return subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pixel_format", "rgb24",
             "-video_size", video_size, "-framerate", "10",
             "-i", "-", "-pix_fmt", "yuv420p",
             "-vcodec", "libx264", "-crf", "23",
             str(output_path)],
            stdin=subprocess.PIPE,
        )

    def make_recording_wrapper(task_env, ffmpeg_proc, frame_every=25):
        robot = task_env.robot
        original = robot.set_arm_joints
        counter = [0]

        def wrapped(target_position, target_velocity, arm_tag):
            original(target_position, target_velocity, arm_tag)
            counter[0] += 1
            if counter[0] % frame_every == 0:
                task_env._update_render()
                task_env.cameras.update_picture()
                rgb = task_env.cameras.get_rgb()["head_camera"]["rgb"]
                try:
                    ffmpeg_proc.stdin.write(rgb.tobytes())
                except BrokenPipeError:
                    pass

        robot.set_arm_joints = wrapped
        return original, counter

    def write_final_frames(task_env, ffmpeg_proc, n=10):
        task_env._update_render()
        task_env.cameras.update_picture()
        rgb = task_env.cameras.get_rgb()["head_camera"]["rgb"]
        for _ in range(n):
            try:
                ffmpeg_proc.stdin.write(rgb.tobytes())
            except BrokenPipeError:
                break

    # Use the seed from the first candidate (all candidates share the same seed)
    env_seed = perturb_candidates[0][1]

    # ==================================================================
    # Phase A: CLEAN expert run
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"Phase A: Clean expert  (task={task_name}, seed={env_seed})")
    print(f"{'='*60}")

    clean_video = out_dir / "clean.mp4"
    ENV_A = class_decorator(task_name)
    ENV_A.suc = 0
    ENV_A.test_num = 0
    env_args_a = copy.deepcopy(base_args)

    try:
        ENV_A.setup_demo(now_ep_num=0, seed=env_seed, is_test=True, **env_args_a)

        ffmpeg_a = start_ffmpeg(clean_video)
        orig_fn, cnt = make_recording_wrapper(ENV_A, ffmpeg_a)

        ENV_A.play_once()
        clean_ok = ENV_A.plan_success and ENV_A.check_success()

        saved_left_paths = deepcopy(ENV_A.left_joint_path)
        saved_right_paths = deepcopy(ENV_A.right_joint_path)

        write_final_frames(ENV_A, ffmpeg_a)
        ENV_A.robot.set_arm_joints = orig_fn
        ffmpeg_a.stdin.close()
        ffmpeg_a.wait()
        ENV_A.close_env(clear_cache=True)

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        try: ffmpeg_a.stdin.close(); ffmpeg_a.wait()
        except Exception: pass
        try: ENV_A.close_env(clear_cache=True)
        except Exception: pass
        sys.exit(1)

    status = "\033[92mSUCCESS\033[0m" if clean_ok else "\033[91mFAIL\033[0m"
    print(f"  Result: {status}  ({cnt[0]} set_arm_joints calls, "
          f"{len(saved_left_paths)} left paths, {len(saved_right_paths)} right paths)")

    if not clean_ok:
        print("  Expert planner failed. Try a different task/seed.")
        sys.exit(1)

    # ==================================================================
    # Phase B: PERTURBED replay
    # ==================================================================
    found = False
    for label, _, pcfg in perturb_candidates:
        print(f"\n{'='*60}")
        print(f"Phase B: Perturbed  ({label})")
        print(f"  type={pcfg.perturb_type.value}  severity={pcfg.severity.value}  "
              f"timing={pcfg.timing.value}  t_on={pcfg.t_on_raw}")
        print(f"{'='*60}")

        pert_video = out_dir / "perturbed.mp4"
        ENV_B = class_decorator(task_name)
        ENV_B.suc = 0
        ENV_B.test_num = 0

        env_args_b = copy.deepcopy(base_args)
        env_args_b["need_plan"] = False
        env_args_b["left_joint_path"] = deepcopy(saved_left_paths)
        env_args_b["right_joint_path"] = deepcopy(saved_right_paths)

        try:
            ENV_B.setup_demo(now_ep_num=0, seed=env_seed, is_test=True,
                             **env_args_b)

            ffmpeg_b = start_ffmpeg(pert_video)

            robot = ENV_B.robot
            orig_set = robot.set_arm_joints
            frame_cnt = [0]

            left_js = robot.get_left_arm_jointState()
            right_js = robot.get_right_arm_jointState()
            q0 = np.array(left_js + right_js, dtype=np.float32)
            state = init_state(u_prev=q0, a_prev=q0)
            perturb = create_perturbation(pcfg, state)

            def perturbed_set_arm_joints(target_position, target_velocity, arm_tag):
                nonlocal perturb
                cur_left = np.array(robot.get_left_arm_jointState(), dtype=np.float32)
                cur_right = np.array(robot.get_right_arm_jointState(), dtype=np.float32)
                u_t = np.concatenate([cur_left, cur_right])

                target_arr = np.array(target_position, dtype=np.float32)
                if arm_tag == "left":
                    u_t[:6] = target_arr[:6]
                elif arm_tag == "right":
                    u_t[7:13] = target_arr[:6]

                a_t = perturb.apply(u_t)

                if arm_tag == "left":
                    perturbed_pos = a_t[:6].astype(np.float64)
                else:
                    perturbed_pos = a_t[7:13].astype(np.float64)

                orig_set(perturbed_pos, target_velocity, arm_tag)

                frame_cnt[0] += 1
                if frame_cnt[0] % 25 == 0:
                    ENV_B._update_render()
                    ENV_B.cameras.update_picture()
                    rgb = ENV_B.cameras.get_rgb()["head_camera"]["rgb"]
                    try:
                        ffmpeg_b.stdin.write(rgb.tobytes())
                    except BrokenPipeError:
                        pass

            robot.set_arm_joints = perturbed_set_arm_joints

            ENV_B.play_once()
            pert_ok = ENV_B.plan_success and ENV_B.check_success()
            pert_steps = frame_cnt[0]

            write_final_frames(ENV_B, ffmpeg_b)
            robot.set_arm_joints = orig_set
            ffmpeg_b.stdin.close()
            ffmpeg_b.wait()
            ENV_B.close_env(clear_cache=True)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            try: ffmpeg_b.stdin.close(); ffmpeg_b.wait()
            except Exception: pass
            try: ENV_B.close_env(clear_cache=True)
            except Exception: pass
            continue

        status_str = "\033[91mFAIL\033[0m" if not pert_ok else "\033[92mSUCCESS\033[0m"
        print(f"  Result: {status_str}  ({pert_steps} set_arm_joints calls)")

        if pert_ok:
            print("  -> Perturbation didn't cause failure, trying next...")
            if pert_video.exists():
                pert_video.unlink()
            continue

        # ---- Found the pair ----
        found = True
        summary = {
            "task": task_name,
            "label": label,
            "env_seed": env_seed,
            "clean": {"success": True, "video": str(clean_video)},
            "perturbed": {
                "success": False,
                "steps": pert_steps,
                "video": str(pert_video),
                "perturbation": pcfg.to_dict(),
            },
            "timestamp": datetime.now().isoformat(),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(_numpy_to_list(summary), f, indent=2)

        print(f"\n{'='*60}")
        print(f"FOUND paired demo!")
        print(f"  Clean:     {clean_video}")
        print(f"  Perturbed: {pert_video}")
        print(f"  Summary:   {out_dir / 'summary.json'}")
        print(f"{'='*60}")
        break

    if not found:
        if args.perturb_json:
            print(f"\nPerturbation from {args.perturb_json} did not cause failure.")
        else:
            print(f"\nNo perturbed failure found across all candidates.")
        sys.exit(1)


if __name__ == "__main__":
    main()
