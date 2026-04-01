"""
LingbotVA policy adapter for the perturbed benchmark.

Wraps the WebsocketClientPolicy (self-contained in policies/lingbot_va/).
Action space: 16D end-effector (ee).

The LingbotVA server must be launched separately before running the benchmark.
This adapter only handles the client-side websocket communication.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Optional
import imageio.v2 as iio
import pyarrow.parquet as pq

from policies.base_adapter import PolicyAdapter
from policies.lingbot_va.websocket_client_policy import WebsocketClientPolicy


class LingbotVAAdapter(PolicyAdapter):
    """Adapter for LingbotVA policy (EE-based, client-server websocket)."""

    def __init__(self):
        self.model = None  # WebsocketClientPolicy
        self._host = "127.0.0.1"
        self._port = 8000
        self._video_guidance_scale = 5.0
        self._action_guidance_scale = 5.0
        self._save_visualization = False
        self._prompt = None
        self._first = True
        self._first_obs = None
        self._inint_eef_pose = None
        self._perturb_override_mode = "none"
        self._episode_perturb_raw_override = None
        self._episode_seed = None
        self._episode_task_name = None
        self._chunk_idx = 0
        self._gt_warmstart_chunks = 0
        self._gt_warmstart_dataset_root = ""
        self._gt_warmstart_frame_chunk_size = 2
        self._gt_warmstart_action_per_frame = 16
        self._gt_seed_to_episode_index = None
        self._gt_warmstart_chunk_actions = []
        self._gt_cache_teacher_force = False
        self._gt_cache_chunk_actions = []
        self._gt_cache_chunk_obs = []
        self._tracked_actor_attr = None
        self._tracked_actor_name = None

        # Lazy-loaded modules
        self._R = None  # scipy Rotation

    def load(self, config: Dict[str, Any]) -> None:
        """Load LingbotVA client.

        Expected config keys:
            robotwin_root: path to robotwin repo (for env imports)
            port: websocket port (default 8000)
            video_guidance_scale: (default 5.0)
            action_guidance_scale: (default 5.0)
            save_visualization: (default False)
        """
        robotwin_root = config.get("robotwin_root", "")
        if robotwin_root and robotwin_root not in sys.path:
            sys.path.insert(0, robotwin_root)

        from scipy.spatial.transform import Rotation as R
        self._R = R

        self._port = config.get("port", 8000)
        self._host = config.get("host", "127.0.0.1")
        self._video_guidance_scale = config.get("video_guidance_scale", 5.0)
        self._action_guidance_scale = config.get("action_guidance_scale", 5.0)
        self._save_visualization = config.get("save_visualization", False)
        self._perturb_override_mode = str(config.get("perturb_override_mode", "none")).lower()
        self._gt_warmstart_chunks = int(config.get("gt_warmstart_chunks", 0) or 0)
        self._gt_warmstart_dataset_root = str(config.get("gt_warmstart_dataset_root", "") or "")
        self._gt_warmstart_frame_chunk_size = int(config.get("gt_warmstart_frame_chunk_size", 2) or 2)
        self._gt_warmstart_action_per_frame = int(config.get("gt_warmstart_action_per_frame", 16) or 16)
        self._gt_cache_teacher_force = bool(config.get("gt_cache_teacher_force", False))

        # Bypass proxy for local websocket server.
        local_hosts = {"127.0.0.1", "localhost", "0.0.0.0", self._host}
        existing_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        merged_no_proxy = [h.strip() for h in existing_no_proxy.split(",") if h.strip()]
        for host in local_hosts:
            if host not in merged_no_proxy:
                merged_no_proxy.append(host)
        os.environ["NO_PROXY"] = ",".join(merged_no_proxy)
        os.environ["no_proxy"] = os.environ["NO_PROXY"]

        # Create websocket client (self-contained, no external repo needed)
        self.model = WebsocketClientPolicy(host=self._host, port=self._port)

    def reset(self, task_env, instruction: str) -> None:
        """Reset for a new episode: send reset command to server."""
        self._prompt = instruction
        self._first = True
        self._first_obs = None
        self._action_log = []
        self._executed_action_count = 0
        self._chunk_idx = 0
        self._gt_warmstart_chunk_actions = []
        self._gt_cache_chunk_actions = []
        self._gt_cache_chunk_obs = []
        self._tracked_actor_attr = None
        self._tracked_actor_name = None

        # Send reset to server
        self.model.infer(dict(
            reset=True,
            prompt=self._prompt,
            save_visualization=self._save_visualization,
        ))

        # Capture initial EE pose
        initial_obs = task_env.get_obs()
        self._inint_eef_pose = (
            initial_obs['endpose']['left_endpose']
            + [initial_obs['endpose']['left_gripper']]
            + initial_obs['endpose']['right_endpose']
            + [initial_obs['endpose']['right_gripper']]
        )
        self._inint_eef_pose = np.array(self._inint_eef_pose, dtype=np.float64)

        # Format initial observation for server
        initial_formatted = self._format_obs(initial_obs)
        self._first_obs = initial_formatted
        self._prepare_gt_warmstart_actions()
        self._prepare_gt_cache_teacher_force()
        self._resolve_tracked_actor(task_env)

    def set_episode_context(
        self,
        *,
        perturbation_log: Optional[Dict[str, Any]] = None,
        setting_id: Optional[str] = None,
        repeat_idx: Optional[int] = None,
        seed: Optional[int] = None,
        task_name: Optional[str] = None,
    ) -> None:
        self._episode_seed = int(seed) if seed is not None else None
        self._episode_task_name = task_name
        self._episode_perturb_raw_override = None
        if self._perturb_override_mode != "gt_raw":
            if not perturbation_log:
                return
        if not perturbation_log:
            return
        config = perturbation_log.get("config") or perturbation_log
        params = config.get("params") or {}
        scale = params.get("S")
        if scale is None:
            return
        self._episode_perturb_raw_override = np.asarray(scale, dtype=np.float32)

    def _load_gt_seed_to_episode_index(self) -> Dict[int, int]:
        if self._gt_seed_to_episode_index is not None:
            return self._gt_seed_to_episode_index
        if not self._gt_warmstart_dataset_root:
            self._gt_seed_to_episode_index = {}
            return self._gt_seed_to_episode_index
        meta_path = Path(self._gt_warmstart_dataset_root) / "meta" / "episodes.jsonl"
        mapping: Dict[int, int] = {}
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                source_seed = row.get("source_episode", {}).get("seed")
                if source_seed is None:
                    continue
                mapping[int(source_seed)] = int(row["episode_index"])
        self._gt_seed_to_episode_index = mapping
        return mapping

    def _load_gt_episode_rollout(self, *, load_video: bool) -> Optional[Dict[str, Any]]:
        if not self._gt_warmstart_dataset_root or self._episode_seed is None:
            return None
        seed_to_ep = self._load_gt_seed_to_episode_index()
        if self._episode_seed not in seed_to_ep:
            return None

        episode_index = seed_to_ep[self._episode_seed]
        root = Path(self._gt_warmstart_dataset_root)
        parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "observation.joint_state", "action"],
        )
        data = table.to_pydict()
        states = np.asarray(data["observation.state"], dtype=np.float32)
        joint_states = np.asarray(data["observation.joint_state"], dtype=np.float32)
        abs_actions = np.asarray(data["action"], dtype=np.float32)

        video_frames = None
        if load_video:
            video_frames = {}
            for cam_key in (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ):
                video_path = root / "videos" / "chunk-000" / cam_key / f"episode_{episode_index:06d}.mp4"
                reader = iio.get_reader(str(video_path))
                frames = [frame for frame in reader]
                reader.close()
                video_frames[cam_key] = frames

        return {
            "episode_index": int(episode_index),
            "states": states,
            "joint_states": joint_states,
            "abs_actions": abs_actions,
            "video_frames": video_frames,
        }

    def _prepare_gt_warmstart_actions(self) -> None:
        self._gt_warmstart_chunk_actions = []
        if self._gt_warmstart_chunks <= 0:
            return
        rollout = self._load_gt_episode_rollout(load_video=False)
        if rollout is None:
            return
        rel_actions = self._absolute_actions_to_relative(rollout["abs_actions"])
        self._gt_warmstart_chunk_actions = self._chunk_relative_actions(
            rel_actions,
            num_chunks=self._gt_warmstart_chunks,
        )

    def _prepare_gt_cache_teacher_force(self) -> None:
        self._gt_cache_chunk_actions = []
        self._gt_cache_chunk_obs = []
        if not self._gt_cache_teacher_force:
            return
        rollout = self._load_gt_episode_rollout(load_video=True)
        if rollout is None:
            return
        rel_actions = self._absolute_actions_to_relative(rollout["abs_actions"])
        self._gt_cache_chunk_actions = self._chunk_relative_actions(rel_actions, num_chunks=None)
        self._gt_cache_chunk_obs = self._build_gt_cache_obs_chunks(
            states=rollout["states"],
            joint_states=rollout["joint_states"],
            video_frames=rollout["video_frames"],
            num_chunks=len(self._gt_cache_chunk_actions),
        )

    def _absolute_actions_to_relative(self, abs_actions: np.ndarray) -> np.ndarray:
        if self._inint_eef_pose is None:
            raise RuntimeError("Initial EEF pose is required before building GT warm-start actions")
        init_pose = self._inint_eef_pose.astype(np.float32)
        left_rel = self._absolute_pose_seq_to_relative(abs_actions[:, :7], init_pose[:7])
        right_rel = self._absolute_pose_seq_to_relative(abs_actions[:, 8:15], init_pose[8:15])
        return np.concatenate(
            [left_rel, abs_actions[:, 7:8], right_rel, abs_actions[:, 15:16]],
            axis=1,
        ).astype(np.float32)

    def _absolute_pose_seq_to_relative(self, pose_seq: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
        R = self._R
        trans = pose_seq[:, :3] - init_pose[None, :3]
        rot = R.from_quat(pose_seq[:, 3:7])
        init_rot = R.from_quat(np.tile(init_pose[3:7][None, :], (pose_seq.shape[0], 1)))
        rel_rot = (init_rot.inv() * rot).as_quat().astype(np.float32)
        return np.concatenate([trans.astype(np.float32), rel_rot], axis=1)

    def _chunk_relative_actions(self, rel_actions: np.ndarray, num_chunks: Optional[int]) -> List[np.ndarray]:
        chunks: List[np.ndarray] = []
        cursor = 0
        frame_chunk_size = self._gt_warmstart_frame_chunk_size
        action_per_frame = self._gt_warmstart_action_per_frame
        zero_frame = np.zeros((action_per_frame, rel_actions.shape[1]), dtype=np.float32)

        chunk_idx = 0
        while True:
            if num_chunks is not None and chunk_idx >= num_chunks:
                break
            if chunk_idx == 0:
                first_frame = zero_frame
                second_frame = rel_actions[cursor:cursor + action_per_frame]
                if second_frame.shape[0] < action_per_frame:
                    break
                cursor += action_per_frame
                frame_actions = np.stack([first_frame, second_frame], axis=0)
            else:
                needed = frame_chunk_size * action_per_frame
                chunk_actions = rel_actions[cursor:cursor + needed]
                if chunk_actions.shape[0] < needed:
                    break
                cursor += needed
                frame_actions = chunk_actions.reshape(frame_chunk_size, action_per_frame, -1)
            chunks.append(np.transpose(frame_actions, (2, 0, 1)).astype(np.float32))
            chunk_idx += 1
        return chunks

    def _build_gt_cache_obs_chunks(
        self,
        *,
        states: np.ndarray,
        joint_states: np.ndarray,
        video_frames: Dict[str, List[np.ndarray]],
        num_chunks: int,
    ) -> List[List[Dict[str, Any]]]:
        chunks: List[List[Dict[str, Any]]] = []
        cursor = 0
        frame_chunk_size = self._gt_warmstart_frame_chunk_size
        action_per_frame = self._gt_warmstart_action_per_frame
        obs_update_stride = max(action_per_frame // 4, 1)
        max_idx = int(states.shape[0] - 1)

        for chunk_idx in range(num_chunks):
            executed_action_num = action_per_frame if chunk_idx == 0 else frame_chunk_size * action_per_frame
            obs_indices = [
                min(cursor + step, max_idx)
                for step in range(obs_update_stride, executed_action_num + 1, obs_update_stride)
            ]
            chunk_obs = [
                self._format_gt_cache_obs(frame_idx, states, joint_states, video_frames)
                for frame_idx in obs_indices
            ]
            chunks.append(chunk_obs)
            cursor += executed_action_num
        return chunks

    def _format_gt_cache_obs(
        self,
        frame_idx: int,
        states: np.ndarray,
        joint_states: np.ndarray,
        video_frames: Dict[str, List[np.ndarray]],
    ) -> Dict[str, Any]:
        max_high = len(video_frames["observation.images.cam_high"]) - 1
        max_left = len(video_frames["observation.images.cam_left_wrist"]) - 1
        max_right = len(video_frames["observation.images.cam_right_wrist"]) - 1
        idx_high = min(frame_idx, max_high)
        idx_left = min(frame_idx, max_left)
        idx_right = min(frame_idx, max_right)
        idx_state = min(frame_idx, states.shape[0] - 1)
        idx_joint = min(frame_idx, joint_states.shape[0] - 1)
        return {
            "observation.images.cam_high": np.asarray(
                video_frames["observation.images.cam_high"][idx_high],
                dtype=np.uint8,
            ),
            "observation.images.cam_left_wrist": np.asarray(
                video_frames["observation.images.cam_left_wrist"][idx_left],
                dtype=np.uint8,
            ),
            "observation.images.cam_right_wrist": np.asarray(
                video_frames["observation.images.cam_right_wrist"][idx_right],
                dtype=np.uint8,
            ),
            "observation.state": np.asarray(states[idx_state], dtype=np.float32),
            "observation.joint_state": np.asarray(joint_states[idx_joint], dtype=np.float32),
            "task": self._prompt,
        }

    def _resolve_tracked_actor(self, task_env) -> None:
        self._tracked_actor_attr = None
        self._tracked_actor_name = None
        for attr in ("bottle", "box", "object", "can", "pillbottle", "stapler"):
            actor = getattr(task_env, attr, None)
            if actor is None:
                continue
            if not hasattr(actor, "get_pose"):
                continue
            self._tracked_actor_attr = attr
            try:
                self._tracked_actor_name = actor.get_name()
            except Exception:
                self._tracked_actor_name = None
            return

    def _get_tracked_actor(self, task_env):
        if self._tracked_actor_attr is None:
            self._resolve_tracked_actor(task_env)
        if self._tracked_actor_attr is None:
            return None
        return getattr(task_env, self._tracked_actor_attr, None)

    def _get_tracked_actor_gripper_contacts(self, task_env, actor_name: str) -> tuple[list, list]:
        positions = []
        link_names = []
        get_contact_positions = getattr(task_env, "get_gripper_actor_contact_position", None)
        if callable(get_contact_positions):
            try:
                positions = [np.asarray(p, dtype=np.float32) for p in get_contact_positions(actor_name)]
            except Exception:
                positions = []

        try:
            contacts = task_env.scene.get_contacts()
            gripper_names = set(getattr(task_env.robot, "gripper_name", []))
            for contact in contacts:
                name0 = contact.bodies[0].entity.name
                name1 = contact.bodies[1].entity.name
                if name0 == actor_name and name1 in gripper_names:
                    link_names.append(name1)
                elif name1 == actor_name and name0 in gripper_names:
                    link_names.append(name0)
        except Exception:
            link_names = []
        return positions, sorted(set(link_names))

    def _capture_tracked_actor_state(self, task_env) -> Optional[Dict[str, Any]]:
        actor = self._get_tracked_actor(task_env)
        if actor is None:
            return None
        try:
            pose = actor.get_pose()
            pose_list = np.asarray(list(pose.p) + list(pose.q), dtype=np.float32)
        except Exception:
            pose_list = None
        try:
            functional = np.asarray(actor.get_functional_point(0), dtype=np.float32)
        except Exception:
            functional = None
        actor_name = self._tracked_actor_name or actor.get_name()
        contact_positions, contact_links = self._get_tracked_actor_gripper_contacts(task_env, actor_name)
        return {
            "tracked_actor_attr": self._tracked_actor_attr,
            "tracked_actor_name": actor_name,
            "tracked_actor_pose": pose_list,
            "tracked_actor_functional_point": functional,
            "tracked_actor_gripper_contact": bool(len(contact_positions) > 0),
            "tracked_actor_gripper_contact_count": int(len(contact_positions)),
            "tracked_actor_contact_links": contact_links,
        }

    def _execute_chunk_actions(
        self,
        *,
        task_env,
        action: np.ndarray,
        infer_meta: Optional[Dict[str, Any]],
        actions_raw: Optional[np.ndarray],
        action_source: str,
        cache_obs_override: Optional[List[Dict[str, Any]]] = None,
        cache_action_override: Optional[np.ndarray] = None,
        cache_source: str = "executed",
    ) -> None:
        chunk_log = {
            "actions_raw": actions_raw,
            "actions_denormed": action.copy(),
            "actions_executed": [],
            "observed_states": [],
            "observed_action_steps": [],
            "tracked_actor_pose": [],
            "tracked_actor_functional_point": [],
            "tracked_actor_gripper_contact": [],
            "tracked_actor_gripper_contact_count": [],
            "tracked_actor_contact_links": [],
            "tracked_actor_observed_action_steps": [],
            "tracked_actor_name": self._tracked_actor_name,
            "tracked_actor_attr": self._tracked_actor_attr,
            "causal_meta": None,
        }
        if self._first and self._first_obs is not None:
            chunk_log["observed_states"].append(
                np.asarray(self._first_obs["observation.state"], dtype=np.float32).copy()
            )
            chunk_log["observed_action_steps"].append(int(self._executed_action_count))
            tracked_state = self._capture_tracked_actor_state(task_env)
            if tracked_state is not None:
                if tracked_state["tracked_actor_pose"] is not None:
                    chunk_log["tracked_actor_pose"].append(tracked_state["tracked_actor_pose"])
                if tracked_state["tracked_actor_functional_point"] is not None:
                    chunk_log["tracked_actor_functional_point"].append(tracked_state["tracked_actor_functional_point"])
                chunk_log["tracked_actor_gripper_contact"].append(tracked_state["tracked_actor_gripper_contact"])
                chunk_log["tracked_actor_gripper_contact_count"].append(tracked_state["tracked_actor_gripper_contact_count"])
                chunk_log["tracked_actor_contact_links"].append(tracked_state["tracked_actor_contact_links"])
                chunk_log["tracked_actor_observed_action_steps"].append(int(self._executed_action_count))
                chunk_log["tracked_actor_name"] = tracked_state["tracked_actor_name"]
                chunk_log["tracked_actor_attr"] = tracked_state["tracked_actor_attr"]

        assert action.shape[2] % 4 == 0
        obs_update_stride = action.shape[2] // 4
        start_idx = 1 if self._first else 0
        key_frame_list = []
        executed_action_steps = 0

        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                executed_action_steps += 1
                ee_action = action[:, i, j]

                if action.shape[0] == 16:
                    ee_action = self._add_init_pose(ee_action, self._inint_eef_pose)
                    ee_action = np.concatenate([
                        ee_action[:3],
                        ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                        ee_action[7:11],
                        ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                        ee_action[15:16],
                    ])
                elif action.shape[0] == 14:
                    ee_action = np.concatenate([
                        ee_action[:3],
                        self._euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                        ee_action[6:10],
                        self._euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                        ee_action[13:14],
                    ])
                else:
                    raise NotImplementedError(f"Unsupported action dim: {action.shape[0]}")

                chunk_log["actions_executed"].append(ee_action.copy())
                task_env.take_action(ee_action, action_type='ee')
                self._executed_action_count += 1

                if (j + 1) % obs_update_stride == 0:
                    obs = self._format_obs(task_env.get_obs())
                    key_frame_list.append(obs)
                    chunk_log["observed_states"].append(
                        np.asarray(obs["observation.state"], dtype=np.float32).copy()
                    )
                    chunk_log["observed_action_steps"].append(int(self._executed_action_count))
                    tracked_state = self._capture_tracked_actor_state(task_env)
                    if tracked_state is not None:
                        if tracked_state["tracked_actor_pose"] is not None:
                            chunk_log["tracked_actor_pose"].append(tracked_state["tracked_actor_pose"])
                        if tracked_state["tracked_actor_functional_point"] is not None:
                            chunk_log["tracked_actor_functional_point"].append(tracked_state["tracked_actor_functional_point"])
                        chunk_log["tracked_actor_gripper_contact"].append(tracked_state["tracked_actor_gripper_contact"])
                        chunk_log["tracked_actor_gripper_contact_count"].append(tracked_state["tracked_actor_gripper_contact_count"])
                        chunk_log["tracked_actor_contact_links"].append(tracked_state["tracked_actor_contact_links"])
                        chunk_log["tracked_actor_observed_action_steps"].append(int(self._executed_action_count))
                        chunk_log["tracked_actor_name"] = tracked_state["tracked_actor_name"]
                        chunk_log["tracked_actor_attr"] = tracked_state["tracked_actor_attr"]

        self._action_log.append(chunk_log)
        self._first = False

        cache_obs = cache_obs_override if cache_obs_override is not None else key_frame_list
        cache_action = cache_action_override if cache_action_override is not None else action

        cache_ret = self.model.infer(dict(
            obs=cache_obs,
            compute_kv_cache=True,
            imagine=False,
            save_visualization=self._save_visualization,
            state=cache_action,
        ))
        chunk_log["causal_meta"] = {
            "server_infer_meta": infer_meta or {"action_source": action_source},
            "server_kv_meta": cache_ret.get("rollout_meta", {}),
            "obs_update_stride": int(obs_update_stride),
            "start_idx": int(start_idx),
            "executed_action_steps": int(executed_action_steps),
            "executed_key_frame_count": int(len(key_frame_list)),
            "cache_key_frame_count": int(len(cache_obs)),
            "action_shape": [int(v) for v in action.shape],
            "action_source": action_source,
            "cache_action_shape": [int(v) for v in cache_action.shape],
            "cache_source": cache_source,
            "warmstart_chunk_idx": int(self._chunk_idx),
            "tracked_actor_name": chunk_log.get("tracked_actor_name"),
            "tracked_actor_attr": chunk_log.get("tracked_actor_attr"),
        }
        self._chunk_idx += 1

    def step(self, task_env, observation: Dict[str, Any]) -> None:
        """Run one LingbotVA inference step (one chunk of actions).

        Sends observation to server, receives actions, converts EE -> executes.
        """
        if self._chunk_idx < len(self._gt_warmstart_chunk_actions):
            gt_action = self._gt_warmstart_chunk_actions[self._chunk_idx]
            if self._first:
                obs_for_server = self._first_obs
            else:
                obs_for_server = self._first_obs
            ret = self.model.infer(dict(
                obs=obs_for_server,
                prompt=self._prompt,
                save_visualization=self._save_visualization,
                video_guidance_scale=self._video_guidance_scale,
                action_guidance_scale=self._action_guidance_scale,
                perturb_raw_override=self._episode_perturb_raw_override,
            ))
            self._execute_chunk_actions(
                task_env=task_env,
                action=gt_action,
                infer_meta={
                    **ret.get("rollout_meta", {}),
                    "action_source": "gt_warmstart",
                    "chunk_idx": int(self._chunk_idx),
                },
                actions_raw=ret.get("actions_raw"),
                action_source="gt_warmstart",
            )
            return

        if self._gt_cache_teacher_force and self._chunk_idx < len(self._gt_cache_chunk_actions):
            if self._first:
                obs_for_server = self._first_obs
            else:
                obs_for_server = self._first_obs

            ret = self.model.infer(dict(
                obs=obs_for_server,
                prompt=self._prompt,
                save_visualization=self._save_visualization,
                video_guidance_scale=self._video_guidance_scale,
                action_guidance_scale=self._action_guidance_scale,
                perturb_raw_override=self._episode_perturb_raw_override,
            ))
            self._execute_chunk_actions(
                task_env=task_env,
                action=ret["action"],
                infer_meta={
                    **ret.get("rollout_meta", {}),
                    "action_source": "model_gt_cache_teacher_force",
                    "chunk_idx": int(self._chunk_idx),
                },
                actions_raw=ret.get("actions_raw"),
                action_source="model_gt_cache_teacher_force",
                cache_obs_override=self._gt_cache_chunk_obs[self._chunk_idx],
                cache_action_override=self._gt_cache_chunk_actions[self._chunk_idx],
                cache_source="gt_teacher_force",
            )
            return

        if self._first:
            obs_for_server = self._first_obs
        else:
            obs_for_server = self._first_obs  # server uses cached obs

        # Get chunk from server
        ret = self.model.infer(dict(
            obs=obs_for_server,
            prompt=self._prompt,
            save_visualization=self._save_visualization,
            video_guidance_scale=self._video_guidance_scale,
            action_guidance_scale=self._action_guidance_scale,
            perturb_raw_override=self._episode_perturb_raw_override,
        ))
        action = ret['action']
        infer_meta = ret.get("rollout_meta", {})
        self._execute_chunk_actions(
            task_env=task_env,
            action=action,
            infer_meta=infer_meta,
            actions_raw=ret.get("actions_raw"),
            action_source="model",
        )

    def get_action_log(self) -> list:
        """Return the collected action log and reset it."""
        log = self._action_log
        self._action_log = []
        return log

    def _format_obs(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Format env observation for LingbotVA server.

        The recovery-sidecar EEF-state path expects ``observation.state`` to be
        the 16D task-space state used in training:
        ``left_xyz + left_quat + left_gripper + right_xyz + right_quat + right_gripper``.

        RobotWin online env observations also expose a 14D joint-space vector at
        ``joint_action.vector``. Keep that available as ``observation.joint_state``
        for debugging, but do not send it as ``observation.state``.
        """
        eef_state = self._extract_eef_state(observation)
        joint_state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        return {
            "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
            "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
            "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
            "observation.state": eef_state,
            "observation.joint_state": joint_state,
            "task": self._prompt,
        }

    def _extract_eef_state(self, observation: Dict[str, Any]) -> np.ndarray:
        """Extract the 16D EEF task-space state used by Lingbot-VA training."""
        endpose = observation["endpose"]
        eef_state = np.asarray(
            endpose["left_endpose"]
            + [endpose["left_gripper"]]
            + endpose["right_endpose"]
            + [endpose["right_gripper"]],
            dtype=np.float32,
        )
        if eef_state.shape != (16,):
            raise ValueError(f"Expected 16D EEF state, got shape {eef_state.shape}")
        return eef_state

    def _add_eef_pose(self, new_pose, init_pose):
        """Add delta EE pose to initial pose."""
        R = self._R
        new_pose_R = R.from_quat(new_pose[3:7][None])
        init_pose_R = R.from_quat(init_pose[3:7][None])
        out_rot = (init_pose_R * new_pose_R).as_quat().reshape(-1)
        out_trans = new_pose[:3] + init_pose[:3]
        return np.concatenate([out_trans, out_rot, new_pose[7:8]])

    def _add_init_pose(self, new_pose, init_pose):
        """Convert delta EE to absolute EE for both arms."""
        left_pose = self._add_eef_pose(new_pose[:8], init_pose[:8])
        right_pose = self._add_eef_pose(new_pose[8:], init_pose[8:])
        return np.concatenate([left_pose, right_pose])

    def _euler2quat(self, roll, pitch, yaw):
        """Convert euler angles to quaternion [x, y, z, w]."""
        R = self._R
        return R.from_euler('xyz', [roll, pitch, yaw]).as_quat()

    @property
    def action_type(self) -> str:
        return "ee"

    @property
    def name(self) -> str:
        return "lingbot"
