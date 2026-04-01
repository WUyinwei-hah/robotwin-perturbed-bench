"""
LingbotVA qpos policy adapter for the perturbed benchmark.

Wraps the websocket client and executes 14D qpos actions directly.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np

from policies.base_adapter import PolicyAdapter
from policies.lingbot_va.websocket_client_policy import WebsocketClientPolicy


class LingbotVAQPosAdapter(PolicyAdapter):

    def __init__(self):
        self.model = None
        self._host = "127.0.0.1"
        self._port = 8000
        self._video_guidance_scale = 5.0
        self._action_guidance_scale = 1.0
        self._save_visualization = False
        self._prompt = None
        self._first = True
        self._first_obs = None
        self._action_log = []
        self._executed_action_count = 0
        self._chunk_idx = 0
        self._episode_seed = None
        self._episode_task_name = None
        self._episode_perturbation_log = None

    def load(self, config: Dict[str, Any]) -> None:
        robotwin_root = config.get("robotwin_root", "")
        if robotwin_root and robotwin_root not in sys.path:
            sys.path.insert(0, robotwin_root)

        self._port = int(config.get("port", 8000))
        self._host = str(config.get("host", "127.0.0.1"))
        self._video_guidance_scale = float(config.get("video_guidance_scale", 5.0))
        self._action_guidance_scale = float(config.get("action_guidance_scale", 1.0))
        self._save_visualization = bool(config.get("save_visualization", False))

        local_hosts = {"127.0.0.1", "localhost", "0.0.0.0", self._host}
        existing_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        merged_no_proxy = [h.strip() for h in existing_no_proxy.split(",") if h.strip()]
        for host in local_hosts:
            if host not in merged_no_proxy:
                merged_no_proxy.append(host)
        os.environ["NO_PROXY"] = ",".join(merged_no_proxy)
        os.environ["no_proxy"] = os.environ["NO_PROXY"]

        self.model = WebsocketClientPolicy(host=self._host, port=self._port)

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
        self._episode_perturbation_log = perturbation_log

    def reset(self, task_env, instruction: str) -> None:
        self._prompt = instruction
        self._first = True
        self._first_obs = None
        self._action_log = []
        self._executed_action_count = 0
        self._chunk_idx = 0

        self.model.infer(
            dict(
                reset=True,
                prompt=self._prompt,
                save_visualization=self._save_visualization,
                gt_perturbation_log=self._episode_perturbation_log,
            )
        )

        initial_obs = task_env.get_obs()
        self._first_obs = self._format_obs(initial_obs)

    def step(self, task_env, observation: Dict[str, Any]) -> None:
        obs_for_server = self._first_obs if self._first else self._first_obs
        ret = self.model.infer(
            dict(
                obs=obs_for_server,
                prompt=self._prompt,
                save_visualization=self._save_visualization,
                video_guidance_scale=self._video_guidance_scale,
                action_guidance_scale=self._action_guidance_scale,
            )
        )
        action = np.asarray(ret["action"], dtype=np.float32)

        assert action.ndim == 3 and action.shape[0] == 14, (
            f"Expected qpos action shape [14, F, N], got {action.shape}"
        )
        assert action.shape[2] % 4 == 0, f"Expected action_per_frame divisible by 4, got {action.shape}"

        obs_update_stride = action.shape[2] // 4
        start_idx = 1 if self._first else 0
        key_frame_list = []
        chunk_log = {
            "actions_raw": ret.get("actions_raw"),
            "actions_denormed": action.copy(),
            "actions_executed": [],
            "observed_states": [],
            "observed_action_steps": [],
            "causal_meta": None,
        }

        if self._first and self._first_obs is not None:
            chunk_log["observed_states"].append(
                np.asarray(self._first_obs["observation.state"], dtype=np.float32).copy()
            )
            chunk_log["observed_action_steps"].append(0)

        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                qpos_action = action[:, i, j].flatten().astype(np.float32)
                chunk_log["actions_executed"].append(qpos_action.copy())
                task_env.take_action(qpos_action, action_type="qpos")
                self._executed_action_count += 1

                if (j + 1) % obs_update_stride == 0:
                    obs = self._format_obs(task_env.get_obs())
                    key_frame_list.append(obs)
                    chunk_log["observed_states"].append(
                        np.asarray(obs["observation.state"], dtype=np.float32).copy()
                    )
                    chunk_log["observed_action_steps"].append(int(self._executed_action_count))

        cache_ret = self.model.infer(
            dict(
                obs=key_frame_list,
                compute_kv_cache=True,
                imagine=False,
                save_visualization=self._save_visualization,
                state=action,
                perturb_raw_override=self._get_perturb_raw_override(),
            )
        )

        chunk_log["causal_meta"] = {
            "server_infer_meta": ret.get("rollout_meta", {}),
            "server_kv_meta": cache_ret.get("rollout_meta", {}),
            "obs_update_stride": int(obs_update_stride),
            "start_idx": int(start_idx),
            "executed_action_steps": int(self._executed_action_count),
            "executed_key_frame_count": int(len(key_frame_list)),
            "action_shape": [int(v) for v in action.shape],
            "chunk_idx": int(self._chunk_idx),
            "episode_seed": self._episode_seed,
            "task_name": self._episode_task_name,
        }

        self._action_log.append(chunk_log)
        self._first = False
        self._chunk_idx += 1

    def get_action_log(self) -> list:
        log = self._action_log
        self._action_log = []
        return log

    def _format_obs(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        joint_state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        return {
            "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
            "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
            "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
            "observation.state": joint_state,
            "observation.joint_state": joint_state,
            "task": self._prompt,
        }

    def _get_perturb_raw_override(self):
        if os.environ.get("LINGBOT_VA_QPOS_USE_GT_PERTURB", "0") != "1":
            return None
        perturb_log = self._episode_perturbation_log or {}
        scale = (((perturb_log.get("config") or {}).get("params") or {}).get("S"))
        if scale is None:
            return None
        return np.asarray(scale, dtype=np.float32)

    @property
    def action_type(self) -> str:
        return "qpos"

    @property
    def name(self) -> str:
        return "lingbot_va_qpos"
