"""
LingbotVA policy adapter for the perturbed benchmark.

Wraps the WebsocketClientPolicy (self-contained in policies/lingbot_va/).
Action space: 16D end-effector (ee).

The LingbotVA server must be launched separately before running the benchmark.
This adapter only handles the client-side websocket communication.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from typing import Any, Dict, List, Optional

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

    def step(self, task_env, observation: Dict[str, Any]) -> None:
        """Run one LingbotVA inference step (one chunk of actions).

        Sends observation to server, receives actions, converts EE -> executes.
        """
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
        ))
        action = ret['action']

        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4

        start_idx = 1 if self._first else 0
        key_frame_list = []

        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
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

                task_env.take_action(ee_action, action_type='ee')

                if (j + 1) % action_per_frame == 0:
                    obs = self._format_obs(task_env.get_obs())
                    key_frame_list.append(obs)

        self._first = False

        # Send key frames back to server for KV cache update
        self.model.infer(dict(
            obs=key_frame_list,
            compute_kv_cache=True,
            imagine=False,
            save_visualization=self._save_visualization,
            state=action,
        ))

    def _format_obs(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Format env observation for LingbotVA server."""
        return {
            "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
            "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
            "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
            "observation.state": observation["joint_action"]["vector"],
            "task": self._prompt,
        }

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
