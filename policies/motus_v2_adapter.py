"""
MotusV2 policy adapter for the perturbed benchmark.

Wraps the MotusV2 deploy_policy (policies/motus_v2_policy/).
Supports video conditioning (9 condition frames + 16 predicted frames).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from policies.base_adapter import PolicyAdapter


class MotusV2Adapter(PolicyAdapter):
    """Adapter for Motus V2 policy (video-conditioned, local model)."""

    def __init__(self):
        self.model = None
        self._eval_fn = None
        self._reset_fn = None

    def load(self, config: Dict[str, Any]) -> None:
        """Load MotusV2 model.

        Expected config keys:
            checkpoint_path: path to checkpoint
            wan_path: path to WAN model
            vlm_path: path to VLM model
            device: cuda device string (optional)
        """
        _bench_root = str(Path(__file__).resolve().parent.parent)
        policy_dir = os.path.join(_bench_root, "policies", "motus_v2_policy")

        if policy_dir not in sys.path:
            sys.path.insert(0, policy_dir)

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "motus_v2_deploy", os.path.join(policy_dir, "deploy_policy.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self._eval_fn = module.eval
        self._reset_fn = module.reset_model

        usr_args = {
            "ckpt_setting": config["checkpoint_path"],
            "wan_path": config["wan_path"],
            "vlm_path": config["vlm_path"],
        }
        for k, v in config.items():
            if k not in usr_args:
                usr_args[k] = v

        self.model = module.get_model(usr_args)

    def reset(self, task_env, instruction: str) -> None:
        self._reset_fn(self.model)

    def step(self, task_env, observation: Dict[str, Any]) -> None:
        self._eval_fn(task_env, self.model, observation)

    @property
    def action_type(self) -> str:
        return "qpos"

    @property
    def name(self) -> str:
        return "MotusV2"
