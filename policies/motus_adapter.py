"""
Motus policy adapter for the perturbed benchmark.

Wraps the self-contained Motus deploy_policy (policies/motus_policy/).
Action space: 14D qpos.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from policies.base_adapter import PolicyAdapter


class MotusAdapter(PolicyAdapter):
    """Adapter for Motus V1 policy (qpos-based, local model)."""

    def __init__(self):
        self.model = None
        self._eval_fn = None
        self._reset_fn = None

    def load(self, config: Dict[str, Any]) -> None:
        """Load Motus model.

        Expected config keys:
            robotwin_root: path to robotwin repo (for env imports)
            checkpoint_path: path to checkpoint
            wan_path: path to WAN model
            vlm_path: path to VLM model
            device: cuda device string
        """
        robotwin_root = config.get("robotwin_root", "")
        if robotwin_root and robotwin_root not in sys.path:
            sys.path.insert(0, robotwin_root)

        # Use self-contained Motus policy code
        _bench_root = str(Path(__file__).resolve().parent.parent)
        policy_dir = os.path.join(_bench_root, "policies", "motus_policy")

        if policy_dir not in sys.path:
            sys.path.insert(0, policy_dir)

        # Import Motus deploy_policy from self-contained copy
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "motus_deploy", os.path.join(policy_dir, "deploy_policy.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self._eval_fn = module.eval
        self._reset_fn = module.reset_model

        # Build usr_args for get_model
        usr_args = {
            "ckpt_setting": config["checkpoint_path"],
            "wan_path": config["wan_path"],
            "vlm_path": config["vlm_path"],
        }
        # Merge any extra args
        for k, v in config.items():
            if k not in usr_args and k not in ("robotwin_root",):
                usr_args[k] = v

        self.model = module.get_model(usr_args)

    def reset(self, task_env, instruction: str) -> None:
        """Reset model caches for a new episode."""
        self._reset_fn(self.model)

    def step(self, task_env, observation: Dict[str, Any]) -> None:
        """Run one Motus inference step.

        The Motus eval() function internally calls:
          model.update_obs(observation)
          actions = model.get_action(instruction)
          for action in actions:
              TASK_ENV.take_action(action, action_type='qpos')
        """
        self._eval_fn(task_env, self.model, observation)

    @property
    def action_type(self) -> str:
        return "qpos"

    @property
    def name(self) -> str:
        return "Motus"
