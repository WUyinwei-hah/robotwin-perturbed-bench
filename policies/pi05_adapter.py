"""
Pi0.5 policy adapter for the perturbed benchmark.

Wraps the self-contained pi05 deploy_policy (policies/pi05_policy/).
Action space: 14D qpos.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from policies.base_adapter import PolicyAdapter


class Pi05Adapter(PolicyAdapter):
    """Adapter for Pi0.5 policy (qpos-based, JAX model)."""

    def __init__(self):
        self.model = None
        self._eval_fn = None
        self._reset_fn = None

    def load(self, config: Dict[str, Any]) -> None:
        """Load Pi0.5 model.

        Expected config keys:
            robotwin_root: path to robotwin repo (for env imports)
            train_config_name: training config name
            model_name: model name
            checkpoint_id: checkpoint id (default 30000)
            pi0_step: action horizon (default 50)
        """
        robotwin_root = config.get("robotwin_root", "")
        if robotwin_root and robotwin_root not in sys.path:
            sys.path.insert(0, robotwin_root)

        # Use self-contained Pi0.5 policy code
        _bench_root = str(Path(__file__).resolve().parent.parent)
        policy_dir = os.path.join(_bench_root, "policies", "pi05_policy")
        policy_src_dir = os.path.join(policy_dir, "src")

        if policy_dir not in sys.path:
            sys.path.insert(0, policy_dir)
        if policy_src_dir not in sys.path:
            sys.path.insert(0, policy_src_dir)

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pi05_deploy", os.path.join(policy_dir, "deploy_policy.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self._eval_fn = module.eval
        self._reset_fn = module.reset_model

        usr_args = {
            "train_config_name": config["train_config_name"],
            "model_name": config["model_name"],
            "checkpoint_id": config.get("checkpoint_id", 30000),
            "pi0_step": config.get("pi0_step", 50),
            "checkpoint_dir": config.get("checkpoint_dir", None),
        }

        self.model = module.get_model(usr_args)

    def reset(self, task_env, instruction: str) -> None:
        """Reset observation windows for a new episode."""
        self._reset_fn(self.model)

    def step(self, task_env, observation: Dict[str, Any]) -> None:
        """Run one Pi0.5 inference step.

        The Pi0.5 eval() function internally calls:
          model.update_observation_window(rgb, state)
          actions = model.get_action()[:pi0_step]
          for action in actions:
              TASK_ENV.take_action(action)  # default action_type='qpos'
        """
        self._eval_fn(task_env, self.model, observation)

    @property
    def action_type(self) -> str:
        return "qpos"

    @property
    def name(self) -> str:
        return "Pi05"
