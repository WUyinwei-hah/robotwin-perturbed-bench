"""
Perturbed Environment Wrapper

Wraps TASK_ENV to inject perturbation at the 14D qpos execution layer,
regardless of whether the policy outputs qpos or ee actions.

For qpos policies: perturbation is applied directly to the 14D action.
For ee policies: the env does IK/path planning internally. We intercept
  the robot.set_arm_joints() calls to perturb the target joint positions
  at each TOPP waypoint. This ensures actuator-level disturbance consistency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from benchmark.perturbation_engine import (
    ACTION_DIM,
    GRIPPER_DIMS,
    JOINT_MASK,
    BasePerturbation,
    PerturbationConfig,
    PerturbationState,
    create_perturbation,
    init_state,
)

logger = logging.getLogger(__name__)


class PerturbedEnvWrapper:
    """Wraps TASK_ENV to inject perturbation at the qpos execution level.

    Usage:
        wrapper = PerturbedEnvWrapper(task_env, perturb_cfg)
        wrapper.activate()   # hooks take_action
        # ... run policy ...
        wrapper.deactivate() # restore original take_action
    """

    def __init__(self, task_env, perturb_cfg: PerturbationConfig):
        self.env = task_env
        self.cfg = perturb_cfg
        self.perturb: Optional[BasePerturbation] = None
        self._original_take_action = None
        self._action_step = 0
        self._active = False

    def activate(self) -> None:
        """Hook take_action on the env to inject perturbation."""
        if self._active:
            return

        # Initialize perturbation state from current robot state
        obs = self.env.get_obs()
        q_0 = np.array(obs["joint_action"]["vector"], dtype=np.float32)
        state = init_state(u_prev=q_0, a_prev=q_0)
        self.perturb = create_perturbation(self.cfg, state)
        self._action_step = 0

        # Monkey-patch take_action
        self._original_take_action = self.env.take_action
        self.env.take_action = self._perturbed_take_action
        self._active = True

        logger.info(
            f"PerturbedEnvWrapper activated: type={self.cfg.perturb_type.value}, "
            f"severity={self.cfg.severity.value}, timing={self.cfg.timing.value}, "
            f"t_on_raw={self.cfg.t_on_raw}"
        )

    def deactivate(self) -> None:
        """Restore original take_action."""
        if not self._active:
            return
        self.env.take_action = self._original_take_action
        self._active = False

    def _perturbed_take_action(self, action, action_type='qpos'):
        """Intercept action, apply perturbation at the qpos level, send to env.

        For qpos actions (14D): perturb directly and forward.
        For ee actions (16D): forward to original take_action but with
          robot.set_arm_joints wrapped to perturb each joint target.
        """
        if action_type == 'qpos':
            u_t = np.array(action, dtype=np.float32)
            a_t = self.perturb.apply(u_t)
            self._action_step += 1
            self._original_take_action(a_t.tolist(), action_type='qpos')

        elif action_type == 'ee':
            # For EE: we wrap the robot's set_arm_joints to perturb
            # the joint targets after IK/path planning.
            # We apply ONE perturbation step per take_action call (not per waypoint).
            #
            # Strategy: let IK run normally, but perturb the final target qpos
            # for each arm before TOPP execution.
            self._perturbed_ee_take_action(action)

        else:
            # Fallback: pass through without perturbation
            self._original_take_action(action, action_type=action_type)

    def _perturbed_ee_take_action(self, action):
        """Handle EE actions with qpos-level perturbation.

        We wrap robot.set_arm_joints to apply a perturbation offset derived
        from the perturbation engine. The offset is computed once per
        take_action call and applied consistently to all waypoints.
        """
        robot = self.env.robot

        # Compute perturbation offset for this step:
        # Get current qpos as the "command" for perturbation engine
        left_js = robot.get_left_arm_jointState()
        right_js = robot.get_right_arm_jointState()
        current_qpos = np.array(left_js + right_js, dtype=np.float32)

        # Run perturbation on current qpos to get the offset
        perturbed_qpos = self.perturb.apply(current_qpos)
        # offset = perturbed - original (what the perturbation adds)
        offset = perturbed_qpos - current_qpos
        self._action_step += 1

        # Extract per-arm offsets (joint dims only, skip gripper)
        left_offset = offset[:6]   # joints 0-5
        right_offset = offset[7:13]  # joints 7-12

        # Wrap set_arm_joints to add offset to target positions
        original_set_arm_joints = robot.set_arm_joints

        def perturbed_set_arm_joints(target_position, target_velocity, arm_tag):
            perturbed_pos = np.array(target_position, dtype=np.float64).copy()
            if arm_tag == "left":
                perturbed_pos += left_offset.astype(np.float64)
            elif arm_tag == "right":
                perturbed_pos += right_offset.astype(np.float64)
            original_set_arm_joints(perturbed_pos, target_velocity, arm_tag)

        # Temporarily install the wrapper
        robot.set_arm_joints = perturbed_set_arm_joints
        try:
            self._original_take_action(action, action_type='ee')
        finally:
            # Restore original
            robot.set_arm_joints = original_set_arm_joints

    def get_perturbation_log(self) -> Dict[str, Any]:
        """Return perturbation metadata for logging."""
        return {
            "config": self.cfg.to_dict(),
            "total_action_steps": self._action_step,
            "final_step_raw": self.perturb.state.step_raw if self.perturb else 0,
        }
