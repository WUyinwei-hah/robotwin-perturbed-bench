"""
Abstract base class for policy adapters.

Each adapter wraps a specific policy (Motus, Pi0.5, LingbotVA) and exposes
a unified interface for the benchmark runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PolicyAdapter(ABC):
    """Unified interface for all policies in the benchmark."""

    @abstractmethod
    def load(self, config: Dict[str, Any]) -> None:
        """Load model weights and initialize the policy.

        Args:
            config: policy-specific configuration dict
        """
        ...

    @abstractmethod
    def reset(self, task_env, instruction: str) -> None:
        """Reset the policy for a new episode.

        Args:
            task_env: the TASK_ENV instance
            instruction: language instruction for the task
        """
        ...

    @abstractmethod
    def step(self, task_env, observation: Dict[str, Any]) -> None:
        """Run one policy step: get action(s) and execute via task_env.take_action().

        The adapter is responsible for calling task_env.take_action() internally.

        Args:
            task_env: the TASK_ENV instance
            observation: current observation dict from task_env.get_obs()
        """
        ...

    @property
    @abstractmethod
    def action_type(self) -> str:
        """Return 'qpos' or 'ee' depending on the policy's action space."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable policy name."""
        ...
