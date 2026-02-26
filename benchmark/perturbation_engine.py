"""
Perturbation Engine for Perturbed RoboTwin Benchmark (forked from Motus v1.3)

Supports 5 perturbation types: Scale, Coupling, Low-pass IIR/FIR, Bias.
Drop is excluded from the benchmark.

Two timing modes:
  - always_on:        active from step 0 to episode end
  - onset_then_always: active from t_on_raw to episode end

Convention:
  - ACTION_DIM = 14: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
  - Gripper dims (6, 13) are never perturbed.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

ACTION_DIM = 14
GRIPPER_DIMS = [6, 13]  # left gripper, right gripper
JOINT_DIMS = [i for i in range(ACTION_DIM) if i not in GRIPPER_DIMS]  # 12 joint dims

Q_MIN_DEFAULT = np.full(ACTION_DIM, -3.14, dtype=np.float32)
Q_MAX_DEFAULT = np.full(ACTION_DIM, 3.14, dtype=np.float32)
DELTA_MAX_DEFAULT = np.full(ACTION_DIM, 0.5, dtype=np.float32)


# ============================================================
# Enums
# ============================================================

class PerturbationType(str, Enum):
    SCALE = "scale"
    COUPLING = "coupling"
    LOWPASS_IIR = "lowpass_iir"
    LOWPASS_FIR = "lowpass_fir"
    BIAS = "bias"


# Benchmark perturbation types (no drop)
BENCHMARK_PERTURB_TYPES = list(PerturbationType)


class Severity(str, Enum):
    LM = "lm"
    HIGH = "high"


class TimingMode(str, Enum):
    ALWAYS_ON = "always_on"           # active from step 0
    ONSET_THEN_ALWAYS = "onset_then_always"  # active from t_on_raw


class Scenario(str, Enum):
    """Internal scenario mapping for compatibility."""
    A_ALWAYS = "A"
    B_ONSET = "B"


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class PerturbationConfig:
    """Immutable configuration for a perturbation episode."""
    perturb_type: PerturbationType
    severity: Severity
    timing: TimingMode
    t_on_raw: int  # raw index where perturbation activates (0 for always_on)
    params: Dict[str, Any]  # type-specific parameters
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation_type": self.perturb_type.value,
            "severity": self.severity.value,
            "timing": self.timing.value,
            "t_on_raw": self.t_on_raw,
            "params": _numpy_to_list(self.params),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerturbationConfig":
        return cls(
            perturb_type=PerturbationType(d["perturbation_type"]),
            severity=Severity(d["severity"]),
            timing=TimingMode(d["timing"]),
            t_on_raw=d["t_on_raw"],
            params=d["params"],
            seed=d.get("seed"),
        )


@dataclass
class PerturbationState:
    """Mutable internal state of the perturbation channel."""
    u_prev: np.ndarray  # [14] last command
    a_prev: np.ndarray  # [14] last channel output
    step_raw: int = 0

    # FIR history buffer
    u_hist: Optional[Deque] = None

    def copy(self) -> "PerturbationState":
        s = PerturbationState(
            u_prev=self.u_prev.copy(),
            a_prev=self.a_prev.copy(),
            step_raw=self.step_raw,
        )
        if self.u_hist is not None:
            s.u_hist = deque([x.copy() for x in self.u_hist], maxlen=self.u_hist.maxlen)
        return s


# ============================================================
# Utility functions
# ============================================================

def joint_mask_14d() -> np.ndarray:
    """Boolean mask: True for joint dims, False for gripper dims."""
    mask = np.ones(ACTION_DIM, dtype=bool)
    for g in GRIPPER_DIMS:
        mask[g] = False
    return mask


JOINT_MASK = joint_mask_14d()


def clip_action(
    u_raw: np.ndarray,
    u_prev: np.ndarray,
    q_min: np.ndarray = Q_MIN_DEFAULT,
    q_max: np.ndarray = Q_MAX_DEFAULT,
    delta_max: Optional[np.ndarray] = DELTA_MAX_DEFAULT,
) -> np.ndarray:
    """Clip action to joint limits and optionally rate-limit."""
    u = np.clip(u_raw, q_min, q_max)
    if delta_max is not None:
        du = np.clip(u - u_prev, -delta_max, delta_max)
        u = u_prev + du
    return u


def _numpy_to_list(obj):
    """Recursively convert numpy arrays/scalars to native Python for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_list(x) for x in obj]
    return obj


# ============================================================
# Base Perturbation
# ============================================================

class BasePerturbation:
    """Base class for all perturbation types."""

    def __init__(self, cfg: PerturbationConfig, state: PerturbationState):
        self.cfg = cfg
        self.state = state

    @property
    def active(self) -> bool:
        """Whether perturbation is currently active at this step."""
        return self.state.step_raw >= self.cfg.t_on_raw

    def apply(self, u_t: np.ndarray) -> np.ndarray:
        """Apply perturbation channel to command u_t, return ã_t.
        MUST update self.state (u_prev, a_prev, step_raw)."""
        raise NotImplementedError

    def _passthrough(self, u_t: np.ndarray) -> np.ndarray:
        """No perturbation: output = input. Update state and return."""
        a_t = u_t.copy()
        self.state.u_prev = u_t.copy()
        self.state.a_prev = a_t.copy()
        self.state.step_raw += 1
        return a_t

    def _update_state(self, u_t: np.ndarray, a_t: np.ndarray):
        """Common state update after apply."""
        self.state.u_prev = u_t.copy()
        self.state.a_prev = a_t.copy()
        self.state.step_raw += 1


# ============================================================
# Scale Perturbation
# ============================================================

class ScalePerturbation(BasePerturbation):
    """Element-wise scale on action increments.
    Forward:  ã_t = ã_prev + S·(u_t - u_prev)
    """

    def __init__(self, cfg: PerturbationConfig, state: PerturbationState):
        super().__init__(cfg, state)
        self.S = np.array(cfg.params["S"], dtype=np.float32)
        assert self.S.shape == (ACTION_DIM,), f"S must be [14], got {self.S.shape}"
        for g in GRIPPER_DIMS:
            self.S[g] = 1.0

    def apply(self, u_t: np.ndarray) -> np.ndarray:
        if not self.active:
            return self._passthrough(u_t)
        du = u_t - self.state.u_prev
        a_t = self.state.a_prev.copy()
        a_t[JOINT_MASK] += self.S[JOINT_MASK] * du[JOINT_MASK]
        a_t[~JOINT_MASK] = u_t[~JOINT_MASK]
        self._update_state(u_t, a_t)
        return a_t


# ============================================================
# Coupling Perturbation
# ============================================================

class CouplingPerturbation(BasePerturbation):
    """Local cross-coupling (banded matrix) on action increments.
    Forward:  ã_t = ã_prev + M·(u_t - u_prev)
    """

    def __init__(self, cfg: PerturbationConfig, state: PerturbationState):
        super().__init__(cfg, state)
        self.M = np.array(cfg.params["M"], dtype=np.float32)
        assert self.M.shape == (ACTION_DIM, ACTION_DIM), f"M must be [14,14], got {self.M.shape}"
        for g in GRIPPER_DIMS:
            self.M[g, :] = 0.0
            self.M[:, g] = 0.0
            self.M[g, g] = 1.0

    def apply(self, u_t: np.ndarray) -> np.ndarray:
        if not self.active:
            return self._passthrough(u_t)
        du = u_t - self.state.u_prev
        a_t = self.state.a_prev + self.M @ du
        a_t[~JOINT_MASK] = u_t[~JOINT_MASK]
        self._update_state(u_t, a_t)
        return a_t


# ============================================================
# Low-pass IIR Perturbation
# ============================================================

class LowPassIIRPerturbation(BasePerturbation):
    """First-order IIR low-pass (exponential smoothing).
    Forward:  ã_t = (1-α)·ã_prev + α·u_t
    """

    def __init__(self, cfg: PerturbationConfig, state: PerturbationState):
        super().__init__(cfg, state)
        self.alpha = float(cfg.params["alpha"])
        assert 0 < self.alpha <= 1.0, f"alpha must be in (0, 1], got {self.alpha}"

    def apply(self, u_t: np.ndarray) -> np.ndarray:
        if not self.active:
            return self._passthrough(u_t)
        a_t = u_t.copy()
        a_t[JOINT_MASK] = (
            (1 - self.alpha) * self.state.a_prev[JOINT_MASK]
            + self.alpha * u_t[JOINT_MASK]
        )
        a_t[~JOINT_MASK] = u_t[~JOINT_MASK]
        self._update_state(u_t, a_t)
        return a_t


# ============================================================
# Low-pass FIR Perturbation
# ============================================================

class LowPassFIRPerturbation(BasePerturbation):
    """N-tap FIR filter.
    Forward:  ã_t = Σ_{k=0}^{N} w_k · u_{t-k}
    """

    def __init__(self, cfg: PerturbationConfig, state: PerturbationState):
        super().__init__(cfg, state)
        self.w = np.array(cfg.params["w"], dtype=np.float32)
        self.N = len(self.w) - 1
        assert self.w[0] > 0, f"w_0 must be > 0, got {self.w[0]}"
        assert abs(self.w.sum() - 1.0) < 1e-5, f"FIR weights must sum to 1, got {self.w.sum()}"
        if state.u_hist is None:
            state.u_hist = deque(maxlen=self.N)

    def apply(self, u_t: np.ndarray) -> np.ndarray:
        if not self.active:
            self.state.u_hist.append(u_t.copy())
            return self._passthrough(u_t)

        a_t = u_t.copy()
        weighted = self.w[0] * u_t[JOINT_MASK]
        hist_list = list(self.state.u_hist)
        for k in range(1, self.N + 1):
            if k <= len(hist_list):
                u_past = hist_list[-(k)][JOINT_MASK]
            else:
                u_past = hist_list[0][JOINT_MASK] if hist_list else self.state.u_prev[JOINT_MASK]
            weighted += self.w[k] * u_past

        a_t[JOINT_MASK] = weighted
        a_t[~JOINT_MASK] = u_t[~JOINT_MASK]
        self.state.u_hist.append(u_t.copy())
        self._update_state(u_t, a_t)
        return a_t


# ============================================================
# Bias Perturbation
# ============================================================

class BiasPerturbation(BasePerturbation):
    """Fixed additive bias on joint commands.
    Forward:  ã_t = u_t + b
    """

    def __init__(self, cfg: PerturbationConfig, state: PerturbationState):
        super().__init__(cfg, state)
        self.b = np.array(cfg.params["b"], dtype=np.float32)
        assert self.b.shape == (ACTION_DIM,), f"b must be [14], got {self.b.shape}"
        for g in GRIPPER_DIMS:
            self.b[g] = 0.0

    def apply(self, u_t: np.ndarray) -> np.ndarray:
        if not self.active:
            return self._passthrough(u_t)
        a_t = u_t.copy()
        a_t[JOINT_MASK] += self.b[JOINT_MASK]
        a_t[~JOINT_MASK] = u_t[~JOINT_MASK]
        self._update_state(u_t, a_t)
        return a_t


# ============================================================
# Factory & Sampling
# ============================================================

PERTURBATION_CLASSES = {
    PerturbationType.SCALE: ScalePerturbation,
    PerturbationType.COUPLING: CouplingPerturbation,
    PerturbationType.LOWPASS_IIR: LowPassIIRPerturbation,
    PerturbationType.LOWPASS_FIR: LowPassFIRPerturbation,
    PerturbationType.BIAS: BiasPerturbation,
}


def create_perturbation(cfg: PerturbationConfig, state: PerturbationState) -> BasePerturbation:
    """Factory: create perturbation instance from config."""
    cls = PERTURBATION_CLASSES[cfg.perturb_type]
    return cls(cfg, state)


def init_state(
    u_prev: np.ndarray,
    a_prev: np.ndarray,
) -> PerturbationState:
    """Create initial perturbation state."""
    return PerturbationState(
        u_prev=u_prev.copy(),
        a_prev=a_prev.copy(),
    )


def sample_perturbation_params(
    perturb_type: PerturbationType,
    severity: Severity,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Sample perturbation parameters for the given type and severity."""
    if perturb_type == PerturbationType.SCALE:
        return _sample_scale_params(severity, rng)
    elif perturb_type == PerturbationType.COUPLING:
        return _sample_coupling_params(severity, rng)
    elif perturb_type == PerturbationType.LOWPASS_IIR:
        return _sample_iir_params(severity, rng)
    elif perturb_type == PerturbationType.LOWPASS_FIR:
        return _sample_fir_params(severity, rng)
    elif perturb_type == PerturbationType.BIAS:
        return _sample_bias_params(severity, rng)
    else:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")


# ============================================================
# Parameter Sampling Functions
# ============================================================

def _sample_scale_params(severity: Severity, rng: np.random.Generator) -> Dict[str, Any]:
    S = np.ones(ACTION_DIM, dtype=np.float32)
    if severity == Severity.LM:
        delta = rng.uniform(0.05, 0.15)
    else:
        delta = rng.uniform(0.2, 0.4)
    for j in JOINT_DIMS:
        S[j] = rng.uniform(1 - delta, 1 + delta)
    if severity == Severity.LM:
        S[JOINT_MASK] = np.clip(S[JOINT_MASK], 0.7, None)
    return {"S": S}


def _sample_coupling_params(severity: Severity, rng: np.random.Generator) -> Dict[str, Any]:
    M = np.eye(ACTION_DIM, dtype=np.float32)
    if severity == Severity.LM:
        e_max = rng.uniform(0.05, 0.1)
    else:
        e_max = rng.uniform(0.15, 0.25)
    arm_ranges = [(0, 6), (7, 13)]
    bandwidth = rng.choice([1, 2])
    for arm_start, arm_end in arm_ranges:
        for i in range(arm_start, arm_end):
            for offset in range(1, bandwidth + 1):
                j = i + offset
                if arm_start <= j < arm_end:
                    M[i, j] = rng.uniform(-e_max, e_max)
                j = i - offset
                if arm_start <= j < arm_end:
                    M[i, j] = rng.uniform(-e_max, e_max)
    max_attempts = 10
    for attempt in range(max_attempts):
        cond = np.linalg.cond(M)
        if cond < 20:
            break
        E = M - np.eye(ACTION_DIM)
        E *= 0.5
        M = np.eye(ACTION_DIM, dtype=np.float32) + E
    else:
        logger.warning(f"Could not get cond(M)<20 after {max_attempts} attempts, cond={cond:.1f}")
    return {"M": M}


def _sample_iir_params(severity: Severity, rng: np.random.Generator) -> Dict[str, Any]:
    if severity == Severity.LM:
        alpha = rng.uniform(0.4, 0.9)
    else:
        alpha = rng.uniform(0.1, 0.3)
    return {"alpha": float(alpha)}


def _sample_fir_params(severity: Severity, rng: np.random.Generator) -> Dict[str, Any]:
    N = rng.choice([1, 2, 3])
    if severity == Severity.LM:
        w0_min = 0.3
    else:
        w0_min = 0.05
    w0 = rng.uniform(w0_min, 0.9)
    remaining = 1.0 - w0
    tail = rng.dirichlet(np.ones(N))
    weights = np.zeros(N + 1, dtype=np.float32)
    weights[0] = w0
    weights[1:] = remaining * tail
    return {"w": weights}


def _sample_bias_params(severity: Severity, rng: np.random.Generator) -> Dict[str, Any]:
    b = np.zeros(ACTION_DIM, dtype=np.float32)
    if severity == Severity.LM:
        b_max = rng.uniform(0.02, 0.08)
    else:
        b_max = rng.uniform(0.1, 0.25)
    for j in JOINT_DIMS:
        b[j] = rng.uniform(-b_max, b_max)
    return {"b": b}
