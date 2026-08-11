"""Abstract base environment for the MPC vs VLA vs Diffusion benchmark study.

This module defines :class:`BaseEnv`, a minimal gym-like environment interface
that every benchmark in this package implements.  The interface is deliberately
framework-agnostic (no ``gym`` dependency) so that the three controller
families -- Model Predictive Control (MPC), Vision-Language-Action models
(VLA), and Diffusion Policies -- can all be evaluated against the same
environments.

Design notes
------------
* **State-based observations** are consumed by MPC controllers via
  :meth:`BaseEnv.get_state`.
* **Image-based observations** are consumed by VLA / diffusion controllers via
  :meth:`BaseEnv.get_image`.
* **Language instructions** are consumed by VLA controllers via
  :meth:`BaseEnv.get_language_instruction`.
* Every concrete environment must implement :meth:`_compute_obs`,
  :meth:`_compute_reward`, :meth:`is_success`, :meth:`is_collision` and
  :meth:`_render_image`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np

__all__ = ["BaseEnv", "EnvSpec"]


class EnvSpec:
    """A lightweight replacement for ``gym.spaces`` metadata.

    Parameters
    ----------
    shape : tuple
        Shape of the observation or action tensor.
    low : array-like or float
        Lower bound (broadcastable to ``shape``).
    high : array-like or float
        Upper bound (broadcastable to ``shape``).
    dtype : np.dtype
        Numpy dtype of the space.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        low: Any = -np.inf,
        high: Any = np.inf,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.shape = tuple(shape)
        self.low = np.broadcast_to(low, self.shape).astype(dtype)
        self.high = np.broadcast_to(high, self.shape).astype(dtype)
        self.dtype = dtype

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw a uniform random sample from the space."""
        rng = rng if rng is not None else np.random.default_rng()
        if np.isinf(self.low).any() or np.isinf(self.high).any():
            return rng.standard_normal(self.shape).astype(self.dtype)
        return rng.uniform(self.low, self.high).astype(self.dtype)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"EnvSpec(shape={self.shape}, low={self.low}, high={self.high})"


class BaseEnv(ABC):
    """Abstract base class for all benchmark environments.

    Subclasses must populate ``self._observation_space``, ``self._action_space``
    and ``self._max_steps`` in their ``__init__`` (or override the
    corresponding properties).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._step_count = 0
        self._state: np.ndarray | None = None
        # Subclasses set these.
        self._observation_space: EnvSpec | None = None
        self._action_space: EnvSpec | None = None
        self._max_steps: int = 200

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def observation_space(self) -> EnvSpec:
        if self._observation_space is None:
            raise RuntimeError("observation_space has not been set.")
        return self._observation_space

    @property
    def action_space(self) -> EnvSpec:
        if self._action_space is None:
            raise RuntimeError("action_space has not been set.")
        return self._action_space

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def done(self) -> bool:
        """Whether the episode has terminated (success, collision, or timeout)."""
        return self.is_success() or self.is_collision() or self._step_count >= self._max_steps

    # ------------------------------------------------------------------
    # Core gym-like interface
    # ------------------------------------------------------------------
    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply ``action`` and return ``(observation, reward, done, info)``."""

    # ------------------------------------------------------------------
    # Observation helpers (state / image / language)
    # ------------------------------------------------------------------
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the full underlying state vector (used by MPC)."""

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Return the agent-facing observation (state or features)."""

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """Return an RGB image observation ``(H, W, 3)`` (used by VLA/diffusion)."""

    @abstractmethod
    def get_language_instruction(self) -> str:
        """Return a natural-language description of the task (used by VLA)."""

    # ------------------------------------------------------------------
    # Termination predicates
    # ------------------------------------------------------------------
    @abstractmethod
    def is_success(self) -> bool:
        """Return ``True`` if the task goal has been achieved."""

    @abstractmethod
    def is_collision(self) -> bool:
        """Return ``True`` if the agent/body is in collision."""

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        """Render the environment.

        Parameters
        ----------
        mode : str
            ``"rgb_array"`` returns an ``(H, W, 3)`` uint8 numpy array.
            ``"human"`` is a no-op placeholder for display backends.
        """
        if mode == "rgb_array":
            return self.get_image()
        return None

    # ------------------------------------------------------------------
    # Internal helpers used by subclasses
    # ------------------------------------------------------------------
    def _increment_step(self) -> None:
        self._step_count += 1

    def _reset_step_count(self) -> None:
        self._step_count = 0

    def _reseed(self, seed: int | None) -> None:
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
