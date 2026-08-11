"""Demonstration collection utilities.

This module provides :class:`DemonstrationCollector`, a helper that rolls out
policies in any :class:`benchmarks.base_env.BaseEnv` and accumulates
``(observation, action, next_observation, reward, done)`` transitions into a
single dataset dictionary.  The collected data can be saved to / loaded from
NumPy ``.npz`` archives and sampled in mini-batches for imitation-learning
pipelines (VLA, diffusion policy, etc.).

Three collection modes are supported:

* :meth:`DemonstrationCollector.from_env` -- roll out an arbitrary policy
  function ``policy_fn(obs) -> action``.
* :meth:`DemonstrationCollector.from_mpc` -- roll out an MPC controller whose
  interface is ``controller.solve(state, ref) -> action``.
* :meth:`DemonstrationCollector.from_expert` -- roll out a custom expert
  function ``expert_fn(env) -> action`` that has full access to the environment.

Both state-based and image-based observations are supported.  When the
environment provides image observations (``image_mode=True``) the collector
stores rendered images instead of the raw state vector.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .base_env import BaseEnv

__all__ = ["DemonstrationCollector"]


class DemonstrationCollector:
    """Collect demonstration transitions from a :class:`BaseEnv`.

    Parameters
    ----------
    image_mode : bool, optional
        If ``True``, store rendered images (via ``env.get_image()``) as
        observations instead of the state-based observation vector.
    image_key : str, optional
        When ``image_mode`` is ``True`` the key used for image observations in
        the dataset dict.  Defaults to ``"observations"`` so that downstream
        code can treat the dict uniformly.
    """

    def __init__(
        self,
        image_mode: bool = False,
        image_key: str = "observations",
    ) -> None:
        self.image_mode = bool(image_mode)
        self.image_key = image_key
        self._data: Dict[str, List[np.ndarray]] = {
            "observations": [],
            "actions": [],
            "next_observations": [],
            "rewards": [],
            "dones": [],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_obs(self, env: BaseEnv) -> np.ndarray:
        """Return the current observation (image or state) from *env*."""
        if self.image_mode:
            return np.asarray(env.get_image(), dtype=np.uint8)
        return np.asarray(env.get_observation(), dtype=np.float32)

    def _get_next_obs(self, env: BaseEnv) -> np.ndarray:
        """Return the post-step observation (image or state) from *env*."""
        return self._get_obs(env)

    def _append(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self._data["observations"].append(obs)
        self._data["actions"].append(np.asarray(action, dtype=np.float32))
        self._data["next_observations"].append(next_obs)
        self._data["rewards"].append(float(reward))
        self._data["dones"].append(bool(done))

    def _rollout(
        self,
        env: BaseEnv,
        action_fn: Callable[[BaseEnv, np.ndarray], np.ndarray],
        n_episodes: int,
        max_steps: int,
        seeds: Optional[Sequence[int]] = None,
    ) -> None:
        """Run *n_episodes* rollouts using *action_fn*.

        ``action_fn(env, obs)`` is called at every step and must return an
        action array compatible with ``env.action_space``.
        """
        for ep in range(n_episodes):
            seed = seeds[ep] if seeds is not None else None
            env.reset(seed=seed)
            for _ in range(max_steps):
                obs = self._get_obs(env)
                action = action_fn(env, obs)
                _, reward, done, _ = env.step(action)
                next_obs = self._get_next_obs(env)
                self._append(obs, action, next_obs, reward, done)
                if done:
                    break

    # ------------------------------------------------------------------
    # Public collection API
    # ------------------------------------------------------------------
    def from_env(
        self,
        env: BaseEnv,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        n_episodes: int = 10,
        max_steps: Optional[int] = None,
        seeds: Optional[Sequence[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Collect demonstrations using a generic policy function.

        Parameters
        ----------
        env : BaseEnv
            The environment to collect from.
        policy_fn : callable
            ``policy_fn(observation) -> action``.
        n_episodes : int
            Number of episodes to roll out.
        max_steps : int, optional
            Maximum steps per episode.  Defaults to ``env.max_steps``.
        seeds : sequence of int, optional
            Per-episode reset seeds.

        Returns
        -------
        dict
            Dataset with keys ``observations``, ``actions``,
            ``next_observations``, ``rewards``, ``dones``.
        """
        max_steps = max_steps if max_steps is not None else env.max_steps

        def action_fn(e: BaseEnv, obs: np.ndarray) -> np.ndarray:
            return np.asarray(policy_fn(obs), dtype=np.float32)

        self._rollout(env, action_fn, n_episodes, max_steps, seeds)
        return self.get_dataset()

    def from_mpc(
        self,
        env: BaseEnv,
        mpc_controller: Any,
        n_episodes: int = 10,
        max_steps: Optional[int] = None,
        seeds: Optional[Sequence[int]] = None,
        ref_fn: Optional[Callable[[BaseEnv], np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Collect demonstrations using an MPC controller.

        The controller must expose ``solve(state, ref) -> action``.

        Parameters
        ----------
        env : BaseEnv
            The environment to collect from.
        mpc_controller : object
            Object with a ``solve(state, ref)`` method.
        n_episodes : int
            Number of episodes to roll out.
        max_steps : int, optional
            Maximum steps per episode.
        seeds : sequence of int, optional
            Per-episode reset seeds.
        ref_fn : callable, optional
            ``ref_fn(env) -> reference``.  If ``None`` the target is retrieved
            via ``env.get_target()`` when available, otherwise ``None``.

        Returns
        -------
        dict
            Dataset dictionary.
        """
        max_steps = max_steps if max_steps is not None else env.max_steps

        def _get_ref(e: BaseEnv) -> Any:
            if ref_fn is not None:
                return ref_fn(e)
            if hasattr(e, "get_target"):
                return e.get_target()
            return None

        def action_fn(e: BaseEnv, obs: np.ndarray) -> np.ndarray:
            state = e.get_state()
            ref = _get_ref(e)
            action = mpc_controller.solve(state, ref)
            return np.asarray(action, dtype=np.float32)

        self._rollout(env, action_fn, n_episodes, max_steps, seeds)
        return self.get_dataset()

    def from_expert(
        self,
        env: BaseEnv,
        expert_fn: Callable[[BaseEnv], np.ndarray],
        n_episodes: int = 10,
        max_steps: Optional[int] = None,
        seeds: Optional[Sequence[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Collect demonstrations using a custom expert function.

        ``expert_fn(env) -> action`` has full access to the environment
        internals (state, target, obstacles, etc.).

        Parameters
        ----------
        env : BaseEnv
            The environment to collect from.
        expert_fn : callable
            ``expert_fn(env) -> action``.
        n_episodes : int
            Number of episodes to roll out.
        max_steps : int, optional
            Maximum steps per episode.
        seeds : sequence of int, optional
            Per-episode reset seeds.

        Returns
        -------
        dict
            Dataset dictionary.
        """
        max_steps = max_steps if max_steps is not None else env.max_steps

        def action_fn(e: BaseEnv, obs: np.ndarray) -> np.ndarray:
            return np.asarray(expert_fn(e), dtype=np.float32)

        self._rollout(env, action_fn, n_episodes, max_steps, seeds)
        return self.get_dataset()

    # ------------------------------------------------------------------
    # Dataset access
    # ------------------------------------------------------------------
    def get_dataset(self) -> Dict[str, np.ndarray]:
        """Return the collected transitions as stacked numpy arrays.

        Returns
        -------
        dict
            Keys: ``observations``, ``actions``, ``next_observations``,
            ``rewards`` ``(N,)`` float32, ``dones`` ``(N,)`` bool.
        """
        return {
            "observations": np.array(self._data["observations"]),
            "actions": np.array(self._data["actions"], dtype=np.float32),
            "next_observations": np.array(self._data["next_observations"]),
            "rewards": np.array(self._data["rewards"], dtype=np.float32),
            "dones": np.array(self._data["dones"], dtype=bool),
        }

    def __len__(self) -> int:
        return len(self._data["observations"])

    # ------------------------------------------------------------------
    # Batch sampling
    # ------------------------------------------------------------------
    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a random mini-batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample (with replacement).

        Returns
        -------
        dict
            Batch dictionary with the same keys as :meth:`get_dataset`.
        """
        n = len(self)
        if n == 0:
            raise RuntimeError("No transitions collected yet.")
        idx = np.random.randint(0, n, size=batch_size)
        dataset = self.get_dataset()
        return {k: v[idx] for k, v in dataset.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the collected dataset to a NumPy ``.npz`` archive.

        Parameters
        ----------
        path : str
            Destination file path.  A ``.npz`` extension is appended if not
            present.
        """
        if not path.endswith(".npz"):
            path = path + ".npz"
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        dataset = self.get_dataset()
        np.savez(path, **dataset)

    def load(self, path: str) -> Dict[str, np.ndarray]:
        """Load a dataset from a NumPy ``.npz`` archive.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file.

        Returns
        -------
        dict
            The loaded dataset dictionary.
        """
        if not path.endswith(".npz") and not os.path.exists(path):
            path = path + ".npz"
        with np.load(path, allow_pickle=True) as data:
            dataset = {k: data[k] for k in data.files}
        # Restore internal lists so that further collection can append.
        for key in ("observations", "actions", "next_observations",
                    "rewards", "dones"):
            if key in dataset:
                self._data[key] = list(dataset[key])
        return dataset
