"""MetaWorld wrapper (optional).

Wraps a subset of `MetaWorld <https://github.com/Farama-Foundation/Metaworld>`_
manipulation tasks into the :class:`benchmarks.base_env.BaseEnv` interface so
that all three controller families (MPC, VLA, Diffusion) can be evaluated on
realistic robotic manipulation benchmarks.

MetaWorld is an *optional* dependency.  If it is not installed, instantiating
:class:`MetaWorldWrapper` raises an informative :class:`ImportError`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .base_env import BaseEnv, EnvSpec

__all__ = ["MetaWorldWrapper", "SUPPORTED_METAWORLD_TASKS"]


SUPPORTED_METAWORLD_TASKS: Tuple[str, ...] = (
    "reach",
    "push",
    "pick-place",
    "drawer-open",
)


def _import_metaworld():
    """Lazily import metaworld and return ``(metaworld_module, ML1)``."""
    try:
        from metaworld import MetaWorldEnv  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "MetaWorld is not installed.  Install it with:\n"
            "  pip install git+https://github.com/Farama-Foundation/Metaworld.git\n"
            "or  pip install metaworld\n"
            f"(original error: {exc})"
        ) from exc
    import metaworld as mw
    from metaworld import ML1
    return mw, ML1


class MetaWorldWrapper(BaseEnv):
    """Wrap a MetaWorld task into the :class:`BaseEnv` interface.

    Parameters
    ----------
    task_name : str
        One of :data:`SUPPORTED_METAWORLD_TASKS`.
    seed : int, optional
        Random seed.
    max_steps : int
        Episode horizon.
    image_size : int
        Rendered image side length.
    render_mode : str
        ``"rgb_array"`` for image observations.
    """

    def __init__(
        self,
        task_name: str = "reach",
        seed: int | None = None,
        max_steps: int = 200,
        image_size: int = 96,
        camera_name: str = "corner3",
    ) -> None:
        super().__init__(seed=seed)
        if task_name not in SUPPORTED_METAWORLD_TASKS:
            raise ValueError(
                f"Unsupported task '{task_name}'. "
                f"Choose from {SUPPORTED_METAWORLD_TASKS}."
            )
        self.task_name = task_name
        self._max_steps = int(max_steps)
        self.image_size = int(image_size)
        self.camera_name = camera_name

        mw, ML1 = _import_metaworld()
        self._ml1 = ML1(task_name, seed=seed if seed is not None else 0)
        self._env = self._ml1.train_classes[task_name]()
        self._env._set_task_called = True
        self._env._freeze_rand_vec = False
        self._env.set_task(self._ml1.train_tasks[0])

        # Infer spaces from the underlying env.
        obs_low = self._env.observation_space.low
        obs_high = self._env.observation_space.high
        act_low = self._env.action_space.low
        act_high = self._env.action_space.high
        self._observation_space = EnvSpec(
            shape=obs_low.shape, low=obs_low, high=obs_high, dtype=np.float32
        )
        self._action_space = EnvSpec(
            shape=act_low.shape, low=act_low, high=act_high, dtype=np.float32
        )
        self._last_obs: np.ndarray = np.zeros(obs_low.shape, dtype=np.float32)
        self._last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        self._reseed(seed)
        self._reset_step_count()
        obs, info = self._env.reset()
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._last_info = dict(info) if isinstance(info, dict) else {}
        return self.get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self._action_space.shape)
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._last_info = dict(info) if isinstance(info, dict) else {}
        self._increment_step()
        done = bool(terminated) or bool(truncated) or self._step_count >= self._max_steps
        info = dict(self._last_info)
        info["step"] = self._step_count
        return self.get_observation(), float(reward), done, info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        return self._last_obs.astype(np.float32).copy()

    def get_observation(self) -> np.ndarray:
        return self._last_obs.astype(np.float32).copy()

    def get_image(self) -> np.ndarray:
        try:
            img = self._env.render_camera(self.camera_name)
            if img is None:
                img = self._env.render()
            img = np.asarray(img, dtype=np.uint8)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            return self._resize_image(img, self.image_size)
        except Exception:
            # Fallback: blank image.
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def get_language_instruction(self) -> str:
        return {
            "reach": "Reach to the target position with the robot end-effector.",
            "push": "Push the puck to the target position.",
            "pick-place": "Pick up the object and place it at the target.",
            "drawer-open": "Open the drawer.",
        }.get(self.task_name, f"Perform the {self.task_name} task.")

    # ------------------------------------------------------------------
    # Termination predicates
    # ------------------------------------------------------------------
    def is_success(self) -> bool:
        return bool(self._last_info.get("success", 0.0) > 0.5)

    def is_collision(self) -> bool:
        # MetaWorld does not expose a collision flag; treat failure as non-collision.
        return False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
        row_idx = np.linspace(0, h - 1, size).astype(int)
        col_idx = np.linspace(0, w - 1, size).astype(int)
        return img[np.ix_(row_idx, col_idx)]

    # ------------------------------------------------------------------
    # Expert demonstrations
    # ------------------------------------------------------------------
    def generate_expert_demonstrations(
        self, n_demos: int, max_steps: int | None = None
    ) -> Dict[str, np.ndarray]:
        """Collect demonstrations using MetaWorld's built-in scripted policy."""
        max_steps = max_steps or self._max_steps
        policy = self._ml1.train_classes[self.task_name]().__class__  # noqa
        try:
            from metaworld.policies import (
                SawyerReachV2Policy,
                SawyerPushV2Policy,
                SawyerPickPlaceV2Policy,
                SawyerDrawerOpenV2Policy,
            )
            policy_map = {
                "reach": SawyerReachV2Policy,
                "push": SawyerPushV2Policy,
                "pick-place": SawyerPickPlaceV2Policy,
                "drawer-open": SawyerDrawerOpenV2Policy,
            }
            expert = policy_map[self.task_name]()
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Could not import MetaWorld scripted policies. "
                "Ensure a compatible MetaWorld version is installed."
            ) from exc

        all_obs: List[np.ndarray] = []
        all_act: List[np.ndarray] = []
        all_next: List[np.ndarray] = []
        all_rew: List[float] = []
        all_done: List[bool] = []
        for i in range(n_demos):
            self.reset(seed=self._seed + i if self._seed is not None else None)
            for _ in range(max_steps):
                obs = self.get_observation().copy()
                action = expert.get_action(obs, self._env)
                action = np.asarray(action, dtype=np.float32)
                next_obs, reward, done, info = self.step(action)
                all_obs.append(obs)
                all_act.append(action)
                all_next.append(next_obs.copy())
                all_rew.append(reward)
                all_done.append(done)
                if done:
                    break
        return {
            "observations": np.array(all_obs),
            "actions": np.array(all_act),
            "next_observations": np.array(all_next),
            "rewards": np.array(all_rew, dtype=np.float32),
            "dones": np.array(all_done, dtype=bool),
        }
