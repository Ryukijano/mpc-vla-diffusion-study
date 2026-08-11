"""2-D / 3-D reaching benchmark environment.

A point mass must reach a target position while optionally avoiding circular
(2-D) or spherical (3-D) obstacles.  The dynamics are a double integrator,
matching the MPC baselines in ``mpc_baselines_repo`` so that the same dynamics
model can be shared between the environment and the MPC controller.

State
-----
2-D: ``[x, y, vx, vy]``   |   3-D: ``[x, y, z, vx, vy, vz]``

Action
------
2-D: ``[ax, ay]``         |   3-D: ``[ax, ay, az]``

Success
-------
Euclidean distance to target < ``success_threshold``.

Collision
---------
Agent position lies inside any obstacle (radius check).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .base_env import BaseEnv, EnvSpec

__all__ = ["ReachingEnv", "Obstacle"]


class Obstacle:
    """A circular (2-D) or spherical (3-D) obstacle.

    Parameters
    ----------
    center : array-like
        Obstacle centre ``(dim,)``.
    radius : float
        Obstacle radius.
    """

    def __init__(self, center: Sequence[float], radius: float) -> None:
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def contains(self, point: np.ndarray) -> bool:
        return float(np.linalg.norm(point - self.center)) <= self.radius

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Obstacle(center={self.center}, radius={self.radius})"


class ReachingEnv(BaseEnv):
    """Point-mass reaching task with optional obstacles.

    Parameters
    ----------
    dim : int
        2 or 3 -- workspace dimensionality.
    dt : float
        Integration time step.
    success_threshold : float
        Distance to target below which the episode is a success.
    max_steps : int
        Episode horizon.
    image_size : int
        Rendered image side length.
    obstacles : list of :class:`Obstacle`, optional
        Obstacles to avoid.
    workspace : float
        Half-extent of the (symmetric) workspace, e.g. ``5`` -> ``[-5, 5]``.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        dim: int = 2,
        dt: float = 0.05,
        success_threshold: float = 0.1,
        max_steps: int = 200,
        image_size: int = 96,
        obstacles: Sequence[Obstacle] | None = None,
        workspace: float = 5.0,
        seed: int | None = None,
    ) -> None:
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3.")
        super().__init__(seed=seed)
        self.dim = int(dim)
        self.dt = float(dt)
        self.success_threshold = float(success_threshold)
        self._max_steps = int(max_steps)
        self.image_size = int(image_size)
        self.workspace = float(workspace)
        self.obstacles: List[Obstacle] = list(obstacles) if obstacles else []

        self._state = np.zeros(2 * dim, dtype=np.float64)  # [pos, vel]
        self._target = np.zeros(dim, dtype=np.float64)
        self._start = np.zeros(dim, dtype=np.float64)

        self._observation_space = EnvSpec(
            shape=(2 * dim,), low=-np.inf, high=np.inf, dtype=np.float32
        )
        self._action_space = EnvSpec(
            shape=(dim,), low=-1.0, high=1.0, dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        self._reseed(seed)
        self._reset_step_count()
        self._start = self._rng.uniform(-self.workspace, self.workspace, size=self.dim)
        self._target = self._rng.uniform(-self.workspace, self.workspace, size=self.dim)
        # Ensure target is not on top of start.
        while np.linalg.norm(self._target - self._start) < 1.0:
            self._target = self._rng.uniform(-self.workspace, self.workspace, size=self.dim)
        self._state = np.concatenate([self._start, np.zeros(self.dim)])
        # Ensure start is not inside an obstacle.
        for _ in range(50):
            if not any(o.contains(self._start) for o in self.obstacles):
                break
            self._start = self._rng.uniform(-self.workspace, self.workspace, size=self.dim)
            self._state = np.concatenate([self._start, np.zeros(self.dim)])
        return self.get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=float).reshape(self.dim)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Double-integrator dynamics: x' = x + v*dt + 0.5*a*dt^2, v' = v + a*dt.
        pos = self._state[: self.dim]
        vel = self._state[self.dim :]
        new_vel = vel + action * self.dt
        new_pos = pos + vel * self.dt + 0.5 * action * self.dt ** 2
        # Workspace bounds (reflect velocity).
        for i in range(self.dim):
            if new_pos[i] < -self.workspace:
                new_pos[i] = -self.workspace
                new_vel[i] = -abs(new_vel[i]) * 0.5
            elif new_pos[i] > self.workspace:
                new_pos[i] = self.workspace
                new_vel[i] = -abs(new_vel[i]) * 0.5
        self._state = np.concatenate([new_pos, new_vel])
        self._increment_step()
        dist = float(np.linalg.norm(new_pos - self._target))
        reward = -dist - 0.01 * float(np.linalg.norm(action))
        if self.is_success():
            reward += 10.0
        if self.is_collision():
            reward -= 5.0
        info: Dict[str, Any] = {
            "distance": dist,
            "step": self._step_count,
            "target": self._target.copy(),
        }
        done = self.done
        return self.get_observation(), float(reward), done, info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        """Full state vector ``[pos, vel]`` (for MPC). Target appended via info."""
        return self._state.astype(np.float32).copy()

    def get_observation(self) -> np.ndarray:
        return self._state.astype(np.float32).copy()

    def get_image(self) -> np.ndarray:
        return self._render_image()

    def get_language_instruction(self) -> str:
        axis = "plane" if self.dim == 2 else "space"
        return f"Reach the green target in the {axis} while avoiding obstacles."

    def get_target(self) -> np.ndarray:
        """Return the current target position ``(dim,)``."""
        return self._target.copy()

    # ------------------------------------------------------------------
    # Termination predicates
    # ------------------------------------------------------------------
    def is_success(self) -> bool:
        pos = self._state[: self.dim]
        return float(np.linalg.norm(pos - self._target)) < self.success_threshold

    def is_collision(self) -> bool:
        pos = self._state[: self.dim]
        return any(o.contains(pos) for o in self.obstacles)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_image(self) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(self.image_size / 32, self.image_size / 32), dpi=32)
        lim = self.workspace * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axis("off")
        # Obstacles.
        for o in self.obstacles:
            ax.add_patch(Circle(o.center[:2], o.radius, color="black", alpha=0.7))
        # Target.
        ax.add_patch(Circle(self._target[:2], self.success_threshold,
                            color="green", alpha=0.6))
        # Agent.
        ax.add_patch(Circle(self._state[:2], 0.08, color="red", alpha=0.9))
        # Trajectory velocity arrow.
        vel = self._state[self.dim : self.dim + 2]
        if np.linalg.norm(vel) > 1e-3:
            ax.annotate(
                "",
                xy=self._state[:2] + vel * 0.3,
                xytext=self._state[:2],
                arrowprops=dict(arrowstyle="->", color="blue"),
            )
        fig.canvas.draw()
        # Use buffer_rgba() (works across matplotlib versions) and drop alpha.
        img = np.asarray(fig.canvas.buffer_rgba())
        img = img[..., :3].copy()  # (H, W, 3) uint8
        plt.close(fig)
        return self._resize_image(img, self.image_size)

    @staticmethod
    def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
        h, w, _ = img.shape
        row_idx = np.linspace(0, h - 1, size).astype(int)
        col_idx = np.linspace(0, w - 1, size).astype(int)
        return img[np.ix_(row_idx, col_idx)]

    # ------------------------------------------------------------------
    # Expert demonstrations
    # ------------------------------------------------------------------
    def generate_expert_demonstrations(
        self, n_demos: int, max_steps: int | None = None
    ) -> Dict[str, np.ndarray]:
        """Generate expert demonstrations using a straight-line (PD) controller.

        The expert applies a proportional control law towards the target with
        velocity damping.  If obstacles are present, a simple potential-field
        repulsion term is added.

        Returns
        -------
        dict
            Standard demonstration dict with ``observations``, ``actions``,
            ``next_observations``, ``rewards``, ``dones``.
        """
        max_steps = max_steps or self._max_steps
        all_obs: List[np.ndarray] = []
        all_act: List[np.ndarray] = []
        all_next: List[np.ndarray] = []
        all_rew: List[float] = []
        all_done: List[bool] = []
        for i in range(n_demos):
            self.reset(seed=self._seed + i if self._seed is not None else None)
            for _ in range(max_steps):
                obs = self.get_observation().copy()
                pos = self._state[: self.dim]
                vel = self._state[self.dim :]
                # PD towards target.
                err = self._target - pos
                action = 5.0 * err - 2.0 * vel
                # Obstacle repulsion.
                for o in self.obstacles:
                    d = pos - o.center[: self.dim]
                    dist = np.linalg.norm(d)
                    if dist < o.radius * 2.0 and dist > 1e-6:
                        action += (d / dist) * (o.radius * 2.0 - dist) * 10.0
                action = np.clip(action, -1.0, 1.0)
                next_obs, reward, done, _ = self.step(action)
                all_obs.append(obs)
                all_act.append(action.astype(np.float32))
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
