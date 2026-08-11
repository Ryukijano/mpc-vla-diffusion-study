"""PushT benchmark environment.

This is a re-implementation of the canonical *PushT* task from the Diffusion
Policy paper (Chi et al., 2023).  A point agent pushes a T-shaped block on a
2-D plane towards a target pose.  The environment is fully differentiable in
numpy and renders a top-down RGB image suitable for image-based policies
(VLA / diffusion).

State
-----
``[block_x, block_y, block_angle, agent_x, agent_y]``

Action
------
``[agent_dx, agent_dy]`` -- the agent moves by this delta each step.  When the
agent overlaps the block it imparts a push proportional to the overlap.

Success
-------
The IoU (intersection-over-union) between the T-block and the target T-shape
exceeds ``success_iou_threshold`` (default 0.6).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .base_env import BaseEnv, EnvSpec

__all__ = ["PushTEnv"]


# ---------------------------------------------------------------------------
# T-block geometry helpers
# ---------------------------------------------------------------------------
_T_POINTS: np.ndarray = np.array(
    [
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.0],
        [0.25, 0.0],
        [0.25, 0.5],
        [-0.25, 0.5],
        [-0.25, 0.0],
        [-0.5, 0.0],
    ],
    dtype=float,
)
"""Unit T-shape polygon vertices (will be scaled by ``block_size``)."""


def _transform_points(points: np.ndarray, x: float, y: float, angle: float) -> np.ndarray:
    """Rotate ``points`` by ``angle`` (rad) about origin then translate by (x, y)."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T + np.array([x, y])


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of a polygon ``(N, 2)``."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Approximate intersection area of two convex-ish polygons via a grid mask.

    The T-shape is non-convex so we use a rasterisation approach which is robust
    and fast enough for the small image sizes used here.
    """
    all_pts = np.vstack([poly_a, poly_b])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    # Use a coarse grid (200x200) for speed; ~1cm resolution.
    res = 200
    xs = np.linspace(mins[0], maxs[0], res)
    ys = np.linspace(mins[1], maxs[1], res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    mask_a = _points_in_polygon(pts, poly_a).reshape(res, res)
    mask_b = _points_in_polygon(pts, poly_b).reshape(res, res)
    cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
    return float(np.sum(mask_a & mask_b) * cell_area)


def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised point-in-polygon test (ray casting) for a single polygon."""
    n = poly.shape[0]
    inside = np.zeros(pts.shape[0], dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > pts[:, 1]) != (yj > pts[:, 1])) & (
            pts[:, 0]
            < (xj - xi) * (pts[:, 1] - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= cond
        j = i
    return inside


class PushTEnv(BaseEnv):
    """PushT pushing benchmark.

    Parameters
    ----------
    block_size : float
        Side length of the T-block square segments (workspace units).
    agent_radius : float
        Radius of the circular agent.
    success_iou_threshold : float
        IoU threshold for success.
    max_steps : int
        Episode length.
    image_size : int
        Side length of the rendered RGB image.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        block_size: float = 1.0,
        agent_radius: float = 0.15,
        success_iou_threshold: float = 0.6,
        max_steps: int = 200,
        image_size: int = 96,
        workspace: float = 10.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.block_size = float(block_size)
        self.agent_radius = float(agent_radius)
        self.success_iou_threshold = float(success_iou_threshold)
        self._max_steps = int(max_steps)
        self.image_size = int(image_size)
        self.workspace = float(workspace)

        self._block = np.array([0.0, 0.0, 0.0])  # x, y, angle
        self._target = np.array([0.0, 0.0, 0.0])  # x, y, angle
        self._agent = np.array([0.0, 0.0])  # x, y

        self._observation_space = EnvSpec(
            shape=(5,), low=-self.workspace, high=self.workspace, dtype=np.float32
        )
        self._action_space = EnvSpec(
            shape=(2,), low=-1.0, high=1.0, dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _block_polygon(self, pose: np.ndarray | None = None) -> np.ndarray:
        pose = self._block if pose is None else pose
        pts = _T_POINTS * self.block_size
        return _transform_points(pts, pose[0], pose[1], pose[2])

    def _target_polygon(self) -> np.ndarray:
        pts = _T_POINTS * self.block_size
        return _transform_points(pts, self._target[0], self._target[1], self._target[2])

    def _block_iou(self) -> float:
        bp = self._block_polygon()
        tp = self._target_polygon()
        inter = _polygon_intersection_area(bp, tp)
        union = _polygon_area(bp) + _polygon_area(tp) - inter
        return inter / (union + 1e-9)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        self._reseed(seed)
        self._reset_step_count()
        # Random block pose in left half, target in right half.
        self._block = np.array(
            [
                self._rng.uniform(-self.workspace / 2, 0.0),
                self._rng.uniform(-self.workspace / 2, self.workspace / 2),
                self._rng.uniform(-np.pi, np.pi),
            ]
        )
        self._target = np.array(
            [
                self._rng.uniform(0.0, self.workspace / 2),
                self._rng.uniform(-self.workspace / 2, self.workspace / 2),
                self._rng.uniform(-np.pi, np.pi),
            ]
        )
        # Agent starts near the block.
        self._agent = self._block[:2] + self._rng.uniform(-1.0, 1.0, size=2)
        self._agent = np.clip(
            self._agent, -self.workspace / 2, self.workspace / 2
        )
        return self.get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=float).reshape(2)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Move agent.
        self._agent = self._agent + action
        self._agent = np.clip(
            self._agent, -self.workspace / 2, self.workspace / 2
        )
        # Push interaction: if agent overlaps block, move block towards agent.
        dx = self._agent[0] - self._block[0]
        dy = self._agent[1] - self._block[1]
        dist = np.hypot(dx, dy)
        contact = self.agent_radius + self.block_size * 0.5
        if dist < contact and dist > 1e-6:
            # Push direction from block to agent (agent pushes block away).
            push = (contact - dist)
            self._block[0] -= (dx / dist) * push * 0.5
            self._block[1] -= (dy / dist) * push * 0.5
            # Small angular perturbation from off-centre pushes.
            self._block[2] += self._rng.uniform(-0.05, 0.05)
        self._increment_step()
        iou = self._block_iou()
        reward = iou - 0.01 * np.linalg.norm(action)
        info: Dict[str, Any] = {"iou": iou, "step": self._step_count}
        done = self.done
        return self.get_observation(), float(reward), done, info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        return np.array(
            [self._block[0], self._block[1], self._block[2],
             self._agent[0], self._agent[1]],
            dtype=np.float32,
        )

    def get_observation(self) -> np.ndarray:
        return self.get_state().astype(np.float32)

    def get_image(self) -> np.ndarray:
        return self._render_image()

    def get_language_instruction(self) -> str:
        return "Push the T-shaped block onto the target T outline."

    # ------------------------------------------------------------------
    # Termination predicates
    # ------------------------------------------------------------------
    def is_success(self) -> bool:
        return self._block_iou() >= self.success_iou_threshold

    def is_collision(self) -> bool:
        # PushT has no hard obstacles; agent leaving workspace is clipped.
        return False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_image(self) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(self.image_size / 32, self.image_size / 32), dpi=32)
        ax.set_xlim(-self.workspace / 2, self.workspace / 2)
        ax.set_ylim(-self.workspace / 2, self.workspace / 2)
        ax.set_aspect("equal")
        ax.axis("off")
        # Target outline.
        tp = MplPolygon(self._target_polygon(), closed=True, fill=True,
                        facecolor="#90ee90", edgecolor="green", alpha=0.4, linewidth=1)
        ax.add_patch(tp)
        # Block.
        bp = MplPolygon(self._block_polygon(), closed=True, fill=True,
                        facecolor="#4682b4", edgecolor="blue", alpha=0.8, linewidth=1)
        ax.add_patch(bp)
        # Agent.
        ax.add_patch(Circle(self._agent, self.agent_radius, color="red", alpha=0.9))
        fig.canvas.draw()
        # Use buffer_rgba() (works across matplotlib versions) and drop alpha.
        img = np.asarray(fig.canvas.buffer_rgba())
        img = img[..., :3].copy()  # (H, W, 3) uint8
        plt.close(fig)
        # Resize to image_size via simple crop/pad.
        return self._resize_image(img, self.image_size)

    @staticmethod
    def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
        """Nearest-neighbour resize to ``(size, size, 3)``."""
        h, w, _ = img.shape
        row_idx = (np.linspace(0, h - 1, size)).astype(int)
        col_idx = (np.linspace(0, w - 1, size)).astype(int)
        return img[np.ix_(row_idx, col_idx)]

    # ------------------------------------------------------------------
    # Expert demonstrations
    # ------------------------------------------------------------------
    def generate_expert_demonstrations(
        self, n_demos: int, max_steps: int | None = None
    ) -> Dict[str, np.ndarray]:
        """Generate expert demonstrations by pushing the block towards the target.

        The expert moves the agent to the side of the block opposite the target
        and pushes directly towards the target centre.

        Returns
        -------
        dict
            Keys: ``observations`` ``(T, obs_dim)``, ``actions`` ``(T, 2)``,
            ``next_observations`` ``(T, obs_dim)``, ``rewards`` ``(T,)``,
            ``dones`` ``(T,)``.
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
                # Expert: position agent behind block relative to target, push.
                block_xy = self._block[:2]
                target_xy = self._target[:2]
                dir_to_target = target_xy - block_xy
                d = np.linalg.norm(dir_to_target)
                if d < 1e-3:
                    action = np.zeros(2)
                else:
                    # Desired agent position: behind block (opposite target).
                    desired_agent = block_xy - (dir_to_target / d) * (self.block_size * 0.6)
                    action = desired_agent - self._agent
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
