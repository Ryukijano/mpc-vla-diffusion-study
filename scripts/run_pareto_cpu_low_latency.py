#!/usr/bin/env python3
"""EXP-004-CPU: Low-latency Pareto sweep of CPU-friendly controllers.

Runs a fast, CPU-only latency-performance sweep across:
  * Linear MPC (horizon 5, 10, 15, 20, 30)
  * Nonlinear MPC / iLQR (iters 1, 3, 5, 10, 15)
  * Minimal Iterative Policy (MIP) (iters 1, 2, 3, 5)
  * Regression Policy (hidden_dim 16, 32, 64)
  * Iterative Regression Policy (iters 1, 2, 3, 5)

Benchmarks: 2-D Reaching and PushT (fallback point-mass tasks).
This script uses only NumPy/PyTorch on the CPU and writes
  - pareto_data.csv
  - latency_table.csv
  - pareto_frontier.png
into the requested output directory.

Usage::

    conda run -n mpc_vla python scripts/run_pareto_cpu_low_latency.py \
        --seeds 0 1 --episodes 10 --output-dir results/exp004_cpu_low_latency
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
STUDY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY_ROOT not in sys.path:
    sys.path.insert(0, STUDY_ROOT)
for _d in [
    "mpc_baselines_repo",
    "mpc_baselines_repo/src",
    "diffusion_baselines",
    "benchmarks",
    "src",
]:
    _full = os.path.join(STUDY_ROOT, _d)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)

# CPU-only configuration
torch.set_num_threads(min(4, os.cpu_count() or 4))
if torch.cuda.is_available():
    # Ensure we do not accidentally allocate on the GPU.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_default_device("cpu")

# Imports that depend on the paths above
from src.utils.dynamics import PointMass2D
from src.utils.obstacles import CircleObstacle, is_in_collision
from src.linear_mpc import LinearMPC
from src.nonlinear_mpc import NonlinearMPC
from src.collision_free_mpc import CollisionFreeMPC, SDFWorld
from diffusion_baselines.regression_policy import RegressionPolicy
from diffusion_baselines.iterative_regression_policy import IterativeRegressionPolicy

# ---------------------------------------------------------------------------
# Fallback benchmarks (point-mass, MPC-compatible)
# ---------------------------------------------------------------------------


class _FallbackReaching:
    """Built-in 2D point-mass reaching benchmark with obstacles."""

    name = "Reaching"
    state_dim = 4
    action_dim = 2

    def __init__(self, seed: int = 0, cluttered: bool = False):
        self.dyn = PointMass2D(mass=1.0, dt=0.1)
        self.goal = np.array([4.0, 4.0, 0.0, 0.0])
        self.start = np.zeros(4)
        if cluttered:
            self.obstacles = [
                CircleObstacle([1.5, 1.5], 0.4),
                CircleObstacle([2.5, 2.5], 0.5),
                CircleObstacle([3.0, 1.0], 0.35),
                CircleObstacle([1.0, 3.0], 0.3),
                CircleObstacle([3.5, 3.0], 0.3),
            ]
            self.name = "Reaching (Cluttered)"
        else:
            self.obstacles = [
                CircleObstacle([2.0, 2.0], 0.5),
                CircleObstacle([3.0, 1.0], 0.4),
            ]
        self.world = SDFWorld(dim=2)
        for obs in self.obstacles:
            self.world.add_sphere(obs.center.tolist(), obs.radius)
        self.rng = np.random.default_rng(seed)
        self.goal_tol = 0.2
        self.max_steps = 80 if cluttered else 60

    def reset(self) -> np.ndarray:
        s = self.start + np.array(
            [
                self.rng.uniform(-0.3, 0.3),
                self.rng.uniform(-0.3, 0.3),
                0.0,
                0.0,
            ]
        )
        return s

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.dyn.dynamics(state, action)

    def is_success(self, state: np.ndarray) -> bool:
        return float(np.linalg.norm(state[:2] - self.goal[:2])) <= self.goal_tol

    def is_collision(self, state: np.ndarray) -> bool:
        return is_in_collision(state[:2], self.obstacles)

    def make_costs(self):
        goal = self.goal
        Q = np.diag([10.0, 10.0, 1.0, 1.0])
        R = np.diag([0.1, 0.1])
        Qf = np.diag([100.0, 100.0, 10.0, 10.0])

        def stage_cost(x, u, k=None):
            dx = x - goal
            return float(dx @ Q @ dx + u @ R @ u)

        def terminal_cost(x):
            dx = x - goal
            return float(dx @ Qf @ dx)

        return stage_cost, terminal_cost, Q, R, Qf


class _FallbackPushT:
    """Built-in PushT-style multi-modal reaching benchmark."""

    name = "PushT"
    state_dim = 4
    action_dim = 2

    def __init__(self, seed: int = 0):
        self.dyn = PointMass2D(mass=1.0, dt=0.1)
        self.goals = [
            np.array([4.0, 2.0, 0.0, 0.0]),
            np.array([2.0, 4.0, 0.0, 0.0]),
        ]
        self.goal = self.goals[0]
        self.start = np.zeros(4)
        self.obstacles = [CircleObstacle([2.0, 2.0], 0.3)]
        self.world = SDFWorld(dim=2)
        for obs in self.obstacles:
            self.world.add_sphere(obs.center.tolist(), obs.radius)
        self.rng = np.random.default_rng(seed)
        self.goal_tol = 0.25
        self.max_steps = 60

    def reset(self) -> np.ndarray:
        s = self.start + np.array(
            [
                self.rng.uniform(-0.3, 0.3),
                self.rng.uniform(-0.3, 0.3),
                0.0,
                0.0,
            ]
        )
        return s

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.dyn.dynamics(state, action)

    def is_success(self, state: np.ndarray) -> bool:
        for g in self.goals:
            if float(np.linalg.norm(state[:2] - g[:2])) <= self.goal_tol:
                return True
        return False

    def is_collision(self, state: np.ndarray) -> bool:
        return is_in_collision(state[:2], self.obstacles)

    def make_costs(self):
        goal = self.goal
        Q = np.diag([10.0, 10.0, 1.0, 1.0])
        R = np.diag([0.1, 0.1])
        Qf = np.diag([100.0, 100.0, 10.0, 10.0])

        def stage_cost(x, u, k=None):
            dx = x - goal
            return float(dx @ Q @ dx + u @ R @ u)

        def terminal_cost(x):
            costs = []
            for g in self.goals:
                dx = x - g
                costs.append(float(dx @ Qf @ dx))
            return min(costs)

        return stage_cost, terminal_cost, Q, R, Qf


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------
def make_benchmark(name: str, seed: int = 0):
    """Return a benchmark instance.

    Tries the real ``benchmarks`` package and falls back to the built-in
    point-mass tasks if it is unavailable.
    """
    name = name.lower()
    if name in ("reaching", "reaching_2d"):
        try:
            from benchmarks import ReachingEnv

            # Wrap the real reaching env so it has the same attributes as the
            # fallback (fixed goal, point-mass dynamics, etc.).
            return _WrappedReaching(ReachingEnv(dim=2, dt=0.05, max_steps=80, seed=seed))
        except Exception as exc:
            warnings.warn(f"Real Reaching unavailable ({exc}); using fallback.")
            return _FallbackReaching(seed=seed, cluttered=False)
    elif name in ("pusht", "pusht_2d"):
        try:
            from benchmarks import PushTEnv

            # Real PushT is a block-pushing contact task that is not directly
            # compatible with the point-mass MPC controllers in this CPU sweep,
            # so we use the point-mass fallback even when benchmarks is on path.
            warnings.warn("PushT uses the point-mass fallback for this CPU sweep.")
        except Exception:
            pass
        return _FallbackPushT(seed=seed)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


class _WrappedReaching:
    """Wrap ``benchmarks.ReachingEnv`` to expose a fixed-goal MPC interface."""

    name = "Reaching"

    def __init__(self, env):
        self.env = env
        self.dt = float(env.dt)
        self.dyn = PointMass2D(mass=1.0, dt=self.dt)
        self.state_dim = 4
        self.action_dim = 2
        self.goal_tol = float(env.success_threshold)
        self.max_steps = int(env._max_steps)
        self.obstacles = [
            CircleObstacle(o.center, o.radius) for o in env.obstacles
        ]
        self.world = SDFWorld(dim=2)
        for o in self.obstacles:
            self.world.add_sphere(o.center.tolist(), o.radius)
        self._state: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None
        self.reset()

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._state = np.asarray(obs, dtype=float).copy()
        self._target = np.asarray(self.env.get_target(), dtype=float).copy()
        # Fix a full-state goal for MPC and keep it constant across resets.
        self.goal = np.array(
            [self._target[0], self._target[1], 0.0, 0.0]
        )
        return self._state

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.env._state = np.asarray(state, dtype=np.float64).copy()
        obs, _, _, _ = self.env.step(action)
        return np.asarray(obs, dtype=float).copy()

    def is_success(self, state: Optional[np.ndarray] = None) -> bool:
        if state is not None:
            self.env._state = np.asarray(state, dtype=np.float64).copy()
        return self.env.is_success()

    def is_collision(self, state: Optional[np.ndarray] = None) -> bool:
        if state is not None:
            self.env._state = np.asarray(state, dtype=np.float64).copy()
        return self.env.is_collision()

    def make_costs(self):
        goal = self.goal
        Q = np.diag([10.0, 10.0, 1.0, 1.0])
        R = np.diag([0.1, 0.1])
        Qf = np.diag([100.0, 100.0, 10.0, 10.0])

        def stage_cost(x, u, k=None):
            dx = x - goal
            return float(dx @ Q @ dx + u @ R @ u)

        def terminal_cost(x):
            dx = x - goal
            return float(dx @ Qf @ dx)

        return stage_cost, terminal_cost, Q, R, Qf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_u_bounds(bench: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) control bounds for a benchmark."""
    if hasattr(bench, "u_bounds") and bench.u_bounds is not None:
        return bench.u_bounds
    if hasattr(bench, "action_space"):
        return (
            np.asarray(bench.action_space.low, dtype=float),
            np.asarray(bench.action_space.high, dtype=float),
        )
    return (
        -5.0 * np.ones(bench.action_dim, dtype=float),
        5.0 * np.ones(bench.action_dim, dtype=float),
    )


def collect_demonstrations(
    bench: Any,
    n_demos: int = 30,
    horizon: int = 16,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Collect (state, action_sequence) expert demos using CollisionFree MPC."""
    stage_cost, term_cost, _, _, _ = bench.make_costs()
    u_bounds = get_u_bounds(bench)

    mpc = CollisionFreeMPC(
        bench.dyn.dynamics,
        stage_cost,
        term_cost,
        bench.world,
        horizon=horizon,
        u_bounds=u_bounds,
        collision_weight=100.0,
        ilqr_iters=12,
    )

    rng = np.random.default_rng(seed)
    demos: List[Tuple[np.ndarray, np.ndarray]] = []
    attempts = 0
    while len(demos) < n_demos and attempts < n_demos * 3:
        attempts += 1
        s = bench.reset()
        u0 = np.zeros((horizon, bench.action_dim))
        try:
            U = mpc.solve(s, warm_start=u0)
            if np.isnan(U).any():
                continue
            demos.append((s.copy(), U.copy()))
        except Exception:
            continue

    print(f"  Collected {len(demos)}/{n_demos} expert demonstrations")
    return demos


def train_learned_controllers(
    demos: List[Tuple[np.ndarray, np.ndarray]],
    bench: Any,
    horizon: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """Train the Regression and MIP baselines on the collected demos."""
    state_dim = bench.state_dim
    action_dim = bench.action_dim
    u_bounds = get_u_bounds(bench)

    controllers: Dict[str, Any] = {}

    # --- Regression sweep (hidden_dim) ---
    for hidden_dim in (16, 32, 64):
        name = f"Regression (hidden={hidden_dim})"
        print(f"  [Train] {name} ...")
        rcp = RegressionPolicy(
            action_dim=action_dim,
            horizon=horizon,
            obs_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            device="cpu",
        )
        rcp.train(
            demos,
            num_epochs=30,
            batch_size=min(16, len(demos)),
            lr=1e-3,
            verbose=False,
        )
        rcp.net.eval()

        def rcp_policy(
            x: np.ndarray,
            _p: RegressionPolicy = rcp,
            _lo: np.ndarray = u_bounds[0],
            _hi: np.ndarray = u_bounds[1],
        ) -> np.ndarray:
            with torch.no_grad():
                a = _p.predict(x).cpu().numpy()
            a = np.asarray(a, dtype=float).reshape(-1, action_dim)
            return np.clip(a[0], _lo, _hi)

        controllers[name] = {
            "family": "Regression",
            "category": "Regression",
            "parameter": f"hidden={hidden_dim}",
            "policy_fn": rcp_policy,
            "controller": rcp,
        }

    # --- MIP sweep (iters) ---
    # Train a single 5-iteration model; individual conditions use subsets.
    print("  [Train] MIP (5 iters) ...")
    mip = IterativeRegressionPolicy(
        action_dim=action_dim,
        horizon=horizon,
        obs_dim=state_dim,
        num_iterations=5,
        hidden_dim=64,
        noise_std=0.1,
        device="cpu",
    )
    mip.train(
        demos,
        num_epochs=30,
        batch_size=min(16, len(demos)),
        lr=1e-3,
        verbose=False,
    )
    for net in mip.nets:
        net.eval()

    for iters in (1, 2, 3, 5):
        name = f"MIP (iters={iters})"

        def mip_policy(
            x: np.ndarray,
            _p: IterativeRegressionPolicy = mip,
            _it: int = iters,
            _lo: np.ndarray = u_bounds[0],
            _hi: np.ndarray = u_bounds[1],
        ) -> np.ndarray:
            a = sample_mip(_p, x, _it)
            a = np.asarray(a, dtype=float).reshape(-1, action_dim)
            return np.clip(a[0], _lo, _hi)

        controllers[name] = {
            "family": "MIP",
            "category": "MIP",
            "parameter": f"iters={iters}",
            "policy_fn": mip_policy,
            "controller": mip,
        }

    return controllers


@torch.no_grad()
def sample_mip(
    policy: IterativeRegressionPolicy,
    obs: np.ndarray,
    iters: int,
) -> np.ndarray:
    """Run the first ``iters`` MIP refinement steps and return the action seq."""
    obs_arr = np.asarray(obs, dtype=np.float32)
    single = obs_arr.ndim == 1
    if single:
        obs_arr = obs_arr[None, :]
    obs_t = torch.from_numpy(obs_arr).to(policy.device)
    bs = obs_t.shape[0]
    prev = torch.zeros(bs, policy.horizon, policy.action_dim, device=policy.device)

    for step in range(min(iters, policy.num_iterations)):
        pred = policy.nets[step](obs_t, prev)
        if step < iters - 1 and policy.noise_std > 0:
            prev = pred + policy.noise_std * torch.randn_like(pred)
        else:
            prev = pred

    if single:
        return prev[0].cpu().numpy()
    return prev.cpu().numpy()


# ---------------------------------------------------------------------------
# Latency profiling
# ---------------------------------------------------------------------------
def profile_latency(
    call_fn: Callable[[], Any],
    n_warmup: int = 10,
    n_timed: int = 100,
) -> Dict[str, float]:
    """Profile CPU inference latency with ``time.perf_counter_ns``."""
    for _ in range(n_warmup):
        call_fn()

    timings_ns: List[int] = []
    for _ in range(n_timed):
        t0 = time.perf_counter_ns()
        call_fn()
        t1 = time.perf_counter_ns()
        timings_ns.append(t1 - t0)

    timings_ms = np.asarray(timings_ns, dtype=np.float64) / 1e6
    mean_ms = float(np.mean(timings_ms))
    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(timings_ms)),
        "p50_ms": float(np.percentile(timings_ms, 50)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "p99_ms": float(np.percentile(timings_ms, 99)),
        "min_ms": float(np.min(timings_ms)),
        "max_ms": float(np.max(timings_ms)),
        "throughput_hz": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        "n_samples": n_timed,
    }


# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------
def simulate_rollout(
    bench: Any,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute one closed-loop episode and measure per-step latency."""
    if max_steps is None:
        max_steps = getattr(bench, "max_steps", 60)

    u_bounds = get_u_bounds(bench)

    s = bench.reset()
    path_length = 0.0
    prev_pos = s[:2].copy()
    success = False
    collision = False
    step_latencies: List[float] = []

    curr_state = s.copy()
    for t in range(max_steps):
        t0 = time.perf_counter_ns()
        act = policy_fn(curr_state)
        t1 = time.perf_counter_ns()
        step_latencies.append((t1 - t0) / 1e6)

        act = np.asarray(act, dtype=float).reshape(-1)
        if len(act) > bench.action_dim:
            act = act[: bench.action_dim]
        act = np.clip(act, u_bounds[0], u_bounds[1])

        curr_state = bench.step(curr_state, act)

        pos = curr_state[:2]
        path_length += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos.copy()

        if bench.is_collision(curr_state):
            collision = True
        if bench.is_success(curr_state):
            success = True
            break

    return {
        "success": 1.0 if success else 0.0,
        "collision": 1.0 if collision else 0.0,
        "path_length": path_length,
        "steps": t + 1,
        "mean_latency_ms": float(np.mean(step_latencies)) if step_latencies else 0.0,
    }


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------
def compute_pareto_dominance(
    conditions: List[Dict[str, Any]],
    latency_key: str = "latency_p50_ms",
    success_key: str = "success_rate_mean",
) -> List[Dict[str, Any]]:
    """Mark Pareto-optimal points (lower latency, higher success)."""
    n = len(conditions)
    for i in range(n):
        dom_count = 0
        l_i = conditions[i][latency_key]
        s_i = conditions[i][success_key]

        for j in range(n):
            if i == j:
                continue
            l_j = conditions[j][latency_key]
            s_j = conditions[j][success_key]

            if (l_j <= l_i and s_j >= s_i) and (l_j < l_i or s_j > s_i):
                dom_count += 1

        conditions[i]["dominance_count"] = dom_count
        conditions[i]["is_pareto_optimal"] = dom_count == 0

    return conditions


def compute_low_latency_pareto(
    records: List[Dict[str, Any]],
    latency_ms_threshold: float = 100.0,
) -> List[Dict[str, Any]]:
    """Compute Pareto dominance on the low-latency subset (< threshold)."""
    low = [r for r in records if r["latency_p50_ms"] < latency_ms_threshold]
    low = compute_pareto_dominance(low)
    low_map = {r["condition"]: r for r in low}
    for r in records:
        if r["condition"] in low_map:
            r["is_pareto_optimal_low_latency"] = low_map[r["condition"]][
                "is_pareto_optimal"
            ]
            r["dominance_count_low_latency"] = low_map[r["condition"]][
                "dominance_count"
            ]
        else:
            r["is_pareto_optimal_low_latency"] = False
            r["dominance_count_low_latency"] = -1
    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pareto_frontier(
    pareto_by_bench: Dict[str, List[Dict[str, Any]]],
    output_path: str,
    latency_ms_threshold: float = 100.0,
) -> None:
    """Plot the low-latency Pareto frontier for each benchmark."""
    benchmarks = list(pareto_by_bench.keys())
    n_benches = len(benchmarks)
    fig, axes = plt.subplots(
        1, n_benches, figsize=(7 * n_benches, 6), dpi=200, squeeze=False
    )

    family_styles = {
        "Linear MPC": {
            "color": "#1f77b4",
            "marker": "s",
            "label": "Linear MPC",
        },
        "Nonlinear MPC": {
            "color": "#2ca02c",
            "marker": "o",
            "label": "Nonlinear MPC (iLQR)",
        },
        "MIP": {
            "color": "#ff7f0e",
            "marker": "D",
            "label": "MIP",
        },
        "Regression": {
            "color": "#7f7f7f",
            "marker": "X",
            "label": "Regression",
        },
    }

    for ax_idx, bn in enumerate(benchmarks):
        ax = axes[0, ax_idx]
        conds = [c for c in pareto_by_bench[bn] if c["latency_p50_ms"] < latency_ms_threshold]

        for c in conds:
            fam = c["family"]
            style = family_styles.get(
                fam, {"color": "#333333", "marker": "o", "label": fam}
            )
            x = c["latency_p50_ms"]
            y = c["success_rate_mean"]
            is_pareto = c.get("is_pareto_optimal_low_latency", c["is_pareto_optimal"])
            size = 120 if is_pareto else 80
            edge_w = 2.0 if is_pareto else 0.8
            edge_c = "black" if is_pareto else "white"

            ax.scatter(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                s=size,
                edgecolors=edge_c,
                linewidths=edge_w,
                zorder=5 if is_pareto else 3,
                alpha=0.9,
            )

            if is_pareto:
                short_name = c["condition"].replace("MPC (", "(").replace("Policy", "")
                ax.annotate(
                    short_name,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold",
                    alpha=0.85,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                    ),
                )

        # Low-latency Pareto frontier line
        frontier = [c for c in conds if c.get("is_pareto_optimal_low_latency", c["is_pareto_optimal"])]
        frontier.sort(key=lambda p: p["latency_p50_ms"])
        if len(frontier) >= 2:
            fx = [p["latency_p50_ms"] for p in frontier]
            fy = [p["success_rate_mean"] for p in frontier]
            ax.plot(fx, fy, "--", color="#333333", linewidth=2.0, alpha=0.75, zorder=4)

        title_str = "2D Point-Mass Reaching" if "reach" in bn.lower() else "PushT Benchmark"
        ax.set_title(f"{title_str}\nLow-Latency Pareto Frontier", fontsize=12, fontweight="bold")
        ax.set_xlabel("p50 Inference Latency (ms)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Success Rate", fontsize=11, fontweight="bold")
        ax.set_xlim(0, latency_ms_threshold)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.35)

        # Legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="w",
                markerfacecolor=style["color"],
                markersize=9,
                label=style["label"],
            )
            for style in family_styles.values()
        ]
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle="--",
                linewidth=2.0,
                label="Pareto Frontier",
            )
        )
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5)

    fig.suptitle(
        f"EXP-004-CPU: Low-Latency Pareto Sweep (<{latency_ms_threshold:.0f} ms)",
        fontsize=14,
        fontweight="heavy",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved Pareto frontier figure to: {output_path}")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_pareto_cpu_low_latency(
    benchmarks: List[str],
    seeds: List[int],
    episodes: int,
    n_warmup: int,
    n_timed: int,
    horizon: int,
    n_demos: int,
    output_dir: str,
):
    """Run the CPU-only low-latency Pareto sweep."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print("EXP-004-CPU: CPU-only low-latency Pareto sweep")
    print("=" * 90)
    print(f"  Benchmarks:   {benchmarks}")
    print(f"  Seeds:        {seeds}")
    print(f"  Episodes:     {episodes} per condition per seed")
    print(f"  Profiling:    {n_warmup} warmup, {n_timed} timed calls")
    print(f"  Horizon:      {horizon}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Device:       CPU only")
    print("=" * 90 + "\n")

    master_pareto_data: List[Dict[str, Any]] = []
    master_latency_table: List[Dict[str, Any]] = []
    pareto_by_bench: Dict[str, List[Dict[str, Any]]] = {}

    for bench_name in benchmarks:
        print(f"\n{'─' * 80}")
        print(f"BENCHMARK: {bench_name.upper()}")
        print(f"{'─' * 80}")

        # --- Build the benchmark and collect demos once ---
        bench = make_benchmark(bench_name, seed=seeds[0])
        bench.u_bounds = get_u_bounds(bench)

        print("\n  [Data] Collecting expert demonstrations...")
        demos = collect_demonstrations(bench, n_demos=n_demos, horizon=horizon, seed=seeds[0])
        if len(demos) < 5:
            warnings.warn(f"Only {len(demos)} demos collected; skipping {bench_name}")
            continue

        print("\n  [Train] Training learned baselines...")
        learned = train_learned_controllers(demos, bench, horizon=horizon, seed=seeds[0])

        # --- Build all controllers ---
        sample_state = bench.reset().astype(np.float32)
        stage_cost, terminal_cost, Q, R, Qf = bench.make_costs()
        u_bounds = get_u_bounds(bench)

        # Linear MPC sweep
        A, B = bench.dyn.linearize(
            np.zeros(bench.state_dim), np.zeros(bench.action_dim)
        )
        linear_mpcs: Dict[int, LinearMPC] = {}
        for H in (5, 10, 15, 20, 30):
            lmpc = LinearMPC(A, B, Q, R, Qf, horizon=H, u_bounds=u_bounds)
            linear_mpcs[H] = lmpc

        conditions_dict: Dict[str, Dict[str, Any]] = {}

        # Linear MPC conditions
        for H in (5, 10, 15, 20, 30):
            c_name = f"Linear MPC (H={H})"
            ref = np.tile(bench.goal, (H + 1, 1))
            _m = linear_mpcs[H]
            conditions_dict[c_name] = {
                "family": "Linear MPC",
                "category": "Linear MPC",
                "parameter": f"H={H}",
                "profile_fn": lambda _m=_m, _s=sample_state, _r=ref: _m.solve(
                    _s, _r
                ).control,
                "policy_fn": lambda s, _m=_m, _r=ref: _m.solve(s, _r).control,
            }

        # Nonlinear MPC / iLQR sweep (one object per condition to avoid warm-start cross-talk)
        for it in (1, 3, 5, 10, 15):
            c_name = f"Nonlinear MPC (iters={it})"
            nmpc = NonlinearMPC(
                bench.dyn.dynamics,
                stage_cost,
                terminal_cost,
                horizon=horizon,
                u_bounds=u_bounds,
            )
            _it = it
            conditions_dict[c_name] = {
                "family": "Nonlinear MPC",
                "category": "Nonlinear MPC",
                "parameter": f"iters={it}",
                "profile_fn": lambda _m=nmpc, _s=sample_state, _it=_it: _m.solve(
                    _s, max_iter=_it
                )["action"],
                "policy_fn": lambda s, _m=nmpc, _it=_it: _m.solve(s, max_iter=_it)[
                    "action"
                ],
            }

        # Add learned baselines
        for c_name, spec in learned.items():
            _p = spec["controller"]
            _pf = spec["policy_fn"]
            conditions_dict[c_name] = {
                "family": spec["family"],
                "category": spec["category"],
                "parameter": spec["parameter"],
                "profile_fn": lambda _pf=_pf, _s=sample_state: _pf(_s),
                "policy_fn": _pf,
            }

        print(f"\n  [Latency] Profiling {len(conditions_dict)} conditions "
              f"({n_warmup} warmup, {n_timed} timed) ...")
        cond_latencies: Dict[str, Dict[str, float]] = {}
        for c_name, c_spec in conditions_dict.items():
            try:
                lat_res = profile_latency(
                    c_spec["profile_fn"],
                    n_warmup=n_warmup,
                    n_timed=n_timed,
                )
                cond_latencies[c_name] = lat_res
                print(
                    f"    {c_name:<30} -> p50: {lat_res['p50_ms']:>7.3f} ms | "
                    f"mean: {lat_res['mean_ms']:>7.3f} ms | "
                    f"p99: {lat_res['p99_ms']:>7.3f} ms"
                )
            except Exception as exc:
                print(f"    [WARN] {c_name} latency profiling failed: {exc}")
                cond_latencies[c_name] = {
                    "mean_ms": float("nan"),
                    "std_ms": 0.0,
                    "p50_ms": float("nan"),
                    "p95_ms": float("nan"),
                    "p99_ms": float("nan"),
                    "min_ms": float("nan"),
                    "max_ms": float("nan"),
                    "throughput_hz": 0.0,
                    "n_samples": n_timed,
                }

        # --- Closed-loop evaluation across seeds ---
        cond_evaluations: Dict[str, List[Dict[str, Any]]] = {}

        for seed in seeds:
            print(f"\n  [Eval] Seed {seed}: {episodes} episodes per condition")
            bench = make_benchmark(bench_name, seed=seed)
            bench.u_bounds = get_u_bounds(bench)

            for c_name, c_spec in conditions_dict.items():
                ep_results: List[Dict[str, Any]] = []
                for ep in range(episodes):
                    try:
                        metrics = simulate_rollout(
                            bench, c_spec["policy_fn"], max_steps=bench.max_steps
                        )
                        ep_results.append(metrics)
                    except Exception as exc:
                        print(f"      [warn] {c_name} episode {ep} failed: {exc}")
                        continue

                if not ep_results:
                    continue

                sr = float(np.mean([m["success"] for m in ep_results]))
                cr = float(np.mean([m["collision"] for m in ep_results]))
                pl = float(np.mean([m["path_length"] for m in ep_results]))
                cond_evaluations.setdefault(c_name, []).append(
                    {
                        "seed": seed,
                        "success_rate": sr,
                        "collision_rate": cr,
                        "path_length": pl,
                        "n_episodes": len(ep_results),
                        "mean_step_latency_ms": float(
                            np.mean([m["mean_latency_ms"] for m in ep_results])
                        ),
                    }
                )
                print(
                    f"    {c_name:<30} -> Success: {sr*100:>5.1f}% | "
                    f"Collisions: {cr*100:>4.1f}%"
                )

        # --- Aggregate records ---
        bench_records: List[Dict[str, Any]] = []
        for c_name, seed_list in cond_evaluations.items():
            if c_name not in cond_latencies:
                continue
            lat = cond_latencies[c_name]
            c_spec = conditions_dict[c_name]

            srs = [s["success_rate"] for s in seed_list]
            crs = [s["collision_rate"] for s in seed_list]
            pls = [s["path_length"] for s in seed_list]

            record: Dict[str, Any] = {
                "benchmark": bench_name,
                "family": c_spec["family"],
                "condition": c_name,
                "category": c_spec["category"],
                "parameter": c_spec["parameter"],
                "latency_mean_ms": lat["mean_ms"],
                "latency_std_ms": lat["std_ms"],
                "latency_p50_ms": lat["p50_ms"],
                "latency_p95_ms": lat["p95_ms"],
                "latency_p99_ms": lat["p99_ms"],
                "latency_min_ms": lat["min_ms"],
                "latency_max_ms": lat["max_ms"],
                "throughput_hz": lat["throughput_hz"],
                "success_rate_mean": float(np.mean(srs)),
                "success_rate_std": float(np.std(srs)),
                "collision_rate_mean": float(np.mean(crs)),
                "path_length_mean": float(np.mean(pls)),
                "n_seeds": len(seeds),
                "n_episodes": episodes,
            }
            bench_records.append(record)

        # --- Pareto dominance ---
        bench_records = compute_pareto_dominance(bench_records)
        bench_records = compute_low_latency_pareto(bench_records)
        pareto_by_bench[bench_name] = bench_records
        master_pareto_data.extend(bench_records)

        for r in bench_records:
            master_latency_table.append(
                {
                    "benchmark": r["benchmark"],
                    "family": r["family"],
                    "controller": r["condition"],
                    "parameter": r["parameter"],
                    "mean_ms": r["latency_mean_ms"],
                    "std_ms": r["latency_std_ms"],
                    "p50_ms": r["latency_p50_ms"],
                    "p95_ms": r["latency_p95_ms"],
                    "p99_ms": r["latency_p99_ms"],
                    "min_ms": r["latency_min_ms"],
                    "max_ms": r["latency_max_ms"],
                    "throughput_hz": r["throughput_hz"],
                    "n_eval_calls": n_timed,
                }
            )

    # -----------------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------------
    pareto_csv_path = os.path.join(output_dir, "pareto_data.csv")
    if master_pareto_data:
        fieldnames = list(master_pareto_data[0].keys())
        with open(pareto_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(master_pareto_data)
        print(f"\n[Data] Saved Pareto dataset to: {pareto_csv_path}")

    latency_csv_path = os.path.join(output_dir, "latency_table.csv")
    if master_latency_table:
        fieldnames = list(master_latency_table[0].keys())
        with open(latency_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(master_latency_table)
        print(f"[Data] Saved Latency table to: {latency_csv_path}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    pareto_png_path = os.path.join(output_dir, "pareto_frontier.png")
    if pareto_by_bench:
        plot_pareto_frontier(pareto_by_bench, pareto_png_path)

    # -----------------------------------------------------------------------
    # JSON summary
    # -----------------------------------------------------------------------
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "benchmarks": benchmarks,
                "seeds": seeds,
                "episodes": episodes,
                "n_warmup": n_warmup,
                "n_timed": n_timed,
                "horizon": horizon,
                "n_conditions": len(master_pareto_data),
                "pareto_data": master_pareto_data,
                "latency_table": master_latency_table,
            },
            f,
            indent=2,
        )
    print(f"[Data] Saved metrics summary to: {summary_path}")

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("EXP-004-CPU SUMMARY: LOW-LATENCY PARETO SWEEP")
    print("=" * 90)
    for bn in benchmarks:
        if bn not in pareto_by_bench:
            continue
        print(f"\nBenchmark: {bn.upper()}")
        header = (
            f"  {'Controller':<28} {'Family':<16} {'p50(ms)':>9} "
            f"{'Success':>8} {'±std':>6} {'Pareto*':>8} {'Low-Lat-Pareto':>14}"
        )
        print(header)
        print("  " + "-" * 90)
        conds = sorted(pareto_by_bench[bn], key=lambda x: x["latency_p50_ms"])
        for c in conds:
            pareto_str = "★" if c["is_pareto_optimal"] else " "
            low_str = "★" if c["is_pareto_optimal_low_latency"] else " "
            print(
                f"  {c['condition']:<28} {c['family']:<16} "
                f"{c['latency_p50_ms']:>9.3f} {c['success_rate_mean']*100:>7.1f}% "
                f"{c['success_rate_std']*100:>5.1f}% {pareto_str:>8} {low_str:>14}"
            )
    print("\n" + "=" * 90)

    return pareto_by_bench


def main():
    parser = argparse.ArgumentParser(
        description="EXP-004-CPU: Low-latency CPU Pareto sweep."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["all", "reaching", "pusht"],
        help="Benchmark to run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Random seeds for evaluation",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per seed",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Warmup calls for latency profiling",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=100,
        help="Timed calls for latency profiling",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Action prediction horizon",
    )
    parser.add_argument(
        "--n-demos",
        type=int,
        default=30,
        help="Number of expert demonstrations to collect",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(STUDY_ROOT, "results", "exp004_cpu_low_latency"),
        help="Output directory",
    )
    args = parser.parse_args()

    benchmarks = (
        ["reaching", "pusht"] if args.benchmark == "all" else [args.benchmark]
    )

    run_pareto_cpu_low_latency(
        benchmarks=benchmarks,
        seeds=args.seeds,
        episodes=args.episodes,
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
        horizon=args.horizon,
        n_demos=args.n_demos,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
