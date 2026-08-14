#!/usr/bin/env python3
"""Main experiment runner for the MPC vs VLA vs Diffusion comparison study.

This script ties together all controller families (MPC, VLA, Diffusion) and
benchmarks, running a four-phase experimental pipeline:

  Phase 1: Collect demonstrations from MPC experts on each benchmark.
  Phase 2: Train learning-based controllers on the collected demos.
  Phase 3: Evaluate ALL controllers on ALL benchmarks.
  Phase 4: Collect metrics and generate comparison tables + plots.

It is designed to run on the DGX Spark (GB10 Grace Blackwell) with the
``mpc_vla`` conda environment (PyTorch 2.12.0.dev+cu128).

Usage::

    conda run -n mpc_vla python run_experiments.py --benchmark all --controllers all --seeds 0 1 2 --episodes 50
    conda run -n mpc_vla python run_experiments.py --benchmark pusht --controllers mpc diffusion --episodes 20
    conda run -n mpc_vla python run_experiments.py --quick   # fast smoke test
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

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make all module directories importable
# ---------------------------------------------------------------------------
STUDY_ROOT = os.path.dirname(os.path.abspath(__file__))

# Module directories (relative to study root)
_MODULE_DIRS = [
    "mpc_baselines_repo",          # MPC baselines (exists)
    "vla_baselines",               # VLA baselines (may not exist yet)
    "diffusion_baselines",         # Diffusion baselines (may not exist yet)
    "benchmarks",                  # Benchmarks (may not exist yet)
    "src",                         # Shared source (may contain future modules)
    "mpc_baselines_repo/src",      # MPC source root for `import src.*`
]
for _d in _MODULE_DIRS:
    _full = os.path.join(STUDY_ROOT, _d)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)

# Also add study root itself for config access
if STUDY_ROOT not in sys.path:
    sys.path.insert(0, STUDY_ROOT)


# ---------------------------------------------------------------------------
# Graceful import helpers
# ---------------------------------------------------------------------------
def _try_import(module_path: str, names: List[str]) -> Tuple[Optional[Any], List[str]]:
    """Attempt to import ``names`` from ``module_path``.

    Returns (module_or_None, list_of_successfully_imported_names).
    On failure, prints a warning and returns (None, []).
    """
    try:
        mod = __import__(module_path, fromlist=names)
        return mod, names
    except ImportError as exc:
        print(f"  [WARNING] Could not import {module_path}: {exc}")
        return None, []
    except Exception as exc:
        print(f"  [WARNING] Error importing {module_path}: {exc}")
        return None, []


# --- MPC baselines (from mpc_baselines_repo) --------------------------------
print("[imports] Loading MPC baselines...")
_MPC_OK = False
try:
    from src.utils import PointMass2D
    from src.utils.obstacles import CircleObstacle, is_in_collision
    from src.linear_mpc import LinearMPC
    from src.nonlinear_mpc import NonlinearMPC
    from src.collision_free_mpc import CollisionFreeMPC, SDFWorld
    from src.diffusion_warm_start import (
        DiffusionWarmStartMPC,
        SimpleDiffusionPolicy,
        MinimalIterativePolicy,
    )
    _MPC_OK = True
    print("  [OK] MPC baselines loaded (LinearMPC, NonlinearMPC, CollisionFreeMPC, "
          "DiffusionWarmStartMPC, SimpleDiffusionPolicy, MinimalIterativePolicy)")
except Exception as exc:
    print(f"  [WARNING] MPC baselines import failed: {exc}")
    traceback.print_exc()

# --- VLA baselines (may not exist yet) --------------------------------------
print("[imports] Loading VLA baselines...")
_VLA_OK = False
OpenVLAWrapper = None
SmallVLA = None
try:
    from vla_baselines import OpenVLAInference, SmallVLA  # type: ignore
    OpenVLAWrapper = OpenVLAInference  # expose under expected alias
    _VLA_OK = True
    print("  [OK] VLA baselines loaded (OpenVLAInference, SmallVLA)")
except ImportError:
    print("  [SKIP] vla_baselines not found — VLA controllers will be unavailable")
except Exception as exc:
    print(f"  [WARNING] VLA baselines import error: {exc}")

# --- Diffusion baselines (may not exist yet) --------------------------------
print("[imports] Loading diffusion baselines...")
_DIFFUSION_OK = False
DDPMPolicy = None
FlowMatchingPolicy = None
RegressionPolicy = None
IterativeRegressionPolicy = None
try:
    from diffusion_baselines import (  # type: ignore
        DDPMPolicy,
        FlowMatchingPolicy,
        RegressionPolicy,
        IterativeRegressionPolicy,
    )
    _DIFFUSION_OK = True
    print("  [OK] Diffusion baselines loaded (DDPMPolicy, FlowMatchingPolicy, "
          "RegressionPolicy, IterativeRegressionPolicy)")
except ImportError:
    print("  [SKIP] diffusion_baselines not found — standalone diffusion controllers "
          "will be unavailable (MPC-bundled diffusion still works)")
except Exception as exc:
    print(f"  [WARNING] Diffusion baselines import error: {exc}")

# --- Benchmarks (may not exist yet) -----------------------------------------
print("[imports] Loading benchmarks...")
_BENCH_OK = False
PushT = None
Reaching = None
DemonstrationCollector = None
Evaluator = None
try:
    from benchmarks import (  # type: ignore
        PushT,
        Reaching,
        DemonstrationCollector,
        Evaluator,
    )
    _BENCH_OK = True
    print("  [OK] Benchmarks loaded (PushT, Reaching, DemonstrationCollector, Evaluator)")
except ImportError:
    print("  [SKIP] benchmarks module not found — using built-in fallback benchmarks")
except Exception as exc:
    print(f"  [WARNING] Benchmarks import error: {exc}")


# ===========================================================================
# Built-in fallback benchmarks (used when benchmarks/ module is unavailable)
# ===========================================================================
class _FallbackReaching:
    """Built-in 2D point-mass reaching benchmark with obstacles.

    This mirrors the task in ``mpc_baselines_repo/scripts/run_comparison.py``
    and works with the MPC dynamics models.
    """

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
        """Return a (possibly perturbed) start state."""
        s = self.start + np.array([
            self.rng.uniform(-0.3, 0.3),
            self.rng.uniform(-0.3, 0.3),
            0.0, 0.0,
        ])
        return s

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.dyn.dynamics(state, action)

    def is_success(self, state: np.ndarray) -> bool:
        return float(np.linalg.norm(state[:2] - self.goal[:2])) <= self.goal_tol

    def is_collision(self, state: np.ndarray) -> bool:
        return is_in_collision(state[:2], self.obstacles)

    def make_costs(self):
        """Return (stage_cost_fn, terminal_cost_fn, Q, R, Qf)."""
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
    """Built-in PushT-style benchmark.

    A simplified T-block pushing task modelled as a 2D reaching task with
    a multi-modal goal structure (the T can be pushed from left or right,
    creating two solution modes).  This is a lightweight stand-in for the
    full PushT environment.
    """

    name = "PushT"
    state_dim = 4
    action_dim = 2

    def __init__(self, seed: int = 0):
        self.dyn = PointMass2D(mass=1.0, dt=0.1)
        # Two possible goal positions (multi-modal)
        self.goals = [
            np.array([4.0, 2.0, 0.0, 0.0]),
            np.array([2.0, 4.0, 0.0, 0.0]),
        ]
        self.goal = self.goals[0]  # primary goal
        self.start = np.zeros(4)
        self.obstacles = [
            CircleObstacle([2.0, 2.0], 0.3),
        ]
        self.world = SDFWorld(dim=2)
        for obs in self.obstacles:
            self.world.add_sphere(obs.center.tolist(), obs.radius)
        self.rng = np.random.default_rng(seed)
        self.goal_tol = 0.25
        self.max_steps = 60

    def reset(self) -> np.ndarray:
        s = self.start + np.array([
            self.rng.uniform(-0.3, 0.3),
            self.rng.uniform(-0.3, 0.3),
            0.0, 0.0,
        ])
        return s

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.dyn.dynamics(state, action)

    def is_success(self, state: np.ndarray) -> bool:
        # Success if close to EITHER goal (multi-modal)
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
            # Minimum distance to any goal (multi-modal terminal cost)
            costs = []
            for g in self.goals:
                dx = x - g
                costs.append(float(dx @ Qf @ dx))
            return min(costs)

        return stage_cost, terminal_cost, Q, R, Qf


def _make_benchmark(name: str, seed: int = 0):
    """Create a benchmark instance by name, using standard point-mass / MPC dynamics."""
    if name == "pusht":
        return _FallbackPushT(seed=seed)
    elif name == "reaching":
        return _FallbackReaching(seed=seed, cluttered=False)
    elif name == "reaching_cluttered":
        return _FallbackReaching(seed=seed, cluttered=True)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


# ===========================================================================
# Phase 1: Demonstration collection
# ===========================================================================
def collect_demonstrations(bench, n_demos: int, horizon: int,
                           u_bounds: Tuple[np.ndarray, np.ndarray],
                           seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Collect expert demonstrations using Collision-Free MPC.

    Returns a list of (state, action_sequence) pairs where action_sequence
    has shape (horizon, action_dim).
    """
    if not _MPC_OK:
        print("  [WARNING] MPC baselines not available — cannot collect demos")
        return []

    stage_cost, terminal_cost, _, _, _ = bench.make_costs()
    rng = np.random.default_rng(seed)

    mpc = CollisionFreeMPC(
        bench.dyn.dynamics, stage_cost, terminal_cost, bench.world,
        horizon=horizon, u_bounds=u_bounds, collision_weight=100.0,
        ilqr_iters=15,
    )

    demos = []
    for i in range(n_demos):
        s = np.array([
            rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0, 0.0,
        ])
        try:
            U = mpc.solve(s)
            demos.append((s.copy(), U.copy()))
        except Exception:
            continue

    print(f"  Collected {len(demos)}/{n_demos} expert demos from Collision-Free MPC")
    return demos


# ===========================================================================
# Phase 2: Train learning-based controllers
# ===========================================================================
def train_learning_controllers(demos, bench, horizon, net_cfg, seed=0):
    """Train learning-based controllers on the collected demonstrations.

    Returns a dict {controller_name: policy_fn}.
    """
    controllers = {}
    state_dim = bench.state_dim
    action_dim = bench.action_dim
    hidden_dim = net_cfg.get("hidden_dim", 64)
    epochs = net_cfg.get("training_epochs", 60)
    batch_size = net_cfg.get("batch_size", 16)
    lr = net_cfg.get("learning_rate", 1e-2)
    diff_steps = net_cfg.get("diffusion_steps", 8)
    n_diff_samples = net_cfg.get("num_diffusion_samples", 8)
    refine_steps = net_cfg.get("refine_steps", 5)
    ilqr_iters = net_cfg.get("ilqr_iters", 15)
    u_bounds = (
        net_cfg.get("u_low", -5.0 * np.ones(action_dim)),
        net_cfg.get("u_high", 5.0 * np.ones(action_dim)),
    )

    if len(demos) == 0:
        print("  [WARNING] No demonstrations available — skipping learning-based controllers")
        return controllers

    # --- SimpleDiffusionPolicy (from mpc_baselines_repo) + Warm-Start MPC ---
    if _MPC_OK:
        try:
            print("  [training] SimpleDiffusionPolicy (DDPM)...")
            policy = SimpleDiffusionPolicy(
                state_dim=state_dim, action_dim=action_dim, horizon=horizon,
                hidden_dim=hidden_dim, num_diffusion_steps=diff_steps, seed=seed,
            )
            policy.train(demos, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

            stage_cost, terminal_cost, _, _, _ = bench.make_costs()
            cfmpc = CollisionFreeMPC(
                bench.dyn.dynamics, stage_cost, terminal_cost, bench.world,
                horizon=horizon, u_bounds=u_bounds, collision_weight=100.0,
                ilqr_iters=ilqr_iters,
            )
            hybrid = DiffusionWarmStartMPC(
                diffusion_policy=policy, mpc_controller=cfmpc,
                num_diffusion_samples=n_diff_samples, refine_steps=refine_steps,
                obstacles=bench.obstacles,
            )

            def diffusion_ws_policy(x, _h=hybrid, _goal=bench.goal):
                return _h.solve(x, _goal)[0]

            controllers["Diffusion Warm-Start MPC"] = diffusion_ws_policy
            print("    [OK] Diffusion Warm-Start MPC trained")
        except Exception as exc:
            print(f"    [SKIP] Diffusion Warm-Start MPC: {exc}")

    # --- Minimal Iterative Policy (MIP) ---
    if _MPC_OK:
        try:
            print("  [training] MinimalIterativePolicy (MIP)...")
            mip = MinimalIterativePolicy(
                state_dim=state_dim, action_dim=action_dim, horizon=horizon,
                hidden_dim=hidden_dim, noise_std=0.1, seed=seed,
            )
            mip.train(demos, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

            u_lo, u_hi = u_bounds

            def mip_policy(x, _mip=mip, _lo=u_lo, _hi=u_hi):
                a = _mip.sample(x, num_samples=1)[0, 0]
                return np.clip(a, _lo, _hi)

            controllers["MIP (standalone)"] = mip_policy
            print("    [OK] MIP trained")
        except Exception as exc:
            print(f"    [SKIP] MIP: {exc}")

    # --- DDPM Policy (from diffusion_baselines, if available) ---
    if _DIFFUSION_OK and DDPMPolicy is not None:
        try:
            print("  [training] DDPM Policy (diffusion_baselines)...")
            ddpm = DDPMPolicy(
                action_dim=action_dim, horizon=horizon, obs_dim=state_dim,
                num_diffusion_steps=diff_steps, hidden_dim=hidden_dim, num_layers=2,
            )
            ddpm.train(demos, epochs=epochs, batch_size=batch_size, lr=lr)

            def ddpm_policy(x, _p=ddpm, _lo=u_bounds[0], _hi=u_bounds[1]):
                a = _p.sample(x, num_samples=1)
                if hasattr(a, "cpu"):
                    a = a.cpu().numpy()
                return np.clip(a[0, 0], _lo, _hi)

            controllers["DDPM Policy"] = ddpm_policy
            print("    [OK] DDPM Policy trained")
        except Exception as exc:
            print(f"    [SKIP] DDPM Policy: {exc}")

    # --- Flow Matching Policy ---
    if _DIFFUSION_OK and FlowMatchingPolicy is not None:
        try:
            print("  [training] Flow Matching Policy...")
            flow = FlowMatchingPolicy(
                action_dim=action_dim, horizon=horizon, obs_dim=state_dim,
                num_flow_steps=diff_steps, hidden_dim=hidden_dim, num_layers=2,
            )
            flow.train(demos, epochs=epochs, batch_size=batch_size, lr=lr)

            def flow_policy(x, _p=flow, _lo=u_bounds[0], _hi=u_bounds[1]):
                a = _p.sample(x, num_samples=1)
                if hasattr(a, "cpu"):
                    a = a.cpu().numpy()
                return np.clip(a[0, 0], _lo, _hi)

            controllers["Flow Matching Policy"] = flow_policy
            print("    [OK] Flow Matching Policy trained")
        except Exception as exc:
            print(f"    [SKIP] Flow Matching Policy: {exc}")

    # --- Regression Policy ---
    if _DIFFUSION_OK and RegressionPolicy is not None:
        try:
            print("  [training] Regression Policy (RCP)...")
            rcp = RegressionPolicy(
                action_dim=action_dim, horizon=horizon, obs_dim=state_dim,
                hidden_dim=hidden_dim, num_layers=2,
            )
            rcp.train(demos, num_epochs=epochs, batch_size=batch_size, lr=lr)

            def rcp_policy(x, _p=rcp, _lo=u_bounds[0], _hi=u_bounds[1]):
                a = _p.predict(x)
                if hasattr(a, "cpu"):
                    a = a.cpu().numpy()
                return np.clip(a[0], _lo, _hi)

            controllers["Regression Policy"] = rcp_policy
            print("    [OK] Regression Policy trained")
        except Exception as exc:
            print(f"    [SKIP] Regression Policy: {exc}")

    # --- Iterative Regression Policy ---
    if _DIFFUSION_OK and IterativeRegressionPolicy is not None:
        try:
            print("  [training] Iterative Regression Policy...")
            irp = IterativeRegressionPolicy(
                action_dim=action_dim, horizon=horizon, obs_dim=state_dim,
                hidden_dim=hidden_dim, num_iterations=2,
            )
            irp.train(demos, num_epochs=epochs, batch_size=batch_size, lr=lr)

            def irp_policy(x, _p=irp, _lo=u_bounds[0], _hi=u_bounds[1]):
                a = _p.predict(x)
                if hasattr(a, "cpu"):
                    a = a.cpu().numpy()
                return np.clip(a[0], _lo, _hi)

            controllers["Iterative Regression Policy"] = irp_policy
            print("    [OK] Iterative Regression Policy trained")
        except Exception as exc:
            print(f"    [SKIP] Iterative Regression Policy: {exc}")

    # --- SmallVLA ---
    # VLA baselines require image + language demonstrations. State-only demos
    # (the default for the 2-D reaching quick test) are not sufficient.
    if _VLA_OK and SmallVLA is not None:
        has_image_demos = any(isinstance(d, dict) and "image" in d for d in demos[:1])
        if not has_image_demos:
            print("  [SKIP] SmallVLA: requires image-based demonstrations (state-only demos provided)")
        else:
            try:
                print("  [training] SmallVLA...")
                vla = SmallVLA(action_dim=action_dim, horizon=horizon,
                               hidden_dim=hidden_dim)
                vla.train(demos, epochs=epochs, batch_size=batch_size, lr=lr)

                def vla_policy(x, _v=vla, _lo=u_bounds[0], _hi=u_bounds[1]):
                    a = _v.predict_action(x["image"], x.get("instruction", ""))
                    return np.clip(a, _lo, _hi)

                controllers["SmallVLA"] = vla_policy
                print("    [OK] SmallVLA trained")
            except Exception as exc:
                print(f"    [SKIP] SmallVLA: {exc}")

    # --- OpenVLA (typically too large for quick tests, but try) ---
    if _VLA_OK and OpenVLAWrapper is not None:
        has_image_demos = any(isinstance(d, dict) and "image" in d for d in demos[:1])
        if not has_image_demos:
            print("  [SKIP] OpenVLA: requires image-based demonstrations (state-only demos provided)")
        else:
            try:
                print("  [training] OpenVLA (this may take a while)...")
                vla = OpenVLAWrapper()

                def openvla_policy(x, _v=vla, _lo=u_bounds[0], _hi=u_bounds[1]):
                    a = _v.predict_action(x["image"], x.get("instruction", ""))
                    return np.clip(a, _lo, _hi)

                controllers["OpenVLA"] = openvla_policy
                print("    [OK] OpenVLA loaded")
            except Exception as exc:
                print(f"    [SKIP] OpenVLA: {exc}")

    return controllers


# ===========================================================================
# Build MPC controllers (model-based, no training needed)
# ===========================================================================
def build_mpc_controllers(bench, horizon, u_bounds, ilqr_iters=15):
    """Instantiate all MPC controllers for a benchmark."""
    controllers = {}
    if not _MPC_OK:
        print("  [WARNING] MPC baselines not available")
        return controllers

    stage_cost, terminal_cost, Q, R, Qf = bench.make_costs()

    # --- Linear MPC ---
    try:
        A, B = bench.dyn.linearize(np.zeros(bench.state_dim), np.zeros(bench.action_dim))
        lmpc = LinearMPC(A, B, Q, R, Qf, horizon, u_bounds=u_bounds)
        ref = np.tile(bench.goal, (horizon + 1, 1))

        def linear_policy(x, _mpc=lmpc, _ref=ref):
            return _mpc.solve(x, _ref).control

        controllers["Linear MPC"] = linear_policy
    except Exception as exc:
        print(f"  [skip] Linear MPC: {exc}")

    # --- Nonlinear MPC (iLQR) ---
    try:
        nmpc = NonlinearMPC(
            bench.dyn.dynamics, stage_cost, terminal_cost, horizon,
            u_bounds=u_bounds,
        )

        def nonlinear_policy(x, _mpc=nmpc):
            return _mpc.solve(x, max_iter=20)["action"]

        controllers["Nonlinear MPC (iLQR)"] = nonlinear_policy
    except Exception as exc:
        print(f"  [skip] Nonlinear MPC: {exc}")

    # --- Collision-Free MPC ---
    try:
        cfmpc = CollisionFreeMPC(
            bench.dyn.dynamics, stage_cost, terminal_cost, bench.world,
            horizon=horizon, u_bounds=u_bounds, collision_weight=100.0,
            ilqr_iters=ilqr_iters,
        )

        def cf_policy(x, _mpc=cfmpc):
            return _mpc.solve(x)[0]

        controllers["Collision-Free MPC"] = cf_policy
    except Exception as exc:
        print(f"  [skip] Collision-Free MPC: {exc}")

    return controllers


# ===========================================================================
# Phase 3: Closed-loop evaluation
# ===========================================================================
def simulate(bench, start, policy_fn, max_steps=None, goal_tol=None):
    """Run a closed-loop rollout using policy_fn(state) -> action.

    Returns a metrics dict.
    """
    if max_steps is None:
        max_steps = bench.max_steps
    if goal_tol is None:
        goal_tol = bench.goal_tol

    x = np.asarray(start, dtype=float).copy()
    trajectory = [x.copy()]
    solve_times: List[float] = []
    collisions = 0
    path_length = 0.0
    prev_pos = x[:2].copy()
    success = False
    actions = []

    for t in range(max_steps):
        t0 = time.perf_counter()
        try:
            u = policy_fn(x)
        except Exception as exc:
            print(f"    [warn] policy failed at step {t}: {exc}")
            break
        solve_times.append(time.perf_counter() - t0)
        u = np.asarray(u, dtype=float).reshape(-1)
        actions.append(u.copy())
        x = bench.step(x, u)
        trajectory.append(x.copy())

        pos = x[:2]
        path_length += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos.copy()
        if bench.is_collision(x):
            collisions += 1
        if bench.is_success(x):
            success = True
            break

    traj = np.array(trajectory)
    return {
        "success": int(success),
        "path_length": path_length,
        "collision_rate": collisions / max(len(trajectory) - 1, 1),
        "solve_time_s": float(np.mean(solve_times)) if solve_times else float("nan"),
        "latency_ms": float(np.mean(solve_times) * 1000) if solve_times else float("nan"),
        "steps": len(trajectory) - 1,
        "trajectory": traj,
        "actions": np.array(actions) if actions else np.zeros((0, bench.action_dim)),
    }


def evaluate_controllers(bench, controllers, n_episodes, seed=0):
    """Evaluate all controllers on a benchmark for n_episodes.

    Returns {controller_name: aggregated_metrics_dict}.
    """
    rng = np.random.default_rng(seed)
    results = {}

    for name, policy_fn in controllers.items():
        print(f"    Evaluating: {name}")
        runs = []
        for ep in range(n_episodes):
            s = bench.reset()
            try:
                metrics = simulate(bench, s, policy_fn)
                runs.append(metrics)
            except Exception as exc:
                print(f"      [warn] episode {ep} failed: {exc}")

        if not runs:
            results[name] = {
                "success_rate": 0.0, "path_length": float("nan"),
                "collision_rate": 1.0, "solve_time_s": float("nan"),
                "latency_ms": float("nan"), "n_episodes": 0,
            }
            continue

        # Compute mode coverage (how many distinct goal modes are reached)
        mode_coverage = 0.0
        if hasattr(bench, "goals"):
            n_modes_reached = 0
            for m in runs:
                final_pos = m["trajectory"][-1, :2]
                for g in bench.goals:
                    if np.linalg.norm(final_pos - g[:2]) <= bench.goal_tol:
                        n_modes_reached += 1
                        break
            mode_coverage = n_modes_reached / len(runs)

        results[name] = {
            "success_rate": float(np.mean([m["success"] for m in runs])),
            "success_std": float(np.std([m["success"] for m in runs])),
            "path_length": float(np.mean([m["path_length"] for m in runs])),
            "collision_rate": float(np.mean([m["collision_rate"] for m in runs])),
            "solve_time_s": float(np.mean([m["solve_time_s"] for m in runs])),
            "latency_ms": float(np.mean([m["solve_time_s"] for m in runs]) * 1000),
            "mode_coverage": mode_coverage,
            "n_episodes": len(runs),
        }

    return results


# ===========================================================================
# Phase 4: Metrics collection and table generation
# ===========================================================================
def save_results(all_results, output_dir, args):
    """Save results to CSV, JSON, and print a summary table."""
    os.makedirs(output_dir, exist_ok=True)
    tables_dir = os.path.join(output_dir, "tables")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # --- Save full JSON ---
    json_path = os.path.join(metrics_dir, "full_results.json")
    # Convert numpy types for JSON serialization
    def _jsonify(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(_jsonify(all_results), f, indent=2)
    print(f"\n  Full results saved to {json_path}")

    # --- Save master CSV ---
    csv_path = os.path.join(tables_dir, "master_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark", "controller", "seed",
            "success_rate", "success_std", "path_length",
            "collision_rate", "latency_ms", "mode_coverage", "n_episodes",
        ])
        for bench_name, bench_results in all_results["benchmarks"].items():
            for ctrl_name, ctrl_data in bench_results["controllers"].items():
                for seed_result in ctrl_data.get("seeds", []):
                    writer.writerow([
                        bench_name, ctrl_name, seed_result["seed"],
                        seed_result["success_rate"], seed_result.get("success_std", 0.0),
                        seed_result["path_length"], seed_result["collision_rate"],
                        seed_result["latency_ms"], seed_result.get("mode_coverage", 0.0),
                        seed_result["n_episodes"],
                    ])
    print(f"  Master CSV saved to {csv_path}")

    # --- Save aggregated CSV (averaged over seeds) ---
    agg_csv_path = os.path.join(tables_dir, "aggregated_comparison.csv")
    agg_rows = []
    with open(agg_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark", "controller",
            "success_rate_mean", "success_rate_std",
            "path_length_mean", "collision_rate_mean",
            "latency_ms_mean", "mode_coverage_mean",
            "n_seeds", "n_episodes",
        ])
        for bench_name, bench_results in all_results["benchmarks"].items():
            for ctrl_name, ctrl_data in bench_results["controllers"].items():
                seeds = ctrl_data.get("seeds", [])
                if not seeds:
                    continue
                sr = [s["success_rate"] for s in seeds]
                pl = [s["path_length"] for s in seeds]
                cr = [s["collision_rate"] for s in seeds]
                lat = [s["latency_ms"] for s in seeds]
                mc = [s.get("mode_coverage", 0.0) for s in seeds]
                row_dict = {
                    "benchmark": bench_name,
                    "controller": ctrl_name,
                    "success_rate_mean": float(np.mean(sr)),
                    "success_rate_std": float(np.std(sr)),
                    "path_length_mean": float(np.mean(pl)),
                    "collision_rate_mean": float(np.mean(cr)),
                    "latency_ms_mean": float(np.mean(lat)),
                    "mode_coverage_mean": float(np.mean(mc)),
                    "n_seeds": len(seeds),
                    "n_episodes": seeds[0]["n_episodes"],
                }
                agg_rows.append(row_dict)
                writer.writerow([
                    row_dict["benchmark"], row_dict["controller"],
                    row_dict["success_rate_mean"], row_dict["success_rate_std"],
                    row_dict["path_length_mean"], row_dict["collision_rate_mean"],
                    row_dict["latency_ms_mean"], row_dict["mode_coverage_mean"],
                    row_dict["n_seeds"], row_dict["n_episodes"],
                ])
    print(f"  Aggregated CSV saved to {agg_csv_path}")

    # Also save convenience copies in output_dir
    import shutil
    try:
        shutil.copyfile(csv_path, os.path.join(output_dir, "master_comparison.csv"))
        shutil.copyfile(agg_csv_path, os.path.join(output_dir, "aggregated_comparison.csv"))
    except Exception:
        pass

    # Save metrics summary JSON
    summary_json_path = os.path.join(metrics_dir, "metrics_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(agg_rows, f, indent=2)
    try:
        shutil.copyfile(summary_json_path, os.path.join(output_dir, "metrics_summary.json"))
    except Exception:
        pass

    # Generate summary plots
    figures_dir = os.path.join(output_dir, "figures")
    try:
        import generate_report
        generate_report.generate_bar_charts(agg_rows, figures_dir)
        generate_report.generate_pareto_plot(agg_rows, figures_dir)
    except Exception as exc:
        print(f"  [WARNING] Plot generation failed: {exc}")

    # --- Print summary table ---
    print("\n" + "=" * 90)
    print("SUMMARY: Aggregated Results (mean over seeds)")
    print("=" * 90)
    for bench_name, bench_results in all_results["benchmarks"].items():
        print(f"\n  Benchmark: {bench_name}")
        header = (f"  {'Controller':<32} {'Success':>8} {'PathLen':>9} "
                  f"{'CollRate':>9} {'Lat(ms)':>9} {'ModeCov':>8}")
        print(header)
        print("  " + "-" * 86)
        for ctrl_name, ctrl_data in bench_results["controllers"].items():
            seeds = ctrl_data.get("seeds", [])
            if not seeds:
                continue
            sr = float(np.mean([s["success_rate"] for s in seeds]))
            pl = float(np.mean([s["path_length"] for s in seeds]))
            cr = float(np.mean([s["collision_rate"] for s in seeds]))
            lat = float(np.mean([s["latency_ms"] for s in seeds]))
            mc = float(np.mean([s.get("mode_coverage", 0.0) for s in seeds]))
            print(f"  {ctrl_name:<32} {sr:>8.2f} {pl:>9.3f} "
                  f"{cr:>9.3f} {lat:>9.2f} {mc:>8.2f}")
    print("\n" + "=" * 90)


# ===========================================================================
# Main experiment pipeline
# ===========================================================================
def run_experiment(args):
    """Run the full 4-phase experiment pipeline."""
    start_time = time.time()

    # --- Load config ---
    config = {}
    config_path = os.path.join(STUDY_ROOT, "configs", "system_config.yaml")
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        print(f"[WARNING] Could not load config from {config_path}: {exc}")
        print("  Using defaults from command-line arguments.")

    # --- Determine benchmarks ---
    if args.benchmark == "all":
        bench_names = ["pusht", "reaching", "reaching_cluttered"]
    elif "," in args.benchmark:
        bench_names = [b.strip() for b in args.benchmark.split(",") if b.strip()]
    else:
        bench_names = [args.benchmark]

    # --- Determine controller families ---
    if args.controllers == "all":
        ctrl_families = ["mpc", "vla", "diffusion"]
    else:
        ctrl_families = args.controllers.split(",")

    # --- Network config ---
    net_size = args.net_size
    net_cfg = config.get("network_sizes", {}).get(net_size, {})
    if not net_cfg:
        # Fallback defaults
        if net_size == "small":
            net_cfg = {"hidden_dim": 32, "diffusion_steps": 8, "training_epochs": 30,
                       "batch_size": 16, "learning_rate": 0.01, "num_demos": 20,
                       "num_diffusion_samples": 4, "refine_steps": 3, "ilqr_iters": 10}
        else:
            net_cfg = {"hidden_dim": 128, "diffusion_steps": 100, "training_epochs": 200,
                       "batch_size": 64, "learning_rate": 0.001, "num_demos": 100,
                       "num_diffusion_samples": 16, "refine_steps": 5, "ilqr_iters": 20}

    # Override with quick-test settings if --quick
    if args.quick:
        net_cfg = {"hidden_dim": 16, "diffusion_steps": 4, "training_epochs": 10,
                   "batch_size": 8, "learning_rate": 0.01, "num_demos": 10,
                   "num_diffusion_samples": 2, "refine_steps": 2, "ilqr_iters": 5}
        if args.seeds is None:
            args.seeds = [0]
        if args.episodes is None:
            args.episodes = 5
        if args.output_dir is None:
            args.output_dir = os.path.join(STUDY_ROOT, "results", "quick_test")
        bench_names = ["reaching"]
        ctrl_families = ["mpc", "diffusion"]

    # --- Resolve defaults ---
    seeds = args.seeds if args.seeds else config.get("experiment", {}).get(
        "default_seeds", [0, 1, 2, 42, 123])
    n_episodes = args.episodes if args.episodes else config.get("experiment", {}).get(
        "default_episodes", 100)
    horizon = config.get("experiment", {}).get("default_horizon", 15)
    n_demos = net_cfg.get("num_demos", 30)
    output_dir = args.output_dir or os.path.join(STUDY_ROOT, "results")

    # --- Print experiment header ---
    print("\n" + "=" * 90)
    print("MPC vs VLA vs Diffusion — Integrated Comparison Runner")
    print("=" * 90)
    print(f"  Benchmarks:    {bench_names}")
    print(f"  Controllers:   {ctrl_families}")
    print(f"  Seeds:         {seeds}")
    print(f"  Episodes:      {n_episodes}")
    print(f"  Horizon:       {horizon}")
    print(f"  Network size:  {net_size}")
    print(f"  Output dir:    {output_dir}")
    print(f"  MPC available:     {_MPC_OK}")
    print(f"  VLA available:     {_VLA_OK}")
    print(f"  Diffusion avail:   {_DIFFUSION_OK}")
    print(f"  Benchmarks avail:  {_BENCH_OK}")
    print("=" * 90)

    # --- Results container ---
    all_results = {
        "config": {
            "benchmarks": bench_names,
            "controller_families": ctrl_families,
            "seeds": seeds,
            "n_episodes": n_episodes,
            "horizon": horizon,
            "net_size": net_size,
            "net_cfg": {k: v for k, v in net_cfg.items()},
        },
        "benchmarks": {},
    }

    u_bounds = (
        np.array([config.get("experiment", {}).get("control_bounds", [-5.0])[0]] * 2),
        np.array([config.get("experiment", {}).get("control_bounds", [5.0])[1]] * 2),
    )

    # --- Run per benchmark ---
    for bench_name in bench_names:
        print(f"\n{'─' * 90}")
        print(f"BENCHMARK: {bench_name}")
        print(f"{'─' * 90}")

        bench_results = {"controllers": {}}

        for seed in seeds:
            print(f"\n  [Seed {seed}]")
            bench = _make_benchmark(bench_name, seed=seed)

            # --- Phase 1: Collect demonstrations ---
            print(f"\n  Phase 1: Collecting demonstrations from MPC expert...")
            demos = []
            if "mpc" in ctrl_families or "diffusion" in ctrl_families:
                demos = collect_demonstrations(
                    bench, n_demos=n_demos, horizon=horizon,
                    u_bounds=u_bounds, seed=seed,
                )

            # --- Phase 2: Train learning-based controllers ---
            print(f"\n  Phase 2: Training learning-based controllers...")
            learning_controllers = {}
            if "diffusion" in ctrl_families or "vla" in ctrl_families:
                learning_controllers = train_learning_controllers(
                    demos, bench, horizon, net_cfg, seed=seed,
                )

            # --- Build MPC controllers ---
            mpc_controllers = {}
            if "mpc" in ctrl_families:
                print(f"\n  Building MPC controllers...")
                mpc_controllers = build_mpc_controllers(
                    bench, horizon, u_bounds,
                    ilqr_iters=net_cfg.get("ilqr_iters", 15),
                )

            # --- Merge all controllers ---
            all_controllers = {}
            all_controllers.update(mpc_controllers)
            all_controllers.update(learning_controllers)

            if not all_controllers:
                print("  [WARNING] No controllers available — skipping benchmark")
                continue

            # --- Phase 3: Evaluate ALL controllers ---
            print(f"\n  Phase 3: Evaluating {len(all_controllers)} controllers "
                  f"on {n_episodes} episodes...")
            seed_results = evaluate_controllers(
                bench, all_controllers, n_episodes, seed=seed,
            )

            # Store per-seed results
            for ctrl_name, metrics in seed_results.items():
                if ctrl_name not in bench_results["controllers"]:
                    bench_results["controllers"][ctrl_name] = {"seeds": []}
                metrics["seed"] = seed
                bench_results["controllers"][ctrl_name]["seeds"].append(metrics)

        all_results["benchmarks"][bench_name] = bench_results

    # --- Phase 4: Collect metrics and generate tables ---
    print(f"\n{'─' * 90}")
    print("Phase 4: Collecting metrics and generating comparison tables...")
    print(f"{'─' * 90}")
    save_results(all_results, output_dir, args)

    elapsed = time.time() - start_time
    print(f"\nTotal experiment time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("Done!")

    return all_results


# ===========================================================================
# Argument parsing
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MPC vs VLA vs Diffusion — Integrated Comparison Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full experiment (all benchmarks, all controllers, 5 seeds, 100 episodes)
  conda run -n mpc_vla python run_experiments.py

  # Quick smoke test
  conda run -n mpc_vla python run_experiments.py --quick

  # PushT only, MPC + diffusion, 3 seeds, 50 episodes
  conda run -n mpc_vla python run_experiments.py --benchmark pusht --controllers mpc,diffusion --seeds 0 1 2 --episodes 50

  # Reaching with cluttered obstacles, medium networks
  conda run -n mpc_vla python run_experiments.py --benchmark reaching_cluttered --net-size medium --episodes 100
        """,
    )
    parser.add_argument(
        "--benchmark", type=str, default="all",
        help="Benchmark to run: pusht, reaching, reaching_cluttered, all, or comma-separated list (default: all)",
    )
    parser.add_argument(
        "--controllers", type=str, default="all",
        help="Controller families: mpc, vla, diffusion, all, or comma-separated (default: all)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Random seeds (default: [0, 1, 2, 42, 123])",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of evaluation episodes per controller (default: 100)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--net-size", type=str, default="medium",
        choices=["small", "medium", "large"],
        help="Network size preset (default: medium)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: 1 seed, 5 episodes, tiny networks, reaching only",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to system config YAML (default: configs/system_config.yaml)",
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
