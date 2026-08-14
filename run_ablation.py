#!/usr/bin/env python3
"""EXP-001: GCP Component Ablation Runner.

This script isolates the contribution of the three key ingredients of
Generative Computation Policies (GCP / diffusion policies):

  1. **Iterative compute** -- multi-step denoising vs. single-step regression.
  2. **Noise injection** -- stochastic sampling vs. deterministic mapping.
  3. **Multi-step denoising** -- T-step reverse process vs. minimal iterations.

Ablation variants
-----------------
  - **Full DDPM (T=100, with noise)**: complete Markov-chain reverse process.
  - **DDPM no-noise (T=100, deterministic)**: same T steps but sigma=0 (mean only).
  - **DDPM single-step (T=1, with noise)**: collapses to one reverse step.
  - **MIP (2 iterations, noise_std=0.1)**: Minimal Iterative Policy -- regression
    + noise without the full reverse chain.
  - **Pure Regression (RCP)**: single deterministic regression (no noise, no
    iteration).

Each variant is trained on demonstrations collected from a Collision-Free MPC
expert and evaluated in closed loop on PushT and 2-D Reaching benchmarks.

Metrics collected: ``success_rate``, ``mode_coverage``, ``inference_latency``.

Usage::

    conda run -n mpc_vla python run_ablation.py --benchmark all --seeds 0 1 2 --episodes 50
    conda run -n mpc_vla python run_ablation.py --benchmark pusht --seeds 0 --episodes 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup -- make all module directories importable
# ---------------------------------------------------------------------------
STUDY_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STUDY_ROOT)
sys.path.insert(0, os.path.join(STUDY_ROOT, "mpc_baselines_repo"))
sys.path.insert(0, os.path.join(STUDY_ROOT, "mpc_baselines_repo", "src"))

_MODULE_DIRS = [
    "mpc_baselines_repo",
    "vla_baselines",
    "diffusion_baselines",
    "benchmarks",
    "src",
    "mpc_baselines_repo/src",
]
for _d in _MODULE_DIRS:
    _full = os.path.join(STUDY_ROOT, _d)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)


# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------
print("[ablation] Loading modules...")

# --- MPC baselines (needed for demo collection) -----------------------------
_MPC_OK = False
PointMass2D = None
CircleObstacle = None
is_in_collision = None
CollisionFreeMPC = None
SDFWorld = None
SimpleDiffusionPolicy = None
MinimalIterativePolicy = None
try:
    from src.utils import PointMass2D
    from src.utils.obstacles import CircleObstacle, is_in_collision
    from src.collision_free_mpc import CollisionFreeMPC, SDFWorld
    from src.diffusion_warm_start import (
        SimpleDiffusionPolicy,
        MinimalIterativePolicy,
    )
    _MPC_OK = True
    print("  [OK] MPC baselines loaded")
except Exception as exc:
    print(f"  [WARNING] MPC baselines import failed: {exc}")
    traceback.print_exc()

# --- Diffusion baselines (full PyTorch DDPM) ---------------------------------
_DIFFUSION_OK = False
DiffusionPolicy = None
FlowMatchingPolicy = None
try:
    from diffusion_baselines.ddpm_policy import DiffusionPolicy  # type: ignore
    from diffusion_baselines.flow_matching_policy import FlowMatchingPolicy  # type: ignore
    _DIFFUSION_OK = True
    print("  [OK] Diffusion baselines loaded (DiffusionPolicy, FlowMatchingPolicy)")
except ImportError as exc:
    print(f"  [SKIP] diffusion_baselines not found: {exc}")
except Exception as exc:
    print(f"  [WARNING] Diffusion baselines import error: {exc}")

# --- Benchmarks (may not exist or may have different names) ------------------
_BENCH_OK = False
PushTEnv = None
ReachingEnv = None
DemonstrationCollector = None
Evaluator = None
try:
    from benchmarks import (  # type: ignore
        DemonstrationCollector,
        Evaluator,
        PushTEnv,
        ReachingEnv,
    )
    _BENCH_OK = True
    print("  [OK] Benchmarks loaded")
except ImportError:
    print("  [SKIP] benchmarks module not found -- using built-in fallbacks")
except Exception as exc:
    print(f"  [WARNING] Benchmarks import error: {exc}")


# ===========================================================================
# Built-in fallback benchmarks (mirror run_experiments.py)
# ===========================================================================
class _FallbackReaching:
    """2-D point-mass reaching with obstacles."""

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
        self.max_steps = 60

    def reset(self) -> np.ndarray:
        return self.start + np.array([
            self.rng.uniform(-0.3, 0.3),
            self.rng.uniform(-0.3, 0.3),
            0.0, 0.0,
        ])

    def step(self, state, action):
        return self.dyn.dynamics(state, action)

    def is_success(self, state):
        return float(np.linalg.norm(state[:2] - self.goal[:2])) <= self.goal_tol

    def is_collision(self, state):
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
    """Simplified PushT-style multi-modal reaching."""

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
        return self.start + np.array([
            self.rng.uniform(-0.3, 0.3),
            self.rng.uniform(-0.3, 0.3),
            0.0, 0.0,
        ])

    def step(self, state, action):
        return self.dyn.dynamics(state, action)

    def is_success(self, state):
        for g in self.goals:
            if float(np.linalg.norm(state[:2] - g[:2])) <= self.goal_tol:
                return True
        return False

    def is_collision(self, state):
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


def _make_benchmark(name: str, seed: int = 0):
    if name == "pusht":
        return _FallbackPushT(seed=seed)
    elif name == "reaching":
        return _FallbackReaching(seed=seed, cluttered=False)
    elif name == "reaching_cluttered":
        return _FallbackReaching(seed=seed, cluttered=True)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


# ===========================================================================
# Ablation policy variants
# ===========================================================================
class _DeterministicSimpleDiffusionPolicy:
    """Wrapper around SimpleDiffusionPolicy that samples deterministically.

    Identical to SimpleDiffusionPolicy except the reverse-process noise
    (sigma term) is set to zero, so sampling returns only the posterior mean
    at each step.
    """

    def __init__(self, **kwargs):
        self._policy = SimpleDiffusionPolicy(**kwargs)

    @property
    def num_diffusion_steps(self):
        return self._policy.num_diffusion_steps

    @property
    def horizon(self):
        return self._policy.horizon

    @property
    def action_dim(self):
        return self._policy.action_dim

    @property
    def flat_dim(self):
        return self._policy.flat_dim

    def train(self, demos, **kwargs):
        return self._policy.train(demos, **kwargs)

    def sample(self, state: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """Deterministic sampling -- posterior mean only, no noise."""
        p = self._policy
        s = np.asarray(state, dtype=float).reshape(-1)
        samples = np.zeros((num_samples, p.flat_dim))
        for n in range(num_samples):
            x = p.rng.standard_normal(p.flat_dim)
            for t in reversed(range(p.num_diffusion_steps)):
                eps = p._predict_noise(s, x, t)
                beta = p.coeffs["betas"][t]
                sqrt_recip_alpha = p.coeffs["sqrt_recip_alphas"][t]
                mean = sqrt_recip_alpha * (
                    x - beta / p.coeffs["sqrt_one_minus_alpha_bar"][t] * eps
                )
                # No noise added (deterministic)
                x = mean
            samples[n] = x
        return samples.reshape(num_samples, p.horizon, p.action_dim)


class _PureRegressionPolicy:
    """Pure regression policy (RCP) -- single deterministic MLP mapping.

    state -> action_sequence with no noise injection and no iterative
    refinement.  This is the simplest learning baseline and isolates the
    contribution of the generative components.
    """

    def __init__(self, state_dim, action_dim, horizon, hidden_dim=64, seed=0):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.flat_dim = self.action_dim * self.horizon
        self.rng = np.random.default_rng(seed)

        # Single MLP: state -> flat action sequence
        self._W1 = self.rng.standard_normal((state_dim, hidden_dim)) * 0.1
        self._b1 = np.zeros(hidden_dim)
        self._W2 = self.rng.standard_normal((hidden_dim, hidden_dim)) * 0.1
        self._b2 = np.zeros(hidden_dim)
        self._W3 = self.rng.standard_normal((hidden_dim, self.flat_dim)) * 0.1
        self._b3 = np.zeros(self.flat_dim)
        self._trained = False

    def _forward(self, x):
        h1 = np.maximum(0, x @ self._W1 + self._b1)
        h2 = np.maximum(0, h1 @ self._W2 + self._b2)
        out = h2 @ self._W3 + self._b3
        return out

    def train(self, demos, epochs=50, batch_size=16, lr=1e-2, verbose=False):
        if not demos:
            return []
        states = np.array([d[0] for d in demos])
        actions = np.array([d[1].reshape(-1) for d in demos])
        n = len(demos)
        losses = []
        for epoch in range(epochs):
            perm = self.rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = states[idx]
                yb = actions[idx]
                # Forward
                h1 = np.maximum(0, xb @ self._W1 + self._b1)
                h2 = np.maximum(0, h1 @ self._W2 + self._b2)
                pred = h2 @ self._W3 + self._b3
                # MSE loss
                err = pred - yb
                loss = float(np.mean(err ** 2))
                epoch_loss += loss
                n_batches += 1
                # Backprop (manual gradients)
                bs = len(idx)
                grad_out = (2.0 / bs) * err
                grad_W3 = h2.T @ grad_out
                grad_b3 = grad_out.sum(axis=0)
                grad_h2 = grad_out @ self._W3.T
                grad_h2[h2 <= 0] = 0
                grad_W2 = h1.T @ grad_h2
                grad_b2 = grad_h2.sum(axis=0)
                grad_h1 = grad_h2 @ self._W2.T
                grad_h1[h1 <= 0] = 0
                grad_W1 = xb.T @ grad_h1
                grad_b1 = grad_h1.sum(axis=0)
                # SGD update
                self._W3 -= lr * grad_W3
                self._b3 -= lr * grad_b3
                self._W2 -= lr * grad_W2
                self._b2 -= lr * grad_b2
                self._W1 -= lr * grad_W1
                self._b1 -= lr * grad_b1
            avg = epoch_loss / max(n_batches, 1)
            losses.append(avg)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  [RCP] epoch {epoch:4d}/{epochs}  loss={avg:.6f}")
        self._trained = True
        return losses

    def sample(self, state: np.ndarray, num_samples: int = 1) -> np.ndarray:
        s = np.asarray(state, dtype=float).reshape(-1)
        out = np.zeros((num_samples, self.flat_dim))
        for n in range(num_samples):
            out[n] = self._forward(s)
        return out.reshape(num_samples, self.horizon, self.action_dim)


# ===========================================================================
# Phase 1: Collect demonstrations from MPC expert
# ===========================================================================
def collect_demonstrations(bench, n_demos, horizon, u_bounds, seed=0):
    if not _MPC_OK:
        print("  [WARNING] MPC baselines not available -- cannot collect demos")
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

    print(f"  Collected {len(demos)}/{n_demos} expert demos")
    return demos


# ===========================================================================
# Build ablation policy variants
# ===========================================================================
def build_ablation_variants(demos, bench, horizon, net_cfg, seed=0):
    """Build and train all 5 ablation variants.

    Returns dict {variant_name: policy_fn}.
    """
    variants = {}
    state_dim = bench.state_dim
    action_dim = bench.action_dim
    hidden_dim = net_cfg.get("hidden_dim", 64)
    epochs = net_cfg.get("training_epochs", 50)
    batch_size = net_cfg.get("batch_size", 16)
    lr = net_cfg.get("learning_rate", 1e-2)
    u_lo = -5.0 * np.ones(action_dim)
    u_hi = 5.0 * np.ones(action_dim)

    if not demos:
        print("  [WARNING] No demos -- skipping all ablation variants")
        return variants

    # --- Variant 1: Full DDPM (T=100, with noise) ---------------------------
    if _MPC_OK and SimpleDiffusionPolicy is not None:
        try:
            print("  [ablation] Training Full DDPM (T=100, noise)...")
            policy = SimpleDiffusionPolicy(
                state_dim=state_dim, action_dim=action_dim, horizon=horizon,
                hidden_dim=hidden_dim, num_diffusion_steps=100, seed=seed,
            )
            policy.train(demos, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

            def full_ddpm_policy(x, _p=policy, _lo=u_lo, _hi=u_hi):
                a = _p.sample(x, num_samples=1)[0, 0]
                return np.clip(a, _lo, _hi)

            variants["Full DDPM (T=100)"] = full_ddpm_policy
            print("    [OK] Full DDPM trained")
        except Exception as exc:
            print(f"    [SKIP] Full DDPM: {exc}")

    # --- Variant 2: DDPM no-noise (T=100, deterministic) --------------------
    if _MPC_OK and SimpleDiffusionPolicy is not None:
        try:
            print("  [ablation] Training DDPM no-noise (T=100, deterministic)...")
            policy = _DeterministicSimpleDiffusionPolicy(
                state_dim=state_dim, action_dim=action_dim, horizon=horizon,
                hidden_dim=hidden_dim, num_diffusion_steps=100, seed=seed,
            )
            policy.train(demos, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

            def ddpm_deterministic_policy(x, _p=policy, _lo=u_lo, _hi=u_hi):
                a = _p.sample(x, num_samples=1)[0, 0]
                return np.clip(a, _lo, _hi)

            variants["DDPM no-noise (T=100)"] = ddpm_deterministic_policy
            print("    [OK] DDPM no-noise trained")
        except Exception as exc:
            print(f"    [SKIP] DDPM no-noise: {exc}")

    # --- Variant 3: DDPM single-step (T=1, with noise) ----------------------
    if _MPC_OK and SimpleDiffusionPolicy is not None:
        try:
            print("  [ablation] Training DDPM single-step (T=1, noise)...")
            policy = SimpleDiffusionPolicy(
                state_dim=state_dim, action_dim=action_dim, horizon=horizon,
                hidden_dim=hidden_dim, num_diffusion_steps=1, seed=seed,
            )
            policy.train(demos, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

            def ddpm_single_step_policy(x, _p=policy, _lo=u_lo, _hi=u_hi):
                a = _p.sample(x, num_samples=1)[0, 0]
                return np.clip(a, _lo, _hi)

            variants["DDPM single-step (T=1)"] = ddpm_single_step_policy
            print("    [OK] DDPM single-step trained")
        except Exception as exc:
            print(f"    [SKIP] DDPM single-step: {exc}")

    # --- Variant 4: MIP (2 iterations, noise_std=0.1) -----------------------
    if _MPC_OK and MinimalIterativePolicy is not None:
        try:
            print("  [ablation] Training MIP (2-step, noise_std=0.1)...")
            mip = MinimalIterativePolicy(
                state_dim=state_dim, action_dim=action_dim, horizon=horizon,
                hidden_dim=hidden_dim, noise_std=0.1, seed=seed,
            )
            mip.train(demos, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

            def mip_policy(x, _m=mip, _lo=u_lo, _hi=u_hi):
                a = _m.sample(x, num_samples=1)[0, 0]
                return np.clip(a, _lo, _hi)

            variants["MIP (2-iter, noise=0.1)"] = mip_policy
            print("    [OK] MIP trained")
        except Exception as exc:
            print(f"    [SKIP] MIP: {exc}")

    # --- Variant 5: Pure Regression (RCP) -----------------------------------
    try:
        print("  [ablation] Training Pure Regression (RCP)...")
        rcp = _PureRegressionPolicy(
            state_dim=state_dim, action_dim=action_dim, horizon=horizon,
            hidden_dim=hidden_dim, seed=seed,
        )
        rcp.train(demos, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

        def rcp_policy(x, _p=rcp, _lo=u_lo, _hi=u_hi):
            a = _p.sample(x, num_samples=1)[0, 0]
            return np.clip(a, _lo, _hi)

        variants["Pure Regression (RCP)"] = rcp_policy
        print("    [OK] Pure Regression trained")
    except Exception as exc:
        print(f"    [SKIP] Pure Regression: {exc}")

    return variants


# ===========================================================================
# Phase 3: Closed-loop evaluation
# ===========================================================================
def simulate(bench, start, policy_fn, max_steps=None, goal_tol=None):
    if max_steps is None:
        max_steps = bench.max_steps
    if goal_tol is None:
        goal_tol = bench.goal_tol

    x = np.asarray(start, dtype=float).copy()
    trajectory = [x.copy()]
    solve_times = []
    collisions = 0
    path_length = 0.0
    prev_pos = x[:2].copy()
    success = False

    for t in range(max_steps):
        t0 = time.perf_counter()
        try:
            u = policy_fn(x)
        except Exception as exc:
            print(f"    [warn] policy failed at step {t}: {exc}")
            break
        solve_times.append(time.perf_counter() - t0)
        u = np.asarray(u, dtype=float).reshape(-1)
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
        "latency_ms": float(np.mean(solve_times) * 1000) if solve_times else float("nan"),
        "steps": len(trajectory) - 1,
        "trajectory": traj,
    }


def evaluate_variants(bench, variants, n_episodes, seed=0):
    """Evaluate all ablation variants. Returns {name: metrics}."""
    rng = np.random.default_rng(seed)
    results = {}

    for name, policy_fn in variants.items():
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
                "success_rate": 0.0, "mode_coverage": 0.0,
                "latency_ms": float("nan"), "n_episodes": 0,
            }
            continue

        # Mode coverage
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
            "mode_coverage": mode_coverage,
            "latency_ms": float(np.mean([m["latency_ms"] for m in runs])),
            "path_length": float(np.mean([m["path_length"] for m in runs])),
            "collision_rate": float(np.mean([m["collision_rate"] for m in runs])),
            "n_episodes": len(runs),
        }

    return results


# ===========================================================================
# Phase 4: Save results and generate plots
# ===========================================================================
def save_ablation_results(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # --- Save JSON ---
    json_path = os.path.join(output_dir, "ablation_results.json")
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
    print(f"  Ablation JSON saved to {json_path}")

    # --- Save CSV ---
    csv_path = os.path.join(output_dir, "ablation_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark", "variant", "seed",
            "success_rate", "mode_coverage", "latency_ms",
            "path_length", "collision_rate", "n_episodes",
        ])
        for bench_name, bench_data in all_results["benchmarks"].items():
            for variant_name, variant_data in bench_data["variants"].items():
                for seed_result in variant_data.get("seeds", []):
                    writer.writerow([
                        bench_name, variant_name, seed_result["seed"],
                        seed_result["success_rate"], seed_result["mode_coverage"],
                        seed_result["latency_ms"], seed_result["path_length"],
                        seed_result["collision_rate"], seed_result["n_episodes"],
                    ])
    print(f"  Ablation CSV saved to {csv_path}")

    # --- Aggregated CSV (mean over seeds) ---
    agg_csv_path = os.path.join(output_dir, "ablation_aggregated.csv")
    with open(agg_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark", "variant",
            "success_rate_mean", "success_rate_std",
            "mode_coverage_mean", "latency_ms_mean",
            "n_seeds", "n_episodes",
        ])
        for bench_name, bench_data in all_results["benchmarks"].items():
            for variant_name, variant_data in bench_data["variants"].items():
                seeds = variant_data.get("seeds", [])
                if not seeds:
                    continue
                sr = [s["success_rate"] for s in seeds]
                mc = [s["mode_coverage"] for s in seeds]
                lat = [s["latency_ms"] for s in seeds]
                writer.writerow([
                    bench_name, variant_name,
                    float(np.mean(sr)), float(np.std(sr)),
                    float(np.mean(mc)), float(np.mean(lat)),
                    len(seeds), seeds[0]["n_episodes"],
                ])
    print(f"  Aggregated CSV saved to {agg_csv_path}")

    # --- Print summary table ---
    print("\n" + "=" * 90)
    print("EXP-001: GCP Component Ablation -- Summary (mean over seeds)")
    print("=" * 90)
    for bench_name, bench_data in all_results["benchmarks"].items():
        print(f"\n  Benchmark: {bench_name}")
        header = (f"  {'Variant':<32} {'Success':>8} {'ModeCov':>8} "
                  f"{'Lat(ms)':>9}")
        print(header)
        print("  " + "-" * 60)
        for variant_name, variant_data in bench_data["variants"].items():
            seeds = variant_data.get("seeds", [])
            if not seeds:
                continue
            sr = float(np.mean([s["success_rate"] for s in seeds]))
            mc = float(np.mean([s["mode_coverage"] for s in seeds]))
            lat = float(np.mean([s["latency_ms"] for s in seeds]))
            print(f"  {variant_name:<32} {sr:>8.2f} {mc:>8.2f} {lat:>9.2f}")
    print("\n" + "=" * 90)

    # --- Generate bar chart ---
    try:
        _generate_ablation_bar_chart(all_results, output_dir)
    except Exception as exc:
        print(f"  [WARNING] Could not generate bar chart: {exc}")


def _generate_ablation_bar_chart(all_results, output_dir):
    """Generate grouped bar chart comparing ablation variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    bench_names = list(all_results["benchmarks"].keys())
    n_benchmarks = len(bench_names)

    # Collect all variant names (union across benchmarks)
    all_variants = []
    for bn in bench_names:
        for vn in all_results["benchmarks"][bn]["variants"]:
            if vn not in all_variants:
                all_variants.append(vn)
    n_variants = len(all_variants)

    if n_variants == 0 or n_benchmarks == 0:
        print("  [WARNING] No data for bar chart")
        return

    # --- Success rate bar chart ---
    fig, ax = plt.subplots(figsize=(max(10, n_variants * 1.5), 6))
    x = np.arange(n_variants)
    width = 0.8 / n_benchmarks
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    for i, bn in enumerate(bench_names):
        values = []
        for vn in all_variants:
            seeds = all_results["benchmarks"][bn]["variants"].get(vn, {}).get("seeds", [])
            if seeds:
                values.append(float(np.mean([s["success_rate"] for s in seeds])))
            else:
                values.append(0.0)
        ax.bar(x + i * width, values, width, label=bn,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Ablation Variant", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("EXP-001: GCP Component Ablation -- Success Rate", fontsize=14)
    ax.set_xticks(x + width * (n_benchmarks - 1) / 2)
    ax.set_xticklabels([v[:20] for v in all_variants], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    chart_path = os.path.join(figures_dir, "ablation_success_rate.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  Bar chart saved to {chart_path}")

    # --- Latency bar chart ---
    fig, ax = plt.subplots(figsize=(max(10, n_variants * 1.5), 6))
    for i, bn in enumerate(bench_names):
        values = []
        for vn in all_variants:
            seeds = all_results["benchmarks"][bn]["variants"].get(vn, {}).get("seeds", [])
            if seeds:
                lat = [s["latency_ms"] for s in seeds if not np.isnan(s["latency_ms"])]
                values.append(float(np.mean(lat)) if lat else 0.0)
            else:
                values.append(0.0)
        ax.bar(x + i * width, values, width, label=bn,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Ablation Variant", fontsize=12)
    ax.set_ylabel("Inference Latency (ms)", fontsize=12)
    ax.set_title("EXP-001: GCP Component Ablation -- Inference Latency", fontsize=14)
    ax.set_xticks(x + width * (n_benchmarks - 1) / 2)
    ax.set_xticklabels([v[:20] for v in all_variants], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    chart_path = os.path.join(figures_dir, "ablation_latency.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  Latency chart saved to {chart_path}")

    # --- Mode coverage bar chart ---
    fig, ax = plt.subplots(figsize=(max(10, n_variants * 1.5), 6))
    for i, bn in enumerate(bench_names):
        values = []
        for vn in all_variants:
            seeds = all_results["benchmarks"][bn]["variants"].get(vn, {}).get("seeds", [])
            if seeds:
                values.append(float(np.mean([s["mode_coverage"] for s in seeds])))
            else:
                values.append(0.0)
        ax.bar(x + i * width, values, width, label=bn,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Ablation Variant", fontsize=12)
    ax.set_ylabel("Mode Coverage", fontsize=12)
    ax.set_title("EXP-001: GCP Component Ablation -- Mode Coverage", fontsize=14)
    ax.set_xticks(x + width * (n_benchmarks - 1) / 2)
    ax.set_xticklabels([v[:20] for v in all_variants], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    chart_path = os.path.join(figures_dir, "ablation_mode_coverage.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  Mode coverage chart saved to {chart_path}")


# ===========================================================================
# Main ablation pipeline
# ===========================================================================
def run_ablation(args):
    start_time = time.time()

    # --- Load config ---
    config = {}
    config_path = os.path.join(STUDY_ROOT, "configs", "system_config.yaml")
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        print(f"[WARNING] Could not load config: {exc}")

    # --- Determine benchmarks ---
    if args.benchmark == "all":
        bench_names = ["pusht", "reaching"]
    else:
        bench_names = [args.benchmark]

    # --- Network config (small for quick runs) ---
    net_cfg = {
        "hidden_dim": 64,
        "num_layers": 2,
        "training_epochs": args.epochs if args.epochs else 50,
        "batch_size": 16,
        "learning_rate": 0.01,
        "num_demos": args.num_demos if args.num_demos else 30,
    }

    # --- Resolve defaults ---
    seeds = args.seeds if args.seeds else [0, 1, 2]
    n_episodes = args.episodes if args.episodes else 50
    horizon = config.get("experiment", {}).get("default_horizon", 15)
    n_demos = net_cfg["num_demos"]
    output_dir = args.output_dir or os.path.join(
        STUDY_ROOT, "experiments", "EXP-001-mechanism-ablation", "outputs"
    )

    u_bounds = (
        np.array([config.get("experiment", {}).get("control_bounds", [-5.0])[0]] * 2),
        np.array([config.get("experiment", {}).get("control_bounds", [5.0])[1]] * 2),
    )

    # --- Print header ---
    print("\n" + "=" * 90)
    print("EXP-001: GCP Component Ablation Runner")
    print("=" * 90)
    print(f"  Benchmarks:    {bench_names}")
    print(f"  Seeds:         {seeds}")
    print(f"  Episodes:      {n_episodes}")
    print(f"  Horizon:       {horizon}")
    print(f"  Network:       hidden_dim={net_cfg['hidden_dim']}, "
          f"epochs={net_cfg['training_epochs']}")
    print(f"  Output dir:    {output_dir}")
    print(f"  MPC available:     {_MPC_OK}")
    print(f"  Diffusion avail:   {_DIFFUSION_OK}")
    print(f"  Benchmarks avail:  {_BENCH_OK}")
    print("=" * 90)

    all_results = {
        "experiment": "EXP-001: GCP Component Ablation",
        "config": {
            "benchmarks": bench_names,
            "seeds": seeds,
            "n_episodes": n_episodes,
            "horizon": horizon,
            "net_cfg": net_cfg,
        },
        "benchmarks": {},
    }

    # --- Run per benchmark ---
    for bench_name in bench_names:
        print(f"\n{'--' * 45}")
        print(f"BENCHMARK: {bench_name}")
        print(f"{'--' * 45}")

        bench_results = {"variants": {}}

        # Phase 1: Collect demos once (deterministic, seed 0)
        print("\n  Phase 1: Collecting demonstrations...")
        bench_0 = _make_benchmark(bench_name, seed=0)
        demos = collect_demonstrations(
            bench_0, n_demos=n_demos, horizon=horizon,
            u_bounds=u_bounds, seed=0,
        )

        # Phase 2: Build and train ablation variants once
        print("\n  Phase 2: Training ablation variants...")
        variants = build_ablation_variants(
            demos, bench_0, horizon, net_cfg, seed=0,
        )

        if not variants:
            print("  [WARNING] No variants available -- skipping")
            continue

        # Phase 3: Evaluate across seeds
        for seed in seeds:
            print(f"\n  [Seed {seed}]")
            bench = _make_benchmark(bench_name, seed=seed)
            print(f"  Phase 3: Evaluating {len(variants)} variants "
                  f"on {n_episodes} episodes...")
            seed_results = evaluate_variants(
                bench, variants, n_episodes, seed=seed,
            )

            for vname, metrics in seed_results.items():
                if vname not in bench_results["variants"]:
                    bench_results["variants"][vname] = {"seeds": []}
                metrics["seed"] = seed
                bench_results["variants"][vname]["seeds"].append(metrics)

        all_results["benchmarks"][bench_name] = bench_results

    # Phase 4: Save and plot
    print(f"\n{'--' * 45}")
    print("Phase 4: Saving results and generating plots...")
    print(f"{'--' * 45}")
    save_ablation_results(all_results, output_dir)

    elapsed = time.time() - start_time
    print(f"\nTotal ablation time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("Done!")

    return all_results


# ===========================================================================
# Argument parsing
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="EXP-001: GCP Component Ablation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full ablation (all benchmarks, 3 seeds, 50 episodes)
  conda run -n mpc_vla python run_ablation.py

  # PushT only, 1 seed, 10 episodes
  conda run -n mpc_vla python run_ablation.py --benchmark pusht --seeds 0 --episodes 10

  # Custom output dir and epochs
  conda run -n mpc_vla python run_ablation.py --output-dir results/my_ablation --epochs 100
        """,
    )
    parser.add_argument(
        "--benchmark", type=str, default="all",
        choices=["pusht", "reaching", "reaching_cluttered", "all"],
        help="Benchmark to run (default: all = pusht + reaching)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Random seeds (default: [0, 1, 2])",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of evaluation episodes per variant (default: 50)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Training epochs (default: 50)",
    )
    parser.add_argument(
        "--num-demos", type=int, default=None,
        help="Number of expert demos to collect (default: 30)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: experiments/EXP-001-mechanism-ablation/outputs/)",
    )

    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
