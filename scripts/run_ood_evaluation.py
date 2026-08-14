#!/usr/bin/env python3
"""EXP-003: Out-of-Distribution (OOD) robustness evaluation.

This script evaluates the eight controllers from EXP-002 under five
perturbation levels on the reaching / reaching_cluttered / pusht benchmarks:

    L0 : in-distribution (clean)
    L1 : RGB color jitter (+/- 15% per channel)
    L2 : spatial initial offset / camera shift (+/- 15% of workspace)
    L3 : obstacle density increase (+30% extra obstacles in cluttered reaching)
    L4 : combined perturbation

Controllers
-----------
* Linear MPC
* Nonlinear MPC (iLQR)
* Collision-Free MPC
* Diffusion Warm-Start MPC
* SmallVLA
* DDPM
* Flow Matching
* MIP (Minimal Iterative Policy)

Outputs
-------
* {output_dir}/ood_results.json
* {output_dir}/ood_aggregated.csv
* {output_dir}/figures/ood_success_rate.png
* {output_dir}/figures/ood_latency.png
* {output_dir}/figures/ood_degradation.png

Example quick test::

    conda run -n mpc_vla python scripts/run_ood_evaluation.py \\
        --benchmarks reaching --seeds 0 --episodes 5 \\
        --output-dir results/exp003_quick
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
STUDY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY_ROOT not in sys.path:
    sys.path.insert(0, STUDY_ROOT)
for _d in [
    "mpc_baselines_repo",
    "mpc_baselines_repo/src",
    "vla_baselines",
    "diffusion_baselines",
    "benchmarks",
]:
    _full = os.path.join(STUDY_ROOT, _d)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)

# ---------------------------------------------------------------------------
# Optional / delayed imports
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as _mpl_err:  # pragma: no cover
    plt = None

try:
    import scipy.ndimage

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import torch
except Exception as _torch_err:  # pragma: no cover
    torch = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# ---------------------------------------------------------------------------
# Benchmarks & baselines (with soft failures)
# ---------------------------------------------------------------------------
ReachingEnv = None
PushTEnv = None
ReachingObstacle = None

try:
    from benchmarks import ReachingEnv, PushTEnv, Obstacle as ReachingObstacle
except Exception as exc:
    warnings.warn(f"Could not import benchmark envs: {exc}")

SmallVLA = None
try:
    from vla_baselines import SmallVLA
except Exception as exc:
    warnings.warn(f"Could not import SmallVLA: {exc}")

DiffusionPolicy = None
FlowMatchingPolicy = None
IterativeRegressionPolicy = None
try:
    from diffusion_baselines import DiffusionPolicy, FlowMatchingPolicy, IterativeRegressionPolicy
except Exception as exc:
    warnings.warn(f"Could not import diffusion policies: {exc}")

LinearMPC = None
NonlinearMPC = None
CollisionFreeMPC = None
DiffusionWarmStartMPC = None
MinimalIterativePolicy = None
PointMass2D = None
SDFWorld = None
CircleObstacle = None
try:
    from src.linear_mpc import LinearMPC
    from src.nonlinear_mpc import NonlinearMPC
    from src.collision_free_mpc import CollisionFreeMPC
    from src.diffusion_warm_start import DiffusionWarmStartMPC, MinimalIterativePolicy
    from src.utils.dynamics import PointMass2D
    from src.utils.obstacles import CircleObstacle
    from src.collision_free_mpc.sdf_world import SDFWorld
except Exception as exc:
    warnings.warn(f"Could not import MPC baselines: {exc}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTROLLER_NAMES: List[str] = [
    "Linear MPC",
    "Nonlinear MPC (iLQR)",
    "Collision-Free MPC",
    "Diffusion Warm-Start MPC",
    "SmallVLA",
    "DDPM",
    "Flow Matching",
    "MIP",
]

BENCHMARK_NAMES: List[str] = ["reaching", "reaching_cluttered", "pusht"]

PERTURBATION_LEVELS: List[int] = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_arg: str) -> Any:
    """Resolve a device string (or 'auto') into a torch device object."""
    if torch is None:
        return None
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load system config if available."""
    cfg: Dict[str, Any] = {}
    if config_path is None or not os.path.exists(config_path):
        config_path = os.path.join(STUDY_ROOT, "configs", "system_config.yaml")
    if yaml is not None and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            warnings.warn(f"Could not load config {config_path}: {exc}")
    return cfg


def _get_workspace(env: Any) -> float:
    """Return a workspace half-extent suitable for perturbation scaling."""
    if hasattr(env, "workspace"):
        w = float(env.workspace)
        if hasattr(env, "block_size"):  # PushT: actual coordinate range is +/- workspace/2
            return w / 2.0
        return w
    return 5.0


def _get_benchmark_obstacles(benchmark: str, cfg: Dict[str, Any]) -> List[Any]:
    """Return the canonical base obstacle list for a benchmark."""
    if ReachingObstacle is None:
        return []
    if benchmark == "reaching":
        raw = cfg.get("benchmarks", {}).get("reaching", {}).get("obstacles", [])
        if raw:
            return [ReachingObstacle(o["center"], o["radius"]) for o in raw]
        # fallback from config file did not parse dicts? use explicit defaults
        return [
            ReachingObstacle([2.0, 2.0], 0.5),
            ReachingObstacle([3.0, 1.0], 0.4),
        ]
    if benchmark == "reaching_cluttered":
        raw = cfg.get("benchmarks", {}).get("reaching_cluttered", {}).get("obstacles", [])
        if raw:
            return [ReachingObstacle(o["center"], o["radius"]) for o in raw]
        return [
            ReachingObstacle([1.5, 1.5], 0.4),
            ReachingObstacle([2.5, 2.5], 0.5),
            ReachingObstacle([3.0, 1.0], 0.35),
            ReachingObstacle([1.0, 3.0], 0.3),
            ReachingObstacle([3.5, 3.0], 0.3),
        ]
    return []


def _make_env(benchmark: str, cfg: Dict[str, Any], obstacles: Optional[List[Any]] = None):
    """Create a benchmark environment instance."""
    if benchmark in ("reaching", "reaching_cluttered"):
        if ReachingEnv is None:
            raise RuntimeError("ReachingEnv not available")
        max_steps = cfg.get("experiment", {}).get("default_max_steps", 60)
        goal_tol = cfg.get("experiment", {}).get("goal_tolerance", 0.2)
        return ReachingEnv(
            dim=2,
            dt=0.05,
            success_threshold=goal_tol,
            max_steps=max_steps,
            image_size=96,
            obstacles=obstacles or [],
            workspace=5.0,
            seed=None,
        )
    if benchmark == "pusht":
        if PushTEnv is None:
            raise RuntimeError("PushTEnv not available")
        max_steps = cfg.get("experiment", {}).get("default_max_steps", 60)
        return PushTEnv(
            block_size=1.0,
            agent_radius=0.15,
            success_iou_threshold=0.6,
            max_steps=max_steps,
            image_size=96,
            workspace=10.0,
            seed=None,
        )
    raise ValueError(f"Unknown benchmark: {benchmark}")


# ---------------------------------------------------------------------------
# Dynamics & cost helpers for reaching MPC
# ---------------------------------------------------------------------------

def _reaching_dynamics(x: np.ndarray, u: np.ndarray, dt: float, mass: float = 1.0) -> np.ndarray:
    """Double-integrator step matching ReachingEnv (no workspace clipping)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    pos = x[:2]
    vel = x[2:]
    new_vel = vel + (u / mass) * dt
    new_pos = pos + vel * dt + 0.5 * (u / mass) * (dt ** 2)
    return np.concatenate([new_pos, new_vel])


def _linearized_reaching_dynamics(dt: float, mass: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Exact A, B matrices for the _reaching_dynamics."""
    A = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    B = np.array([
        [0.5 * (dt ** 2) / mass, 0.0],
        [0.0, 0.5 * (dt ** 2) / mass],
        [dt / mass, 0.0],
        [0.0, dt / mass],
    ], dtype=np.float64)
    return A, B


def _make_reaching_costs(env: Any, Q: np.ndarray, R: np.ndarray, P: np.ndarray):
    """Return stage / terminal cost functions that read env.get_target() each call."""
    dim = env.dim

    def _goal_state():
        target = env.get_target()
        return np.concatenate([np.asarray(target, dtype=float).reshape(dim), np.zeros(dim)])

    def stage_cost(x: np.ndarray, u: np.ndarray, k: Optional[int] = None) -> float:
        g = _goal_state()
        dx = np.asarray(x, dtype=float).reshape(-1) - g
        u = np.asarray(u, dtype=float).reshape(-1)
        return float(dx @ Q @ dx + u @ R @ u)

    def terminal_cost(x: np.ndarray) -> float:
        g = _goal_state()
        dx = np.asarray(x, dtype=float).reshape(-1) - g
        return float(dx @ P @ dx)

    return stage_cost, terminal_cost


# ---------------------------------------------------------------------------
# Learned checkpoint loading
# ---------------------------------------------------------------------------

def _find_checkpoint(
    controller_key: str,
    benchmark: str,
    study_root: str,
    extensions: Tuple[str, ...] = (".pt", ".npz"),
) -> Optional[str]:
    """Find a trained checkpoint for (controller_key, benchmark).

    Searches results/checkpoints/ first, then dist/hf_models/ as a fallback.
    """
    candidates: List[str] = []

    # Results/checkpoints (release-quality)
    cp_dir = os.path.join(study_root, "results", "checkpoints")
    for ext in extensions:
        candidates.append(os.path.join(cp_dir, f"{controller_key}_{benchmark}{ext}"))
        candidates.append(os.path.join(cp_dir, f"{controller_key}{ext}"))  # generic

    # dist/hf_models (quick smoke-test checkpoints)
    dist_dir = os.path.join(study_root, "dist", "hf_models")
    name_map = {
        "small_vla": "small_vla_quick.pt",
        "ddpm": "diffusion_policy_quick.pt",
        "flow_matching": "flow_matching_quick.pt",
        "mip": "mip_quick.pt",
    }
    if controller_key in name_map:
        candidates.append(os.path.join(dist_dir, controller_key, name_map[controller_key]))

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _chunk_action_sequences(
    observations: np.ndarray,
    actions: np.ndarray,
    horizon: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Convert flat transitions into (obs, action_seq) pairs."""
    T = len(observations)
    demos: List[Tuple[np.ndarray, np.ndarray]] = []
    for t in range(T):
        chunk = []
        for h in range(horizon):
            idx = min(t + h, T - 1)
            chunk.append(actions[idx])
        act_seq = np.stack(chunk, axis=0).astype(np.float32)
        demos.append((observations[t].astype(np.float32), act_seq))
    return demos


def _train_mip_fallback(
    env: Any,
    state_dim: int,
    action_dim: int,
    horizon: int,
    hidden_dim: int = 32,
    n_demos: int = 20,
    max_steps: int = 50,
    epochs: int = 20,
) -> Optional[Any]:
    """Train a tiny MinimalIterativePolicy from the env's built-in expert."""
    if MinimalIterativePolicy is None:
        return None
    try:
        demos = env.generate_expert_demonstrations(n_demos=n_demos, max_steps=max_steps)
        chunked = _chunk_action_sequences(
            demos["observations"], demos["actions"], horizon
        )
        if not chunked:
            return None
        mip = MinimalIterativePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            noise_std=0.1,
            seed=0,
        )
        mip.train(chunked, epochs=epochs, batch_size=min(32, len(chunked)), lr=1e-3, verbose=False)
        return mip
    except Exception as exc:
        warnings.warn(f"MIP fallback training failed: {exc}")
        return None


def _load_learned_controllers(
    benchmark: str,
    env: Any,
    cfg: Dict[str, Any],
    device: Any,
    study_root: str,
    horizon: int,
    no_mip_fallback: bool = False,
) -> Dict[str, Any]:
    """Load the four learning-based controllers from checkpoints.

    Returns a dict mapping short keys to model objects (or None if unavailable).
    """
    loaded: Dict[str, Any] = {}
    action_dim = int(env.action_space.shape[0])
    state_dim = int(env.observation_space.shape[0])
    img_size = getattr(env, "image_size", 96)

    # SmallVLA
    try:
        path = _find_checkpoint("small_vla", benchmark, study_root, extensions=(".pt",))
        if path and SmallVLA is not None:
            vla = SmallVLA.load(path, device=device, text_backend="bow")
            if vla.action_dim == action_dim:
                loaded["small_vla"] = vla
    except Exception as exc:
        warnings.warn(f"SmallVLA load failed: {exc}")

    # DDPM
    try:
        path = _find_checkpoint("ddpm", benchmark, study_root, extensions=(".pt",))
        if path and DiffusionPolicy is not None:
            ddpm = DiffusionPolicy.from_checkpoint(path, device=device)
            if ddpm.action_dim == action_dim and (ddpm.obs_dim is None or ddpm.obs_dim == state_dim):
                loaded["ddpm"] = ddpm
    except Exception as exc:
        warnings.warn(f"DDPM load failed: {exc}")

    # Flow Matching
    try:
        path = _find_checkpoint("flow_matching", benchmark, study_root, extensions=(".pt",))
        if path and FlowMatchingPolicy is not None:
            flow = FlowMatchingPolicy.from_checkpoint(path, device=device)
            if flow.action_dim == action_dim and (flow.obs_dim is None or flow.obs_dim == state_dim):
                loaded["flow"] = flow
    except Exception as exc:
        warnings.warn(f"Flow load failed: {exc}")

    # MIP -- try several checkpoint formats
    mip: Optional[Any] = None
    try:
        path = _find_checkpoint("mip", benchmark, study_root, extensions=(".npz", ".pt"))
        if path:
            if path.endswith(".npz") and MinimalIterativePolicy is not None:
                # MinimalIterativePolicy native npz format
                ckpt = np.load(path, allow_pickle=False)
                state_d = int(ckpt.get("state_dim", state_dim))
                action_d = int(ckpt.get("action_dim", action_dim))
                horizon_d = int(ckpt.get("horizon", horizon))
                hidden_d = int(ckpt.get("hidden_dim", 32))
                mip = MinimalIterativePolicy(
                    state_dim=state_d,
                    action_dim=action_d,
                    horizon=horizon_d,
                    hidden_dim=hidden_d,
                    noise_std=float(ckpt.get("noise_std", 0.1)),
                )
                mip.load(path)
                if mip.action_dim != action_dim:
                    mip = None
            elif path.endswith(".pt") and IterativeRegressionPolicy is not None:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                if (
                    isinstance(ckpt, dict)
                    and "state_dict" in ckpt
                    and len(ckpt.get("state_dict", {})) > 0
                ):
                    cfg_ = ckpt.get("config", {})
                    # map config keys (state_dim vs obs_dim)
                    obs_dim = cfg_.get("obs_dim", cfg_.get("state_dim", state_dim))
                    act_dim = cfg_.get("action_dim", action_dim)
                    if act_dim == action_dim and obs_dim == state_dim:
                        mip = IterativeRegressionPolicy.from_checkpoint(path, device=device)
            if mip is not None:
                loaded["mip"] = mip
    except Exception as exc:
        warnings.warn(f"MIP load failed: {exc}")

    # If no MIP checkpoint, train a tiny one on the env's expert (CPU, fast).
    if "mip" not in loaded and MinimalIterativePolicy is not None:
        if benchmark in ("reaching", "reaching_cluttered") and not no_mip_fallback:
            loaded["mip"] = _train_mip_fallback(
                env, state_dim, action_dim, horizon
            )

    return loaded


# ---------------------------------------------------------------------------
# MPC controller construction
# ---------------------------------------------------------------------------

def _build_mpc_controllers(
    env: Any,
    horizon: int,
    cfg: Dict[str, Any],
    loaded: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the four MPC controllers for a reaching-style env.

    Returns a dict mapping short keys to controller objects.
    """
    controllers: Dict[str, Any] = {}
    if LinearMPC is None or env is None:
        return controllers

    # Only build MPC for point-mass reaching environments
    if not hasattr(env, "dim") or not hasattr(env, "workspace"):
        return controllers

    dim = int(env.dim)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    dt = float(env.dt)
    workspace = float(env.workspace)

    # Cost / control bounds
    Q = np.diag([10.0] * dim + [1.0] * dim)
    R = np.diag([0.1] * action_dim)
    P = np.diag([100.0] * dim + [10.0] * dim)
    u_lo = env.action_space.low
    u_hi = env.action_space.high
    x_lo = np.full(state_dim, -workspace)
    x_hi = np.full(state_dim, workspace)

    A, B = _linearized_reaching_dynamics(dt)
    dynamics = lambda x, u: _reaching_dynamics(x, u, dt)
    stage_cost, terminal_cost = _make_reaching_costs(env, Q, R, P)

    # Linear MPC
    try:
        lmpc = LinearMPC(
            A, B, Q, R, P, horizon,
            x_bounds=(x_lo, x_hi), u_bounds=(u_lo, u_hi)
        )
        controllers["linear_mpc"] = lmpc
    except Exception as exc:
        warnings.warn(f"Linear MPC build failed: {exc}")

    # Nonlinear MPC
    try:
        nmpc = NonlinearMPC(
            dynamics, stage_cost, terminal_cost, horizon,
            u_bounds=(u_lo, u_hi),
        )
        controllers["nonlinear_mpc"] = nmpc
    except Exception as exc:
        warnings.warn(f"Nonlinear MPC build failed: {exc}")

    # Collision-Free MPC
    try:
        sdf = SDFWorld(dim=dim)
        for obs in env.obstacles:
            sdf.add_sphere(obs.center, obs.radius)
        cfmpc = CollisionFreeMPC(
            dynamics, stage_cost, terminal_cost, sdf,
            horizon=horizon,
            u_bounds=(u_lo, u_hi),
            collision_weight=100.0,
            ilqr_iters=cfg.get("experiment", {}).get("ilqr_iters", 10),
        )
        controllers["collision_free_mpc"] = cfmpc
    except Exception as exc:
        warnings.warn(f"Collision-Free MPC build failed: {exc}")

    # Diffusion Warm-Start MPC (wrap the loaded DDPM + a fresh CFMPC)
    if "ddpm" in loaded and "collision_free_mpc" in controllers:
        try:
            ddpm_model = loaded["ddpm"]

            class _NumpyDiffusionWrapper:
                def __init__(self, policy: Any):
                    self.policy = policy

                def sample(self, state: np.ndarray, num_samples: int = 1) -> np.ndarray:
                    with torch.no_grad() if torch is not None else warnings.catch_warnings():
                        actions = self.policy.sample(state, num_samples=num_samples)
                    if isinstance(actions, torch.Tensor):
                        actions = actions.detach().cpu().numpy()
                    return actions

            diff_ws = _NumpyDiffusionWrapper(ddpm_model)
            # Build a dedicated CFMPC with horizon matching the DDPM chunk
            ddpm_horizon = ddpm_model.horizon
            sdf2 = SDFWorld(dim=dim)
            for obs in env.obstacles:
                sdf2.add_sphere(obs.center, obs.radius)
            cfmpc2 = CollisionFreeMPC(
                dynamics, stage_cost, terminal_cost, sdf2,
                horizon=ddpm_horizon,
                u_bounds=(u_lo, u_hi),
                collision_weight=100.0,
                ilqr_iters=cfg.get("experiment", {}).get("ilqr_iters", 10),
            )
            cf_obstacles = [CircleObstacle(obs.center, obs.radius) for obs in env.obstacles]
            hybrid = DiffusionWarmStartMPC(
                diffusion_policy=diff_ws,
                mpc_controller=cfmpc2,
                num_diffusion_samples=cfg.get("network_sizes", {}).get("small", {}).get("num_diffusion_samples", 4),
                refine_steps=cfg.get("network_sizes", {}).get("small", {}).get("refine_steps", 3),
                obstacles=cf_obstacles,
            )
            controllers["diffusion_warm_start_mpc"] = hybrid
        except Exception as exc:
            warnings.warn(f"Diffusion Warm-Start MPC build failed: {exc}")

    return controllers


# ---------------------------------------------------------------------------
# Controller dispatch
# ---------------------------------------------------------------------------

def _build_controller_dispatch(
    env: Any,
    loaded: Dict[str, Any],
    mpc: Dict[str, Any],
    horizon: int,
) -> Dict[str, Dict[str, Any]]:
    """Return a dict of runnable controllers with unified predict() interface."""
    controllers: Dict[str, Dict[str, Any]] = {}
    state_dim = int(env.observation_space.shape[0])

    def _add(name: str, is_image: bool, predict_fn: Callable):
        controllers[name] = {"is_image": is_image, "predict": predict_fn}

    # Linear MPC
    if "linear_mpc" in mpc:
        lmpc = mpc["linear_mpc"]

        def _linear_predict(obs: np.ndarray, env_: Any) -> np.ndarray:
            target = env_.get_target()
            ref_pos = np.concatenate([target, np.zeros(env_.dim)])
            ref = np.tile(ref_pos, (horizon + 1, 1))
            try:
                result = lmpc.solve(obs, ref)
                return result.control
            except Exception:
                # Fallback: proportional drive toward the goal
                goal = np.concatenate([target, np.zeros(env_.dim)])
                err = goal - obs
                return np.clip(5.0 * err[:2] - 2.0 * obs[2:4], -1.0, 1.0)

        _add("Linear MPC", False, _linear_predict)

    # Nonlinear MPC
    if "nonlinear_mpc" in mpc:
        nmpc = mpc["nonlinear_mpc"]

        def _nonlinear_predict(obs: np.ndarray, env_: Any) -> np.ndarray:
            try:
                result = nmpc.solve(obs, max_iter=20)
                return result["action"]
            except Exception:
                target = env_.get_target()
                err = target - obs[:2]
                return np.clip(5.0 * err - 2.0 * obs[2:4], -1.0, 1.0)

        _add("Nonlinear MPC (iLQR)", False, _nonlinear_predict)

    # Collision-Free MPC
    if "collision_free_mpc" in mpc:
        cfmpc = mpc["collision_free_mpc"]

        def _cf_predict(obs: np.ndarray, env_: Any) -> np.ndarray:
            try:
                U = cfmpc.solve(obs)
                return U[0]
            except Exception:
                target = env_.get_target()
                err = target - obs[:2]
                return np.clip(5.0 * err - 2.0 * obs[2:4], -1.0, 1.0)

        _add("Collision-Free MPC", False, _cf_predict)

    # Diffusion Warm-Start MPC
    if "diffusion_warm_start_mpc" in mpc:
        hybrid = mpc["diffusion_warm_start_mpc"]

        def _diffws_predict(obs: np.ndarray, env_: Any) -> np.ndarray:
            try:
                goal = np.concatenate([env_.get_target(), np.zeros(env_.dim)])
                action, *_ = hybrid.solve(obs, goal)
                return action
            except Exception:
                target = env_.get_target()
                err = target - obs[:2]
                return np.clip(5.0 * err - 2.0 * obs[2:4], -1.0, 1.0)

        _add("Diffusion Warm-Start MPC", False, _diffws_predict)

    # SmallVLA
    if "small_vla" in loaded:
        vla = loaded["small_vla"]

        def _vla_predict(img: np.ndarray, env_: Any) -> np.ndarray:
            instr = env_.get_language_instruction()
            actions = vla.predict_action(img, instr)
            if actions.ndim == 2:
                return actions[0]
            return actions

        _add("SmallVLA", True, _vla_predict)

    # DDPM
    if "ddpm" in loaded:
        ddpm = loaded["ddpm"]

        def _ddpm_predict(obs: np.ndarray, env_: Any) -> np.ndarray:
            actions = ddpm.sample(obs, num_samples=1)
            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu().numpy()
            # (num_samples, horizon, action_dim) or (horizon, action_dim)
            if actions.ndim == 3:
                return actions[0, 0]
            return actions[0]

        _add("DDPM", False, _ddpm_predict)

    # Flow Matching
    if "flow" in loaded:
        flow = loaded["flow"]

        def _flow_predict(obs: np.ndarray, env_: Any) -> np.ndarray:
            actions = flow.sample(obs, num_samples=1)
            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu().numpy()
            if actions.ndim == 3:
                return actions[0, 0]
            return actions[0]

        _add("Flow Matching", False, _flow_predict)

    # MIP
    if "mip" in loaded:
        mip_model = loaded["mip"]

        def _mip_predict(obs: np.ndarray, env_: Any) -> np.ndarray:
            if isinstance(mip_model, IterativeRegressionPolicy):
                actions = mip_model.predict(obs)
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().numpy()
                return actions[0]
            else:
                actions = mip_model.sample(obs, num_samples=1)
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().numpy()
                if actions.ndim == 3:
                    return actions[0, 0]
                return actions[0]

        _add("MIP", False, _mip_predict)

    return controllers


# ---------------------------------------------------------------------------
# Perturbation wrapper
# ---------------------------------------------------------------------------

class Perturbation:
    """Observation-level perturbation wrapper.

    Applies L1 color jitter, L2 spatial shift, and L4 combined perturbations to
    an observation before it is passed to a controller.  L3 (obstacle density)
    is an environment modification handled outside this wrapper.
    """

    def __init__(
        self,
        level: int,
        is_image: bool,
        workspace: float,
        img_size: int,
        rng: np.random.Generator,
        n_pos: int = 2,
    ) -> None:
        self.level = int(level)
        self.is_image = bool(is_image)
        self.workspace = float(workspace)
        self.img_size = int(img_size)
        self.rng = rng
        self.n_pos = int(n_pos)

        # Precompute fixed offsets for the episode
        self.state_offset: Optional[np.ndarray] = None
        if self.level in (2, 4) and not self.is_image:
            offset = self.rng.uniform(
                -0.15 * self.workspace, 0.15 * self.workspace, size=self.n_pos
            )
            self.state_offset = np.zeros(self.n_pos, dtype=np.float64)
            self.state_offset[:self.n_pos] = offset

        self.image_shift: Optional[Tuple[int, int]] = None
        if self.level in (2, 4) and self.is_image:
            max_shift = int(round(0.15 * self.img_size))
            if max_shift > 0:
                self.image_shift = (
                    int(self.rng.integers(-max_shift, max_shift + 1)),
                    int(self.rng.integers(-max_shift, max_shift + 1)),
                )
            else:
                self.image_shift = (0, 0)

    def perturb_observation(self, obs: np.ndarray) -> np.ndarray:
        """Return a perturbed observation."""
        if self.is_image:
            return self._perturb_image(obs)
        return self._perturb_state(obs)

    def _perturb_state(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).copy()
        if self.level in (2, 4) and self.state_offset is not None:
            obs[: self.n_pos] += self.state_offset
        if self.level == 4:
            # L4: additional small state observation noise (combined)
            noise = self.rng.normal(0.0, 0.02 * self.workspace, size=obs.shape)
            obs += noise
        return obs.astype(np.float32)

    def _perturb_image(self, img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, dtype=np.float32).copy()

        # L1 / L4: RGB color jitter (+/- 15% per channel)
        if self.level in (1, 4):
            jitter = self.rng.uniform(0.85, 1.15, size=3)
            img = np.clip(img * jitter, 0.0, 255.0)

        # L2 / L4: image shift to simulate camera shift
        if self.level in (2, 4) and self.image_shift is not None:
            dy, dx = self.image_shift
            if _HAS_SCIPY:
                img = scipy.ndimage.shift(img, (dy, dx, 0), mode="nearest", order=1)
            else:
                # Fallback: simple np.roll (wrap-around)
                img = np.roll(img, (dy, dx), axis=(0, 1))

        return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Environment obstacle perturbation (L3)
# ---------------------------------------------------------------------------

def _generate_extra_obstacles(
    env: Any,
    level: int,
    rng: np.random.Generator,
    workspace: float,
) -> List[Any]:
    """Generate +30% extra obstacles for the reaching benchmarks (L3/L4)."""
    if level not in (3, 4):
        return []
    if not hasattr(env, "obstacles") or ReachingObstacle is None:
        return []

    base = list(env.obstacles)
    n_base = len(base)
    n_extra = max(1, int(math.ceil(0.3 * n_base))) if n_base > 0 else 2
    extra: List[Any] = []

    for _ in range(n_extra):
        for _attempt in range(50):
            center = rng.uniform(-workspace, workspace, size=2)
            radius = rng.uniform(0.2, 0.5)
            too_close = any(
                np.linalg.norm(center - o.center) < (radius + o.radius + 0.1)
                for o in base + extra
            )
            if not too_close:
                extra.append(ReachingObstacle(center, radius))
                break
    return extra


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def _run_episode(
    env: Any,
    controller: Dict[str, Any],
    perturbation: Perturbation,
    max_steps: int,
    seed: int,
) -> Dict[str, float]:
    """Run one evaluation episode and return metrics."""
    env.reset(seed=seed)

    total_reward = 0.0
    path_length = 0.0
    latencies: List[float] = []
    collided = False
    succeeded = False

    for step in range(max_steps):
        if controller["is_image"]:
            raw = env.get_image()
        else:
            raw = env.get_observation()
        obs = perturbation.perturb_observation(raw)

        t0 = time.perf_counter()
        try:
            action = controller["predict"](obs, env)
        except Exception as exc:
            warnings.warn(f"Controller raised exception at step {step}: {exc}")
            break
        latency = time.perf_counter() - t0
        latencies.append(latency)

        action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        _, reward, done, _ = env.step(action)
        total_reward += float(reward)
        path_length += float(np.linalg.norm(action))

        if env.is_collision():
            collided = True
        if env.is_success():
            succeeded = True
        if done:
            break

    return {
        "success": float(succeeded),
        "collision": float(collided),
        "return": total_reward,
        "path_length": path_length,
        "mean_latency": float(np.mean(latencies)) if latencies else 0.0,
        "steps": env.step_count,
    }


# ---------------------------------------------------------------------------
# Aggregation, saving, plotting
# ---------------------------------------------------------------------------

def _aggregate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate episode records to (benchmark, controller, level) rows."""
    from collections import defaultdict

    grouped: Dict[Tuple[str, str, int], List[Dict[str, float]]] = defaultdict(list)
    for r in records:
        key = (r["benchmark"], r["controller"], r["level"])
        grouped[key].append(r)

    rows: List[Dict[str, Any]] = []
    for (bench, ctrl, level), episodes in grouped.items():
        sr = [e["success"] for e in episodes]
        pl = [e["path_length"] for e in episodes]
        cr = [e["collision"] for e in episodes]
        lat = [e["mean_latency"] for e in episodes]
        rows.append({
            "benchmark": bench,
            "controller": ctrl,
            "level": level,
            "n_episodes": len(episodes),
            "n_seeds": len({e["seed"] for e in episodes}),
            "success_rate_mean": float(np.mean(sr)),
            "success_rate_std": float(np.std(sr)) if len(sr) > 1 else 0.0,
            "path_length_mean": float(np.mean(pl)),
            "path_length_std": float(np.std(pl)) if len(pl) > 1 else 0.0,
            "collision_rate_mean": float(np.mean(cr)),
            "latency_ms_mean": float(np.mean(lat)) * 1000.0,
            "latency_ms_std": float(np.std(lat)) * 1000.0 if len(lat) > 1 else 0.0,
        })
    return rows


def _save_json(records: List[Dict[str, Any]], output_dir: str, config: Dict[str, Any]):
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "config": config,
        "records": records,
    }
    path = os.path.join(output_dir, "ood_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"Saved {path}")


def _save_csv(rows: List[Dict[str, Any]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ood_aggregated.csv")
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("benchmark,controller,level,n_episodes,n_seeds\n")
        return
    keys = [
        "benchmark", "controller", "level", "n_episodes", "n_seeds",
        "success_rate_mean", "success_rate_std",
        "path_length_mean", "path_length_std",
        "collision_rate_mean",
        "latency_ms_mean", "latency_ms_std",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def _plot_results(rows: List[Dict[str, Any]], output_dir: str):
    if plt is None or not rows:
        warnings.warn("Matplotlib unavailable; skipping plots")
        return
    os.makedirs(output_dir, exist_ok=True)
    benchmarks = sorted({r["benchmark"] for r in rows})
    controllers = sorted({r["controller"] for r in rows})
    levels = sorted({r["level"] for r in rows})

    # Success rate vs level, one figure per benchmark
    for bench in benchmarks:
        bench_rows = [r for r in rows if r["benchmark"] == bench]
        fig, ax = plt.subplots(figsize=(8, 5))
        for ctrl in controllers:
            ctrl_rows = [r for r in bench_rows if r["controller"] == ctrl]
            ctrl_rows = sorted(ctrl_rows, key=lambda r: r["level"])
            xs = [r["level"] for r in ctrl_rows]
            ys = [r["success_rate_mean"] for r in ctrl_rows]
            if ys:
                ax.plot(xs, ys, marker="o", label=ctrl)
        ax.set_xlabel("Perturbation level")
        ax.set_ylabel("Success rate")
        ax.set_title(f"OOD success rate ({bench})")
        ax.set_xticks(levels)
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"ood_success_rate_{bench}.png"), dpi=150)
        plt.close(fig)

    # Overall success rate (combined panel)
    _plot_grouped_bars(rows, output_dir, "success_rate_mean", "Success rate", "ood_success_rate.png")
    # Latency
    _plot_grouped_bars(rows, output_dir, "latency_ms_mean", "Mean latency (ms)", "ood_latency.png")
    # Degradation (drop from L0)
    _plot_degradation(rows, output_dir)


def _plot_grouped_bars(
    rows: List[Dict[str, Any]],
    output_dir: str,
    value_key: str,
    ylabel: str,
    filename: str,
):
    if plt is None or not rows:
        return
    benchmarks = sorted({r["benchmark"] for r in rows})
    controllers = sorted({r["controller"] for r in rows})
    levels = sorted({r["level"] for r in rows})

    n_benches = len(benchmarks)
    fig, axes = plt.subplots(1, n_benches, figsize=(5 * n_benches, 5), sharey=True)
    if n_benches == 1:
        axes = [axes]

    x = np.arange(len(levels))
    width = 0.8 / max(1, len(controllers))

    for ax, bench in zip(axes, benchmarks):
        for i, ctrl in enumerate(controllers):
            values = []
            for lvl in levels:
                matches = [r for r in rows if r["benchmark"] == bench and r["controller"] == ctrl and r["level"] == lvl]
                values.append(matches[0][value_key] if matches else 0.0)
            ax.bar(x + (i - len(controllers) / 2) * width + width / 2, values, width, label=ctrl)
        ax.set_xlabel("Perturbation level")
        ax.set_ylabel(ylabel)
        ax.set_title(bench)
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in levels])
        ax.legend(fontsize="small", loc="best")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def _plot_degradation(rows: List[Dict[str, Any]], output_dir: str):
    if plt is None or not rows:
        return
    benchmarks = sorted({r["benchmark"] for r in rows})
    controllers = sorted({r["controller"] for r in rows})
    levels = sorted({r["level"] for r in rows})

    n_benches = len(benchmarks)
    fig, axes = plt.subplots(1, n_benches, figsize=(5 * n_benches, 5), sharey=True)
    if n_benches == 1:
        axes = [axes]

    for ax, bench in zip(axes, benchmarks):
        for ctrl in controllers:
            baseline = None
            ys = []
            xs = []
            for lvl in levels:
                matches = [r for r in rows if r["benchmark"] == bench and r["controller"] == ctrl and r["level"] == lvl]
                if not matches:
                    continue
                val = matches[0]["success_rate_mean"]
                if lvl == 0:
                    baseline = val
                if baseline is not None:
                    ys.append(baseline - val)
                    xs.append(lvl)
            if ys:
                ax.plot(xs, ys, marker="o", label=ctrl)
        ax.set_xlabel("Perturbation level")
        ax.set_ylabel("Success-rate drop from L0")
        ax.set_title(f"OOD degradation ({bench})")
        ax.set_xticks(levels)
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ood_degradation.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experimental loop
# ---------------------------------------------------------------------------

def _parse_list_arg(arg: str, defaults: List[str]) -> List[str]:
    if arg == "all":
        return list(defaults)
    return [s.strip() for s in arg.split(",") if s.strip()]


def _parse_int_list_arg(arg: str, defaults: List[int]) -> List[int]:
    if arg == "all":
        return list(defaults)
    return [int(s.strip()) for s in arg.split(",") if s.strip()]


def run_experiment(args: argparse.Namespace):
    cfg = _load_config(args.config)
    device = _resolve_device(args.device)

    benchmarks = _parse_list_arg(args.benchmarks, BENCHMARK_NAMES)
    controllers_arg = _parse_list_arg(args.controllers, CONTROLLER_NAMES)
    levels = _parse_int_list_arg(args.perturbation_levels, PERTURBATION_LEVELS)
    seeds = args.seeds if args.seeds is not None else cfg.get("experiment", {}).get("default_seeds", [0, 1, 2, 42, 123])
    n_episodes = args.episodes
    max_steps = args.max_steps or cfg.get("experiment", {}).get("default_max_steps", 60)
    horizon = args.horizon or cfg.get("experiment", {}).get("default_horizon", 8)

    # Auto-detect a quick smoke test and use a small horizon to keep it fast.
    if (
        n_episodes <= 5
        and len(seeds) <= 1
        and args.benchmarks != "all"
        and args.horizon is None
    ):
        horizon = 8

    output_dir = args.output_dir or os.path.join(STUDY_ROOT, "results", "exp003")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 80)
    print("EXP-003: Out-of-Distribution Robustness Evaluation")
    print("=" * 80)
    print(f"  Benchmarks:        {benchmarks}")
    print(f"  Controllers:       {controllers_arg}")
    print(f"  Perturbation levels: {levels}")
    print(f"  Seeds:             {seeds}")
    print(f"  Episodes/seed:     {n_episodes}")
    print(f"  Max steps:         {max_steps}")
    print(f"  Horizon:           {horizon}")
    print(f"  Device:            {device}")
    print(f"  Output dir:        {output_dir}")
    print("=" * 80)

    records: List[Dict[str, Any]] = []

    for benchmark in benchmarks:
        print(f"\nBenchmark: {benchmark}")
        base_obstacles = _get_benchmark_obstacles(benchmark, cfg)

        # Load learned controllers once for this benchmark (on a clean env).
        clean_env = _make_env(benchmark, cfg, obstacles=base_obstacles)
        loaded = _load_learned_controllers(benchmark, clean_env, cfg, device, STUDY_ROOT, horizon, args.no_mip_fallback)

        for level in levels:
            print(f"  Level {level} ...")
            for seed in seeds:
                # One env per (level, seed); reused across episodes.  MPCs are
                # built once so the SDF/costs are shared, while per-episode
                # resets give new start/target.  Extra obstacles (L3/L4) are
                # fixed for the seed.
                env_seed = int(seed * 10000 + level * 1000)
                obs_env = _make_env(benchmark, cfg, obstacles=base_obstacles)

                if level in (3, 4):
                    rng_obs = np.random.default_rng(env_seed)
                    extra = _generate_extra_obstacles(
                        obs_env, level, rng_obs, _get_workspace(obs_env)
                    )
                    if extra:
                        obs_env.obstacles.extend(extra)

                mpc = _build_mpc_controllers(obs_env, horizon, cfg, loaded)
                available = _build_controller_dispatch(obs_env, loaded, mpc, horizon)
                active = {k: v for k, v in available.items() if k in controllers_arg}

                workspace = _get_workspace(obs_env)
                img_size = getattr(obs_env, "image_size", 96)

                for ep in range(n_episodes):
                    ep_seed = int(seed * 10000 + level * 1000 + ep)

                    for ci, (ctrl_name, ctrl) in enumerate(active.items()):
                        obs_env.reset(seed=ep_seed)

                        # Per-controller random stream for the observation perturbation
                        pert_rng = np.random.default_rng(ep_seed + (ci + 1) * 31)
                        n_pos = min(2, obs_env.observation_space.shape[0] // 2)
                        perturbation = Perturbation(
                            level, ctrl["is_image"], workspace, img_size, pert_rng, n_pos=n_pos
                        )

                        try:
                            metrics = _run_episode(obs_env, ctrl, perturbation, max_steps, ep_seed)
                        except Exception as exc:
                            warnings.warn(f"Episode failed for {ctrl_name}: {exc}")
                            traceback.print_exc()
                            metrics = {
                                "success": 0.0,
                                "collision": 1.0,
                                "return": 0.0,
                                "path_length": 0.0,
                                "mean_latency": 0.0,
                                "steps": 0,
                            }

                        records.append({
                            "benchmark": benchmark,
                            "controller": ctrl_name,
                            "level": level,
                            "seed": seed,
                            "episode": ep,
                            "success": metrics["success"],
                            "collision": metrics["collision"],
                            "return": metrics["return"],
                            "path_length": metrics["path_length"],
                            "mean_latency": metrics["mean_latency"],
                            "steps": metrics["steps"],
                        })

    # Save results
    _save_json(records, output_dir, {
        "benchmarks": benchmarks,
        "controllers": controllers_arg,
        "levels": levels,
        "seeds": seeds,
        "episodes": n_episodes,
        "max_steps": max_steps,
        "horizon": horizon,
        "device": str(device),
    })
    rows = _aggregate_records(records)
    _save_csv(rows, output_dir)
    _plot_results(rows, figures_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print("Aggregated OOD Results (mean over episodes / seeds)")
    print("=" * 80)
    for bench in sorted({r["benchmark"] for r in rows}):
        print(f"\n{bench}:")
        for ctrl in sorted({r["controller"] for r in rows if r["benchmark"] == bench}):
            ctrl_rows = [r for r in rows if r["benchmark"] == bench and r["controller"] == ctrl]
            ctrl_rows = sorted(ctrl_rows, key=lambda r: r["level"])
            line = f"  {ctrl:<32} " + " ".join(
                f"L{r['level']}={r['success_rate_mean']:.2f}" for r in ctrl_rows
            )
            print(line)

    print(f"\nAll outputs saved to: {output_dir}")
    return records, rows


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EXP-003: Out-of-Distribution Robustness Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Comma-separated benchmark list or 'all' (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds (default: [0, 1, 2, 42, 123])",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per (controller, level, seed) (default: 20)",
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default="all",
        help="Comma-separated controller names or 'all' (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/exp003)",
    )
    parser.add_argument(
        "--perturbation-levels",
        type=str,
        default="all",
        help="Comma-separated perturbation levels or 'all' (default: all)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (default: from system_config.yaml)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Prediction horizon for MPC and learned action chunks (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for torch models: 'auto', 'cuda', 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to system_config.yaml",
    )
    parser.add_argument(
        "--no-mip-fallback",
        action="store_true",
        help="Disable on-the-fly MIP training when no checkpoint is found",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more diagnostic messages",
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
