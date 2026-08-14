#!/usr/bin/env python3
"""EXP-004: Latency-Performance Pareto Sweep Runner.

Profiles high-precision inference latency on NVIDIA DGX Spark (GB10) across:
  - DDPM: T in [1, 2, 4, 8, 16, 32, 64, 100]
  - Flow Matching: T in [1, 2, 4, 8, 10]
  - MIP (Minimal Iterative Policy): iters in [1, 2, 3, 4, 5]
  - Linear MPC: horizons in [10, 20]
  - Nonlinear MPC: iLQR iters in [5, 15, 30]
  - CollisionFree MPC
  - SmallVLA
  - Regression (single-step MLP baseline)

Evaluates closed-loop success rates on 2D Reaching and PushT benchmarks,
extracts the Pareto frontier, and generates publication-grade Pareto frontier
and latency breakdown plots.

Usage::

    conda run -n mpc_vla python scripts/run_pareto_sweep.py --output-dir results/exp004
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
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

_MODULE_DIRS = [
    "mpc_baselines_repo",
    "mpc_baselines_repo/src",
    "vla_baselines",
    "diffusion_baselines",
    "benchmarks",
    "src",
]
for _d in _MODULE_DIRS:
    _full = os.path.join(STUDY_ROOT, _d)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)

# Import baselines
from src.utils.dynamics import PointMass2D
from src.utils.obstacles import CircleObstacle, is_in_collision
from src.linear_mpc import LinearMPC
from src.nonlinear_mpc import NonlinearMPC
from src.collision_free_mpc import CollisionFreeMPC, SDFWorld
from diffusion_baselines.ddpm_policy import DiffusionPolicy
from diffusion_baselines.flow_matching_policy import FlowMatchingPolicy
from diffusion_baselines.iterative_regression_policy import IterativeRegressionPolicy
from diffusion_baselines.regression_policy import RegressionPolicy
from vla_baselines.small_vla import SmallVLA
from run_experiments import _FallbackReaching, _FallbackPushT
from scripts.collect_env_info import collect_env_info


# ===========================================================================
# Benchmark Construction
# ===========================================================================
def make_benchmark(name: str, seed: int = 0):
    """Instantiate a benchmark environment."""
    if name.lower() in ("reaching", "reaching_2d"):
        return _FallbackReaching(seed=seed, cluttered=False)
    elif name.lower() in ("reaching_cluttered",):
        return _FallbackReaching(seed=seed, cluttered=True)
    elif name.lower() in ("pusht", "pusht_2d"):
        return _FallbackPushT(seed=seed)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


# ===========================================================================
# Fast Image Renderer for VLA
# ===========================================================================
def render_benchmark_image(bench: Any, state: np.ndarray, size: int = 96) -> np.ndarray:
    """Fast software rasterizer for top-down observation image (no matplotlib overhead)."""
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    scale = size / 10.0  # workspace [-5, 5] -> [0, size]

    def to_px(xy):
        px = int(np.clip((xy[0] + 5.0) * scale, 0, size - 1))
        py = int(np.clip((xy[1] + 5.0) * scale, 0, size - 1))
        return px, py

    gx, gy = np.ogrid[:size, :size]

    # Draw obstacles
    if hasattr(bench, "obstacles"):
        for obs in bench.obstacles:
            cx, cy = to_px(obs.center[:2])
            r = int(obs.radius * scale)
            mask = (gx - cx) ** 2 + (gy - cy) ** 2 <= r ** 2
            img[mask] = [40, 40, 40]

    # Draw goal(s)
    goals = getattr(bench, "goals", [bench.goal])
    for g in goals:
        gx_c, gy_c = to_px(g[:2])
        r_g = int(0.35 * scale)
        mask = (gx - gx_c) ** 2 + (gy - gy_c) ** 2 <= r_g ** 2
        img[mask] = [34, 139, 34]  # green

    # Draw agent
    ax, ay = to_px(state[:2])
    r_a = int(0.25 * scale)
    mask = (gx - ax) ** 2 + (gy - ay) ** 2 <= r_a ** 2
    img[mask] = [220, 20, 60]  # red

    return img


# ===========================================================================
# Demonstration Collection
# ===========================================================================
def collect_demonstrations(
    bench: Any,
    n_demos: int = 30,
    horizon: int = 16,
    seed: int = 0,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Dict[str, Any]]]:
    """Collect expert demonstrations using CollisionFree MPC / iLQR."""
    stage_cost, term_cost, _, _, _ = bench.make_costs()
    u_bounds = (-5.0 * np.ones(bench.action_dim), 5.0 * np.ones(bench.action_dim))

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
    demos_state = []
    demos_vla = []

    for i in range(n_demos):
        s = bench.reset()
        u0 = np.zeros((horizon, bench.action_dim))
        try:
            U = mpc._ilqr(s, u0, bench.action_dim)
            # Check for NaN
            if np.isnan(U).any():
                continue
            demos_state.append((s.copy(), U.copy()))

            # Render VLA demonstration
            img = render_benchmark_image(bench, s, size=96)
            instr = "reach green target while avoiding obstacles"
            demos_vla.append({
                "image": img,
                "instruction": instr,
                "action": U.copy(),
            })
        except Exception:
            continue

    return demos_state, demos_vla


# ===========================================================================
# High-Precision Latency Profiling
# ===========================================================================
def profile_latency(
    call_fn: Callable[[], Any],
    is_cuda: bool = False,
    n_warmup: int = 100,
    n_eval: int = 1000,
) -> Dict[str, float]:
    """Profile inference latency with high precision (100 warmup, 1000 timed calls)."""
    # Warmup calls
    for _ in range(n_warmup):
        call_fn()
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    timings_ns = []
    for _ in range(n_eval):
        if is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        call_fn()
        if is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
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
        "n_samples": n_eval,
    }


# ===========================================================================
# Step Sampling Helpers
# ===========================================================================
def sample_ddpm_step_sweep(
    policy: DiffusionPolicy,
    obs_np: np.ndarray,
    T: int,
    eta: float = 0.0,
) -> np.ndarray:
    """Sample action sequence from trained DDPM policy with T diffusion steps."""
    device = policy.device
    obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
    alpha_bar = policy.schedule.alpha_bar

    if T <= 1:
        timesteps = [99]
    else:
        timesteps = list(np.linspace(0, 99, T, dtype=int))
    timesteps = sorted(list(set(timesteps)), reverse=True)

    x = torch.randn(1, policy.horizon, policy.action_dim, device=device)

    for i, t_val in enumerate(timesteps):
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        pred_eps = policy.net(x, t, global_cond=obs_t)

        ab_t = alpha_bar[t_val]
        x0_est = (x - torch.sqrt(1.0 - ab_t) * pred_eps) / torch.sqrt(ab_t)

        if i == len(timesteps) - 1:
            x = x0_est
        else:
            t_prev = timesteps[i + 1]
            ab_prev = alpha_bar[t_prev]
            if eta > 0.0:
                sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * torch.sqrt(1.0 - ab_t / ab_prev)
                noise = torch.randn_like(x)
                x = torch.sqrt(ab_prev) * x0_est + torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0)) * pred_eps + sigma * noise
            else:
                x = torch.sqrt(ab_prev) * x0_est + torch.sqrt(1.0 - ab_prev) * pred_eps

    return x.squeeze(0).cpu().detach().numpy()


def sample_flow_step_sweep(
    policy: FlowMatchingPolicy,
    obs_np: np.ndarray,
    T: int,
) -> np.ndarray:
    """Sample action sequence from trained Flow Matching policy with T Euler steps."""
    device = policy.device
    obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
    x = torch.randn(1, policy.horizon, policy.action_dim, device=device)

    dt = 1.0 / float(T)
    for i in range(T):
        t_cont = i * dt
        t_idx = torch.full((1,), int(t_cont * 10), device=device, dtype=torch.long).clamp(0, 9)
        pred_v = policy.net(x, t_idx, global_cond=obs_t)
        x = x + dt * pred_v

    return x.squeeze(0).cpu().detach().numpy()


def sample_mip_iteration_sweep(
    policy: IterativeRegressionPolicy,
    obs_np: np.ndarray,
    iters: int,
) -> np.ndarray:
    """Sample action sequence from trained MIP with iters refinement steps."""
    device = policy.device
    obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
    prev = torch.zeros(1, policy.horizon, policy.action_dim, device=device)

    for step in range(iters):
        net = policy.nets[step]
        pred = net(obs_t, prev)
        if step < iters - 1 and policy.noise_std > 0:
            prev = pred + policy.noise_std * torch.randn_like(pred)
        else:
            prev = pred

    return prev.squeeze(0).cpu().detach().numpy()


# ===========================================================================
# Closed-Loop Simulation
# ===========================================================================
def simulate_rollout(
    bench: Any,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a single closed-loop rollout."""
    if max_steps is None:
        max_steps = getattr(bench, "max_steps", 60)

    s = bench.reset()
    path_length = 0.0
    prev_pos = s[:2].copy()
    success = False
    collision = False
    step_latencies = []

    curr_state = s.copy()
    for t in range(max_steps):
        t0 = time.perf_counter_ns()
        act = policy_fn(curr_state)
        t1 = time.perf_counter_ns()
        step_latencies.append((t1 - t0) / 1e6)

        act = np.asarray(act, dtype=float).reshape(-1)
        if len(act) > bench.action_dim:
            act = act[:bench.action_dim]

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


# ===========================================================================
# Pareto Frontier Computation
# ===========================================================================
def compute_pareto_dominance(
    conditions: List[Dict[str, Any]],
    latency_key: str = "latency_p50_ms",
    success_key: str = "success_rate_mean",
) -> List[Dict[str, Any]]:
    """Calculate Pareto dominance counts and mark Pareto-optimal points.
    
    A condition A dominates B if:
      latency(A) <= latency(B) and success(A) >= success(B)
      with at least one strict inequality.
    """
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

            # j dominates i if j has <= latency AND >= success, with strict in at least one
            if (l_j <= l_i and s_j >= s_i) and (l_j < l_i or s_j > s_i):
                dom_count += 1

        conditions[i]["dominance_count"] = dom_count
        conditions[i]["is_pareto_optimal"] = (dom_count == 0)

    return conditions


# ===========================================================================
# Plotting
# ===========================================================================
def plot_pareto_frontier(
    pareto_by_bench: Dict[str, List[Dict[str, Any]]],
    output_path: str,
):
    """Generate high-resolution Pareto frontier plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # Palette and marker dictionary
    family_styles = {
        "MPC": {"color": "#1f77b4", "marker": "s", "label": "Classical MPC"},
        "DDPM": {"color": "#d62728", "marker": "o", "label": "DDPM (T-sweep)"},
        "Flow Matching": {"color": "#9467bd", "marker": "^", "label": "Flow Matching (T-sweep)"},
        "MIP": {"color": "#ff7f0e", "marker": "D", "label": "MIP (iter-sweep)"},
        "VLA": {"color": "#2ca02c", "marker": "*", "label": "SmallVLA"},
        "Regression": {"color": "#7f7f7f", "marker": "X", "label": "Regression (RCP)"},
    }

    benchmarks = list(pareto_by_bench.keys())
    for ax_idx, bn in enumerate(benchmarks):
        ax = axes[ax_idx]
        conds = pareto_by_bench[bn]

        # Plot all points
        for c in conds:
            fam = c["family"]
            style = family_styles.get(fam, {"color": "#333333", "marker": "o", "label": fam})
            
            x = c["latency_p50_ms"]
            y = c["success_rate_mean"]
            y_err = c["success_rate_std"]
            x_err_low = max(0, x - c["latency_mean_ms"] * 0.1) # small visual whisker
            x_err_high = c["latency_p95_ms"] - x

            # Error bars
            ax.errorbar(
                x, y,
                yerr=y_err,
                xerr=[[x_err_low * 0.2], [x_err_high]],
                fmt="none",
                ecolor=style["color"],
                elinewidth=1.0,
                capsize=3,
                alpha=0.6,
            )

            # Scatter point
            size = 180 if c["is_pareto_optimal"] else 110
            edge_w = 2.0 if c["is_pareto_optimal"] else 0.8
            edge_c = "black" if c["is_pareto_optimal"] else "white"

            ax.scatter(
                x, y,
                color=style["color"],
                marker=style["marker"],
                s=size,
                edgecolors=edge_c,
                linewidths=edge_w,
                zorder=5 if c["is_pareto_optimal"] else 3,
                alpha=0.95,
            )

            # Annotate key points
            name = c["condition"]
            if c["is_pareto_optimal"] or "T=100" in name or "T=1" in name or "T=4" in name or "iter=2" in name or "H=10" in name or "SmallVLA" in name:
                short_name = name.replace("Policy", "").strip()
                ax.annotate(
                    short_name,
                    (x, y),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold" if c["is_pareto_optimal"] else "normal",
                    alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
                )

        # Plot Pareto frontier step line
        frontier_points = [c for c in conds if c["is_pareto_optimal"]]
        frontier_points.sort(key=lambda p: p["latency_p50_ms"])
        if len(frontier_points) >= 2:
            fx = [p["latency_p50_ms"] for p in frontier_points]
            fy = [p["success_rate_mean"] for p in frontier_points]
            ax.plot(fx, fy, "--", color="#333333", linewidth=2.0, alpha=0.75, zorder=4, label="Pareto Frontier")

        # Titles and axes
        title_str = "2D Point-Mass Reaching" if "reach" in bn.lower() else "PushT Benchmark"
        ax.set_title(f"{title_str}\nLatency–Performance Pareto Frontier (GB10)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Inference Latency (ms, log scale)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Success Rate (Mean over Seeds)", fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

        # Custom legend
        legend_elements = [
            Line2D([0], [0], marker=style["marker"], color="w", markerfacecolor=style["color"],
                   markersize=10, label=style["label"])
            for style in family_styles.values()
        ]
        legend_elements.append(Line2D([0], [0], color="#333333", linestyle="--", linewidth=2.0, label="Pareto Frontier"))
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.suptitle("EXP-004: Latency-Performance Pareto Sweep on NVIDIA DGX Spark (GB10)", fontsize=15, fontweight="heavy", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved Pareto frontier figure to: {output_path}")


def plot_latency_breakdown(
    latency_records: List[Dict[str, Any]],
    output_path: str,
):
    """Generate high-resolution latency breakdown bar chart."""
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

    # Distinct conditions
    unique_conds = []
    seen = set()
    for r in latency_records:
        cond_name = r["condition"]
        if cond_name not in seen:
            seen.add(cond_name)
            unique_conds.append(r)

    # Sort by p50 latency ascending
    unique_conds.sort(key=lambda x: x["latency_p50_ms"])

    labels = [c["condition"] for c in unique_conds]
    medians = [c["latency_p50_ms"] for c in unique_conds]
    p95s = [c["latency_p95_ms"] for c in unique_conds]
    p99s = [c["latency_p99_ms"] for c in unique_conds]
    families = [c["family"] for c in unique_conds]

    fam_colors = {
        "MPC": "#1f77b4",
        "DDPM": "#d62728",
        "Flow Matching": "#9467bd",
        "MIP": "#ff7f0e",
        "VLA": "#2ca02c",
        "Regression": "#7f7f7f",
    }
    colors = [fam_colors.get(f, "#333333") for f in families]

    y_pos = np.arange(len(labels))
    error_high = [p99s[i] - medians[i] for i in range(len(labels))]

    bars = ax.barh(y_pos, medians, xerr=[np.zeros(len(labels)), error_high],
                   color=colors, alpha=0.85, edgecolor="black", linewidth=0.8,
                   capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"})

    # Label text on bars
    for i, bar in enumerate(bars):
        val = medians[i]
        p99_val = p99s[i]
        text_str = f" {val:.2f} ms (p99: {p99_val:.2f} ms)"
        ax.text(val * 1.05, bar.get_y() + bar.get_height() / 2, text_str,
                va="center", ha="left", fontsize=8, fontweight="bold", alpha=0.9)

    # Reference frequency lines
    freq_lines = [
        (1.0, "1 kHz (1.0 ms)"),
        (2.0, "500 Hz (2.0 ms)"),
        (10.0, "100 Hz (10.0 ms)"),
        (20.0, "50 Hz (20.0 ms)"),
        (100.0, "10 Hz (100.0 ms)"),
    ]
    for lat_ms, freq_label in freq_lines:
        ax.axvline(lat_ms, color="red", linestyle=":", alpha=0.4, linewidth=1.2)
        ax.text(lat_ms, -0.8, freq_label, rotation=90, va="bottom", ha="right",
                fontsize=7.5, color="red", alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, fontweight="medium")
    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency (ms, log scale - 1,000 timed calls on GB10)", fontsize=11, fontweight="bold")
    ax.set_title("Inference Latency Profile Across Controller Configurations\n(Median p50 with p99 Error Whiskers on NVIDIA GB10)", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.35)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=fam_colors[f],
               markersize=10, label=f)
        for f in fam_colors
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved latency breakdown figure to: {output_path}")


# ===========================================================================
# Master Execution Pipeline
# ===========================================================================
def run_pareto_sweep(
    benchmarks: List[str],
    seeds: List[int],
    episodes: int = 50,
    n_warmup: int = 100,
    n_timed: int = 1000,
    horizon: int = 16,
    output_dir: str = "results/exp004",
    device_str: str = "cuda",
):
    """Run full EXP-004 Latency-Performance Pareto sweep."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"\n[EXP-004] Starting Pareto sweep on {device} (GB10 Grace Blackwell)")
    print(f"  Benchmarks:   {benchmarks}")
    print(f"  Seeds:        {seeds}")
    print(f"  Episodes:     {episodes} per condition per seed")
    print(f"  Profiling:    {n_warmup} warmup calls, {n_timed} timed calls")
    print(f"  Output Dir:   {output_dir}\n")

    # Save env provenance
    env_info = collect_env_info(STUDY_ROOT)
    env_json_path = os.path.join(output_dir, "env_info.json")
    with open(env_json_path, "w") as f:
        json.dump(env_info, f, indent=2)

    master_pareto_data: List[Dict[str, Any]] = []
    master_latency_table: List[Dict[str, Any]] = []
    pareto_by_bench: Dict[str, List[Dict[str, Any]]] = {}

    for bench_name in benchmarks:
        print(f"\n{'=' * 80}")
        print(f"BENCHMARK: {bench_name.upper()}")
        print(f"{'=' * 80}")

        # Store evaluations across seeds: {cond_name: [seed_metrics_list]}
        cond_evaluations: Dict[str, List[Dict[str, Any]]] = {}
        cond_latencies: Dict[str, Dict[str, float]] = {}

        for seed in seeds:
            print(f"\n--- Benchmark: {bench_name} | Seed: {seed} ---")
            bench = make_benchmark(bench_name, seed=seed)

            # 1. Collect Demonstrations
            print("  [Data] Collecting expert demonstrations...")
            demos_state, demos_vla = collect_demonstrations(bench, n_demos=30, horizon=horizon, seed=seed)
            print(f"  Collected {len(demos_state)} state demos and {len(demos_vla)} VLA demos")

            # 2. Train Models Once
            print("  [Train] Training learning-based controllers on GB10...")
            
            # DDPM
            t0 = time.time()
            ddpm = DiffusionPolicy(
                action_dim=bench.action_dim,
                horizon=horizon,
                obs_dim=bench.state_dim,
                num_diffusion_steps=100,
                hidden_dim=128,
                num_layers=3,
                device=device,
            )
            ddpm.train(demos_state, epochs=30, batch_size=16, lr=1e-3, verbose=False)
            ddpm.net.eval()
            print(f"    DDPM (T=100) trained in {time.time()-t0:.2f}s")

            # Flow Matching
            t0 = time.time()
            flow = FlowMatchingPolicy(
                action_dim=bench.action_dim,
                horizon=horizon,
                obs_dim=bench.state_dim,
                num_flow_steps=10,
                hidden_dim=128,
                num_layers=3,
                device=device,
            )
            flow.train(demos_state, epochs=30, batch_size=16, lr=1e-3, verbose=False)
            flow.net.eval()
            print(f"    Flow Matching (T=10) trained in {time.time()-t0:.2f}s")

            # MIP (Iterative Regression)
            t0 = time.time()
            mip = IterativeRegressionPolicy(
                action_dim=bench.action_dim,
                horizon=horizon,
                obs_dim=bench.state_dim,
                num_iterations=5,
                hidden_dim=128,
                noise_std=0.1,
                device=device,
            )
            mip.train(demos_state, num_epochs=25, batch_size=16, lr=1e-3, verbose=False)
            for net in mip.nets:
                net.eval()
            print(f"    MIP (5-iters) trained in {time.time()-t0:.2f}s")

            # Regression (RCP)
            t0 = time.time()
            rcp = RegressionPolicy(
                action_dim=bench.action_dim,
                horizon=horizon,
                obs_dim=bench.state_dim,
                hidden_dim=128,
                num_layers=3,
                device=device,
            )
            rcp.train(demos_state, num_epochs=30, batch_size=16, lr=1e-3, verbose=False)
            rcp.net.eval()
            print(f"    Regression (RCP) trained in {time.time()-t0:.2f}s")

            # SmallVLA
            t0 = time.time()
            vla = SmallVLA(
                action_dim=bench.action_dim,
                horizon=horizon,
                hidden_dim=256,
                device=device,
            )
            vla.train(demos_vla, epochs=12, batch_size=16, lr=1e-3, verbose=False)
            vla.eval_mode()
            print(f"    SmallVLA trained in {time.time()-t0:.2f}s")

            # 3. Configure Classical MPC Controllers
            stage_cost, term_cost, Q, R, Qf = bench.make_costs()
            u_bounds = (-5.0 * np.ones(bench.action_dim), 5.0 * np.ones(bench.action_dim))
            A, B = bench.dyn.linearize(np.zeros(bench.state_dim), np.zeros(bench.action_dim))

            linear_mpc_10 = LinearMPC(A, B, Q, R, Qf, horizon=10, u_bounds=u_bounds)
            linear_mpc_20 = LinearMPC(A, B, Q, R, Qf, horizon=20, u_bounds=u_bounds)
            ref_10 = np.tile(bench.goal, (11, 1))
            ref_20 = np.tile(bench.goal, (21, 1))

            nmpc = NonlinearMPC(bench.dyn.dynamics, stage_cost, term_cost, horizon=15, u_bounds=u_bounds)
            cfmpc = CollisionFreeMPC(
                bench.dyn.dynamics, stage_cost, term_cost, bench.world,
                horizon=15, u_bounds=u_bounds, collision_weight=100.0, ilqr_iters=15
            )

            # Sample inputs for latency profiling
            sample_state = bench.reset()
            sample_img = render_benchmark_image(bench, sample_state, size=96)
            sample_instr = "reach target"

            # 4. Define All Conditions
            conditions_dict: Dict[str, Dict[str, Any]] = {}

            # Classical MPC
            conditions_dict["Linear MPC (H=10)"] = {
                "family": "MPC", "category": "Linear MPC", "parameter": "H=10", "is_cuda": False,
                "profile_fn": lambda _m=linear_mpc_10, _s=sample_state, _r=ref_10: _m.solve(_s, _r),
                "policy_fn": lambda s, _m=linear_mpc_10, _r=ref_10: _m.solve(s, _r).control,
            }
            conditions_dict["Linear MPC (H=20)"] = {
                "family": "MPC", "category": "Linear MPC", "parameter": "H=20", "is_cuda": False,
                "profile_fn": lambda _m=linear_mpc_20, _s=sample_state, _r=ref_20: _m.solve(_s, _r),
                "policy_fn": lambda s, _m=linear_mpc_20, _r=ref_20: _m.solve(s, _r).control,
            }
            conditions_dict["Nonlinear MPC (iters=5)"] = {
                "family": "MPC", "category": "Nonlinear MPC", "parameter": "iters=5", "is_cuda": False,
                "profile_fn": lambda _m=nmpc, _s=sample_state: _m.solve(_s, max_iter=5),
                "policy_fn": lambda s, _m=nmpc: _m.solve(s, max_iter=5)["action"],
            }
            conditions_dict["Nonlinear MPC (iters=15)"] = {
                "family": "MPC", "category": "Nonlinear MPC", "parameter": "iters=15", "is_cuda": False,
                "profile_fn": lambda _m=nmpc, _s=sample_state: _m.solve(_s, max_iter=15),
                "policy_fn": lambda s, _m=nmpc: _m.solve(s, max_iter=15)["action"],
            }
            conditions_dict["Nonlinear MPC (iters=30)"] = {
                "family": "MPC", "category": "Nonlinear MPC", "parameter": "iters=30", "is_cuda": False,
                "profile_fn": lambda _m=nmpc, _s=sample_state: _m.solve(_s, max_iter=30),
                "policy_fn": lambda s, _m=nmpc: _m.solve(s, max_iter=30)["action"],
            }
            conditions_dict["CollisionFree MPC"] = {
                "family": "MPC", "category": "CollisionFree MPC", "parameter": "ilqr=15", "is_cuda": False,
                "profile_fn": lambda _m=cfmpc, _s=sample_state: _m.solve(_s),
                "policy_fn": lambda s, _m=cfmpc: _m.solve(s)[0],
            }

            # DDPM Step Sweep: T in [1, 2, 4, 8, 16, 32, 64, 100]
            for T in [1, 2, 4, 8, 16, 32, 64, 100]:
                c_name = f"DDPM (T={T})"
                conditions_dict[c_name] = {
                    "family": "DDPM", "category": "DDPM", "parameter": f"T={T}", "is_cuda": True,
                    "profile_fn": lambda _p=ddpm, _s=sample_state, _T=T: sample_ddpm_step_sweep(_p, _s, T=_T),
                    "policy_fn": lambda s, _p=ddpm, _T=T: sample_ddpm_step_sweep(_p, s, T=_T)[0],
                }

            # Flow Matching Step Sweep: T in [1, 2, 4, 8, 10]
            for T in [1, 2, 4, 8, 10]:
                c_name = f"Flow Matching (T={T})"
                conditions_dict[c_name] = {
                    "family": "Flow Matching", "category": "Flow Matching", "parameter": f"T={T}", "is_cuda": True,
                    "profile_fn": lambda _p=flow, _s=sample_state, _T=T: sample_flow_step_sweep(_p, _s, T=_T),
                    "policy_fn": lambda s, _p=flow, _T=T: sample_flow_step_sweep(_p, s, T=_T)[0],
                }

            # MIP Iteration Sweep: iters in [1, 2, 3, 4, 5]
            for it in [1, 2, 3, 4, 5]:
                c_name = f"MIP (iter={it})"
                conditions_dict[c_name] = {
                    "family": "MIP", "category": "MIP", "parameter": f"iter={it}", "is_cuda": True,
                    "profile_fn": lambda _p=mip, _s=sample_state, _it=it: sample_mip_iteration_sweep(_p, _s, iters=_it),
                    "policy_fn": lambda s, _p=mip, _it=it: sample_mip_iteration_sweep(_p, s, iters=_it)[0],
                }

            # Regression Baseline
            conditions_dict["Regression (RCP)"] = {
                "family": "Regression", "category": "Regression", "parameter": "T=1 (no-noise)", "is_cuda": True,
                "profile_fn": lambda _p=rcp, _s=sample_state: _p.predict(_s),
                "policy_fn": lambda s, _p=rcp: _p.predict(s)[0].cpu().detach().numpy(),
            }

            # SmallVLA
            conditions_dict["SmallVLA"] = {
                "family": "VLA", "category": "SmallVLA", "parameter": "ViT-Base", "is_cuda": True,
                "profile_fn": lambda _v=vla, _img=sample_img, _ins=sample_instr: _v.predict_action(_img, _ins),
                "policy_fn": lambda s, _v=vla, _b=bench, _ins=sample_instr: _v.predict_action(render_benchmark_image(_b, s), _ins)[0],
            }

            # 5. Measure Latency on First Seed
            if seed == seeds[0]:
                print(f"\n  [Latency] Profiling high-precision inference latency ({n_warmup} warmup, {n_timed} timed calls)...")
                for c_name, c_spec in conditions_dict.items():
                    lat_res = profile_latency(
                        c_spec["profile_fn"],
                        is_cuda=c_spec["is_cuda"],
                        n_warmup=n_warmup,
                        n_eval=n_timed,
                    )
                    cond_latencies[c_name] = lat_res
                    print(f"    {c_name:<26} -> p50: {lat_res['p50_ms']:>6.3f} ms | mean: {lat_res['mean_ms']:>6.3f} ms | p99: {lat_res['p99_ms']:>6.3f} ms | {lat_res['throughput_hz']:>7.1f} Hz")

            # 6. Evaluate Closed-Loop Episodes
            print(f"\n  [Eval] Running {episodes} closed-loop evaluation episodes per controller...")
            for c_name, c_spec in conditions_dict.items():
                policy_fn = c_spec["policy_fn"]
                ep_results = []
                for ep in range(episodes):
                    metrics = simulate_rollout(bench, policy_fn)
                    ep_results.append(metrics)

                sr = float(np.mean([m["success"] for m in ep_results]))
                cr = float(np.mean([m["collision"] for m in ep_results]))
                pl = float(np.mean([m["path_length"] for m in ep_results]))
                if c_name not in cond_evaluations:
                    cond_evaluations[c_name] = []
                cond_evaluations[c_name].append({
                    "seed": seed,
                    "success_rate": sr,
                    "collision_rate": cr,
                    "path_length": pl,
                    "n_episodes": episodes,
                })
                print(f"    {c_name:<26} -> Success: {sr*100:>5.1f}% | Collisions: {cr*100:>4.1f}%")

        # 7. Aggregate Across Seeds for This Benchmark
        bench_records = []
        for c_name, seed_list in cond_evaluations.items():
            srs = [s["success_rate"] for s in seed_list]
            crs = [s["collision_rate"] for s in seed_list]
            pls = [s["path_length"] for s in seed_list]
            lat = cond_latencies[c_name]
            c_spec = conditions_dict[c_name]

            record = {
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

        # Compute Pareto Dominance
        bench_records = compute_pareto_dominance(bench_records)
        pareto_by_bench[bench_name] = bench_records
        master_pareto_data.extend(bench_records)

        for r in bench_records:
            master_latency_table.append({
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
            })

    # =======================================================================
    # Save CSV Datasets
    # =======================================================================
    # 1. Master Pareto Data CSV
    pareto_csv_path = os.path.join(output_dir, "pareto_data.csv")
    fieldnames_pareto = list(master_pareto_data[0].keys())
    with open(pareto_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_pareto)
        writer.writeheader()
        writer.writerows(master_pareto_data)
    print(f"\n[Data] Saved Pareto dataset to: {pareto_csv_path}")

    # 2. Latency Table CSV
    latency_csv_path = os.path.join(output_dir, "latency_table.csv")
    fieldnames_latency = list(master_latency_table[0].keys())
    with open(latency_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_latency)
        writer.writeheader()
        writer.writerows(master_latency_table)
    print(f"[Data] Saved Latency table to: {latency_csv_path}")

    # 3. Full Metrics JSON
    metrics_json_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_json_path, "w") as f:
        json.dump({
            "env_info": env_info,
            "pareto_results": master_pareto_data,
            "latency_table": master_latency_table,
        }, f, indent=2)
    print(f"[Data] Saved Metrics summary to: {metrics_json_path}")

    # Copy env_info.json to results/env_info.json as requested
    root_env_path = os.path.join(STUDY_ROOT, "results", "env_info.json")
    os.makedirs(os.path.dirname(root_env_path), exist_ok=True)
    shutil.copyfile(env_json_path, root_env_path)
    print(f"[Provenance] Synced hardware provenance to: {root_env_path}")

    # =======================================================================
    # Generate Plots
    # =======================================================================
    pareto_png_path = os.path.join(output_dir, "pareto_frontier.png")
    plot_pareto_frontier(pareto_by_bench, pareto_png_path)

    breakdown_png_path = os.path.join(output_dir, "latency_breakdown.png")
    plot_latency_breakdown(master_pareto_data, breakdown_png_path)

    # =======================================================================
    # Console Summary & Analysis
    # =======================================================================
    print("\n" + "=" * 105)
    print("EXP-004 SUMMARY: LATENCY-PERFORMANCE PARETO SWEEP")
    print("=" * 105)
    for bn in benchmarks:
        print(f"\nBenchmark: {bn.upper()}")
        header = f"  {'Controller':<26} {'Family':<15} {'p50 Lat(ms)':>11} {'p99 Lat(ms)':>11} {'Success':>9} {'±std':>6} {'DomCount':>9} {'Pareto?':>8}"
        print(header)
        print("  " + "-" * 100)
        conds = pareto_by_bench[bn]
        for c in sorted(conds, key=lambda x: x["latency_p50_ms"]):
            pareto_str = "★ YES" if c["is_pareto_optimal"] else "  no"
            print(f"  {c['condition']:<26} {c['family']:<15} {c['latency_p50_ms']:>11.3f} {c['latency_p99_ms']:>11.3f} {c['success_rate_mean']*100:>8.1f}% {c['success_rate_std']*100:>5.1f}% {c['dominance_count']:>9d} {pareto_str:>8}")
    print("\n" + "=" * 105)


def main():
    parser = argparse.ArgumentParser(description="EXP-004: Latency-Performance Pareto Sweep Runner.")
    parser.add_argument("--benchmark", type=str, default="all", choices=["all", "reaching", "pusht"], help="Benchmark name or 'all'")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds for evaluation")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes per seed")
    parser.add_argument("--n-warmup", type=int, default=100, help="Number of warmup calls for latency profiler")
    parser.add_argument("--n-timed", type=int, default=1000, help="Number of timed calls for latency profiler")
    parser.add_argument("--horizon", type=int, default=16, help="Action prediction horizon")
    parser.add_argument("--output-dir", type=str, default="results/exp004", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device")

    args = parser.parse_args()

    benchmarks = ["reaching", "pusht"] if args.benchmark == "all" else [args.benchmark]
    run_pareto_sweep(
        benchmarks=benchmarks,
        seeds=args.seeds,
        episodes=args.episodes,
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
        horizon=args.horizon,
        output_dir=args.output_dir,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
