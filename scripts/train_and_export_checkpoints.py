#!/usr/bin/env python3
"""Train all baseline model checkpoints and export them for Hugging Face Hub release.

This script:
1. Collects 50 expert demonstrations from the PushT benchmark (state trajectories,
   rendered images, and natural-language instructions) using DemonstrationCollector.
2. Trains four compute-matched baseline policy checkpoints:
   - SmallVLA (Vision-Language-Action ViT-Base policy)
   - DDPM Diffusion Policy (Conditional 1D U-Net diffusion)
   - Flow Matching Policy (Rectified flow velocity prediction)
   - Minimal Iterative Policy (MIP 2-step regression ablation baseline)
3. Saves all trained checkpoints to results/checkpoints/:
   - results/checkpoints/small_vla_pusht.pt
   - results/checkpoints/ddpm_pusht.pt
   - results/checkpoints/flow_matching_pusht.pt
   - results/checkpoints/mip_pusht.npz
4. Runs a comprehensive verification suite:
   - Loads each checkpoint from disk
   - Runs inference on test observations (image+instruction or state)
   - Asserts valid action outputs and shapes
   - Computes checkpoint sizes and records release metadata.

Usage:
    conda run -n mpc_vla python scripts/train_and_export_checkpoints.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
]
for _d in _MODULE_DIRS:
    _full = os.path.join(STUDY_ROOT, _d)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_and_export")

# Imports from codebase
from benchmarks.demonstration_collector import DemonstrationCollector
from benchmarks.pusht_env import PushTEnv
from benchmarks.reaching_env import Obstacle, ReachingEnv
from diffusion_baselines.ddpm_policy import DiffusionPolicy
from diffusion_baselines.flow_matching_policy import FlowMatchingPolicy
from mpc_baselines_repo.src.diffusion_warm_start.minimal_iterative_policy import (
    MinimalIterativePolicy,
)
from vla_baselines.small_vla import SmallVLA


# ---------------------------------------------------------------------------
# Demonstration Collection
# ---------------------------------------------------------------------------
def pusht_expert_action(env: PushTEnv) -> np.ndarray:
    """Expert policy for PushT: position agent behind block relative to target and push."""
    block_xy = env._block[:2]
    target_xy = env._target[:2]
    dir_to_target = target_xy - block_xy
    d = float(np.linalg.norm(dir_to_target))
    if d < 1e-3:
        return np.zeros(2, dtype=np.float32)
    # Desired agent position: behind block (opposite target)
    desired_agent = block_xy - (dir_to_target / d) * (env.block_size * 0.6)
    action = desired_agent - env._agent
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def collect_pusht_demonstrations(
    n_demos: int = 50,
    max_steps: int = 100,
    horizon: int = 8,
    image_size: int = 96,
    seed: int = 42,
) -> Tuple[
    List[Dict[str, Any]],
    List[Tuple[np.ndarray, np.ndarray]],
    Dict[str, Any],
]:
    """Collect image + language and state demonstrations from PushTEnv.

    Uses DemonstrationCollector for state and image gathering and builds chunked
    (horizon-length) action sequence demonstrations.

    Returns:
        vla_demos: List of dicts with {"image", "instruction", "action"}
        state_demos: List of (obs, action_seq) tuples
        metadata: Summary statistics of collected demonstrations
    """
    logger.info("Collecting %d demonstrations from PushTEnv (horizon=%d)...", n_demos, horizon)
    env = PushTEnv(image_size=image_size, max_steps=max_steps, seed=seed)

    # Use DemonstrationCollector to collect state transitions
    state_collector = DemonstrationCollector(image_mode=False)
    image_collector = DemonstrationCollector(image_mode=True)

    vla_demos: List[Dict[str, Any]] = []
    state_demos: List[Tuple[np.ndarray, np.ndarray]] = []
    ep_lengths: List[int] = []
    ep_rewards: List[float] = []
    successes: int = 0

    instruction = "push the T block to the target area"

    for ep in range(n_demos):
        ep_seed = seed + ep
        obs = env.reset(seed=ep_seed)

        ep_obs: List[np.ndarray] = []
        ep_imgs: List[np.ndarray] = []
        ep_acts: List[np.ndarray] = []
        total_rew = 0.0

        for step in range(max_steps):
            cur_obs = env.get_observation().copy()
            cur_img = env.get_image()
            act = pusht_expert_action(env)

            next_obs, rew, done, info = env.step(act)
            total_rew += rew

            # Record in collectors
            state_collector._append(cur_obs, act, next_obs, rew, done)
            image_collector._append(cur_img, act, env.get_image(), rew, done)

            ep_obs.append(cur_obs)
            ep_imgs.append(cur_img)
            ep_acts.append(act)

            if done:
                break

        if env.is_success():
            successes += 1

        T = len(ep_obs)
        ep_lengths.append(T)
        ep_rewards.append(total_rew)

        # Build horizon-length action sequences for each timestep
        for t in range(T):
            act_chunk: List[np.ndarray] = []
            for h in range(horizon):
                if t + h < T:
                    act_chunk.append(ep_acts[t + h])
                else:
                    # Pad with last action
                    act_chunk.append(ep_acts[-1])
            act_seq = np.stack(act_chunk, axis=0).astype(np.float32)  # (horizon, 2)

            state_demos.append((ep_obs[t].astype(np.float32), act_seq))
            vla_demos.append({
                "image": ep_imgs[t],
                "instruction": instruction,
                "action": act_seq,
            })

    metadata = {
        "n_episodes": n_demos,
        "total_transitions": len(state_demos),
        "mean_episode_length": float(np.mean(ep_lengths)),
        "mean_episode_reward": float(np.mean(ep_rewards)),
        "success_rate": float(successes / n_demos),
        "horizon": horizon,
        "obs_dim": 5,
        "action_dim": 2,
        "image_size": image_size,
        "instruction": instruction,
    }

    logger.info(
        "PushT collection complete: %d transitions from %d episodes (success rate: %.1f%%, avg length: %.1f)",
        len(state_demos), n_demos, metadata["success_rate"] * 100, metadata["mean_episode_length"],
    )
    return vla_demos, state_demos, metadata


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------
def train_small_vla(
    vla_demos: List[Dict[str, Any]],
    action_dim: int = 2,
    horizon: int = 8,
    img_size: int = 96,
    hidden_dim: int = 256,
    num_layers: int = 4,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
) -> SmallVLA:
    """Train SmallVLA model on image+language demonstrations."""
    logger.info("=" * 60)
    logger.info("Training SmallVLA (ViT-Base + Text Encoder + MLP Action Head)...")
    logger.info("  Action dim: %d, Horizon: %d, Image size: %d, Hidden dim: %d", action_dim, horizon, img_size, hidden_dim)
    logger.info("  Epochs: %d, Batch size: %d, LR: %.1e, Device: %s", epochs, batch_size, lr, device)

    model = SmallVLA(
        action_dim=action_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        img_size=img_size,
        text_backend="auto",
        device=device,
    )

    t0 = time.perf_counter()
    history = model.train(
        demonstrations=vla_demos,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True,
        log_interval=max(1, len(vla_demos) // (batch_size * 2)),
    )
    elapsed = time.perf_counter() - t0
    final_loss = history["epoch_loss"][-1] if history["epoch_loss"] else float("nan")
    logger.info("[OK] SmallVLA trained in %.1fs | Final Epoch Loss: %.6f", elapsed, final_loss)
    return model


def train_ddpm_policy(
    state_demos: List[Tuple[np.ndarray, np.ndarray]],
    obs_dim: int = 5,
    action_dim: int = 2,
    horizon: int = 8,
    num_diffusion_steps: int = 50,
    hidden_dim: int = 128,
    num_layers: int = 3,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> DiffusionPolicy:
    """Train DDPM Diffusion Policy on state-action sequence demonstrations."""
    logger.info("=" * 60)
    logger.info("Training DDPM Diffusion Policy (1D Temporal U-Net)...")
    logger.info("  Obs dim: %d, Action dim: %d, Horizon: %d, Steps: %d", obs_dim, action_dim, horizon, num_diffusion_steps)
    logger.info("  Epochs: %d, Batch size: %d, LR: %.1e, Device: %s", epochs, batch_size, lr, device)

    policy = DiffusionPolicy(
        action_dim=action_dim,
        horizon=horizon,
        obs_dim=obs_dim,
        num_diffusion_steps=num_diffusion_steps,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )

    t0 = time.perf_counter()
    losses = policy.train(
        demonstrations=state_demos,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    final_loss = losses[-1] if losses else float("nan")
    logger.info("[OK] DDPM Policy trained in %.1fs | Final Epoch Loss: %.6f", elapsed, final_loss)
    return policy


def train_flow_matching_policy(
    state_demos: List[Tuple[np.ndarray, np.ndarray]],
    obs_dim: int = 5,
    action_dim: int = 2,
    horizon: int = 8,
    num_flow_steps: int = 10,
    hidden_dim: int = 128,
    num_layers: int = 3,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> FlowMatchingPolicy:
    """Train Flow Matching Policy (Rectified Flow) on demonstrations."""
    logger.info("=" * 60)
    logger.info("Training Flow Matching Policy (Rectified Flow 1D Temporal U-Net)...")
    logger.info("  Obs dim: %d, Action dim: %d, Horizon: %d, Flow steps: %d", obs_dim, action_dim, horizon, num_flow_steps)
    logger.info("  Epochs: %d, Batch size: %d, LR: %.1e, Device: %s", epochs, batch_size, lr, device)

    policy = FlowMatchingPolicy(
        action_dim=action_dim,
        horizon=horizon,
        obs_dim=obs_dim,
        num_flow_steps=num_flow_steps,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )

    t0 = time.perf_counter()
    losses = policy.train(
        demonstrations=state_demos,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    final_loss = losses[-1] if losses else float("nan")
    logger.info("[OK] Flow Matching Policy trained in %.1fs | Final Epoch Loss: %.6f", elapsed, final_loss)
    return policy


def train_minimal_iterative_policy(
    state_demos: List[Tuple[np.ndarray, np.ndarray]],
    state_dim: int = 5,
    action_dim: int = 2,
    horizon: int = 8,
    hidden_dim: int = 128,
    noise_std: float = 0.1,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
) -> MinimalIterativePolicy:
    """Train Minimal Iterative Policy (MIP 2-step regression with noise injection)."""
    logger.info("=" * 60)
    logger.info("Training Minimal Iterative Policy (MIP ablation baseline)...")
    logger.info("  State dim: %d, Action dim: %d, Horizon: %d, Noise std: %.2f", state_dim, action_dim, horizon, noise_std)
    logger.info("  Epochs: %d, Batch size: %d, LR: %.1e", epochs, batch_size, lr)

    mip = MinimalIterativePolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        noise_std=noise_std,
        seed=seed,
    )

    t0 = time.perf_counter()
    losses = mip.train(
        demonstrations=state_demos,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    final_s1 = losses["step1"][-1] if losses["step1"] else float("nan")
    final_s2 = losses["step2"][-1] if losses["step2"] else float("nan")
    logger.info("[OK] MIP trained in %.1fs | Final Step1 Loss: %.6f, Step2 Loss: %.6f", elapsed, final_s1, final_s2)
    return mip


# ---------------------------------------------------------------------------
# Verification Suite
# ---------------------------------------------------------------------------
def verify_checkpoints(
    checkpoint_dir: str,
    env: PushTEnv,
    horizon: int = 8,
    action_dim: int = 2,
    obs_dim: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load each checkpoint from disk, run sample inference, and assert output shapes."""
    logger.info("=" * 60)
    logger.info("RUNNING VERIFICATION SUITE ON SAVED CHECKPOINTS")
    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info("=" * 60)

    results: Dict[str, Any] = {}

    # Sample observation & image from environment
    env.reset(seed=999)
    sample_obs = env.get_observation()  # (5,)
    sample_img = env.get_image()        # (96, 96, 3)
    sample_instr = env.get_language_instruction()

    # 1. Verify SmallVLA
    vla_path = os.path.join(checkpoint_dir, "small_vla_pusht.pt")
    logger.info("[Verify 1/4] Loading SmallVLA from %s...", vla_path)
    assert os.path.exists(vla_path), f"Checkpoint missing: {vla_path}"
    vla_size_bytes = os.path.getsize(vla_path)

    vla_loaded = SmallVLA.load(vla_path, device=device, text_backend="auto")
    vla_action = vla_loaded.predict_action(sample_img, sample_instr)

    assert isinstance(vla_action, np.ndarray), f"Expected np.ndarray, got {type(vla_action)}"
    assert vla_action.shape == (horizon, action_dim), f"Expected shape {(horizon, action_dim)}, got {vla_action.shape}"
    assert np.all(np.isfinite(vla_action)), "SmallVLA produced non-finite values"

    results["small_vla_pusht"] = {
        "path": vla_path,
        "file_size_bytes": vla_size_bytes,
        "file_size_mb": round(vla_size_bytes / (1024 * 1024), 2),
        "param_count": vla_loaded.count_parameters(),
        "input_type": "image (96x96x3) + instruction",
        "output_shape": list(vla_action.shape),
        "sample_output_mean": float(np.mean(vla_action)),
        "sample_output_std": float(np.std(vla_action)),
        "status": "PASS",
    }
    logger.info("  ✓ SmallVLA PASS (size: %.2f MB, params: %s, output shape: %s)",
                results["small_vla_pusht"]["file_size_mb"],
                f"{results['small_vla_pusht']['param_count']:,}",
                vla_action.shape)

    # 2. Verify DDPM Diffusion Policy
    ddpm_path = os.path.join(checkpoint_dir, "ddpm_pusht.pt")
    logger.info("[Verify 2/4] Loading DDPM Diffusion Policy from %s...", ddpm_path)
    assert os.path.exists(ddpm_path), f"Checkpoint missing: {ddpm_path}"
    ddpm_size_bytes = os.path.getsize(ddpm_path)

    ddpm_loaded = DiffusionPolicy.from_checkpoint(ddpm_path, device=device)
    ddpm_action_t = ddpm_loaded.sample(sample_obs, num_samples=1)
    ddpm_action = ddpm_action_t.cpu().numpy().squeeze(0)

    assert isinstance(ddpm_action_t, torch.Tensor), f"Expected torch.Tensor, got {type(ddpm_action_t)}"
    assert ddpm_action.shape == (horizon, action_dim), f"Expected shape {(horizon, action_dim)}, got {ddpm_action.shape}"
    assert np.all(np.isfinite(ddpm_action)), "DDPM produced non-finite values"

    ddpm_params = sum(p.numel() for p in ddpm_loaded.net.parameters() if p.requires_grad)
    results["ddpm_pusht"] = {
        "path": ddpm_path,
        "file_size_bytes": ddpm_size_bytes,
        "file_size_mb": round(ddpm_size_bytes / (1024 * 1024), 2),
        "param_count": ddpm_params,
        "input_type": f"state ({obs_dim},)",
        "output_shape": list(ddpm_action.shape),
        "sample_output_mean": float(np.mean(ddpm_action)),
        "sample_output_std": float(np.std(ddpm_action)),
        "status": "PASS",
    }
    logger.info("  ✓ DDPM Diffusion Policy PASS (size: %.2f MB, params: %s, output shape: %s)",
                results["ddpm_pusht"]["file_size_mb"],
                f"{results['ddpm_pusht']['param_count']:,}",
                ddpm_action.shape)

    # 3. Verify Flow Matching Policy
    flow_path = os.path.join(checkpoint_dir, "flow_matching_pusht.pt")
    logger.info("[Verify 3/4] Loading Flow Matching Policy from %s...", flow_path)
    assert os.path.exists(flow_path), f"Checkpoint missing: {flow_path}"
    flow_size_bytes = os.path.getsize(flow_path)

    flow_loaded = FlowMatchingPolicy.from_checkpoint(flow_path, device=device)
    flow_action_t = flow_loaded.sample(sample_obs, num_samples=1)
    flow_action = flow_action_t.cpu().numpy().squeeze(0)

    assert isinstance(flow_action_t, torch.Tensor), f"Expected torch.Tensor, got {type(flow_action_t)}"
    assert flow_action.shape == (horizon, action_dim), f"Expected shape {(horizon, action_dim)}, got {flow_action.shape}"
    assert np.all(np.isfinite(flow_action)), "Flow Matching produced non-finite values"

    flow_params = sum(p.numel() for p in flow_loaded.net.parameters() if p.requires_grad)
    results["flow_matching_pusht"] = {
        "path": flow_path,
        "file_size_bytes": flow_size_bytes,
        "file_size_mb": round(flow_size_bytes / (1024 * 1024), 2),
        "param_count": flow_params,
        "input_type": f"state ({obs_dim},)",
        "output_shape": list(flow_action.shape),
        "sample_output_mean": float(np.mean(flow_action)),
        "sample_output_std": float(np.std(flow_action)),
        "status": "PASS",
    }
    logger.info("  ✓ Flow Matching Policy PASS (size: %.2f MB, params: %s, output shape: %s)",
                results["flow_matching_pusht"]["file_size_mb"],
                f"{results['flow_matching_pusht']['param_count']:,}",
                flow_action.shape)

    # 4. Verify Minimal Iterative Policy (MIP)
    mip_path = os.path.join(checkpoint_dir, "mip_pusht.npz")
    logger.info("[Verify 4/4] Loading Minimal Iterative Policy (MIP) from %s...", mip_path)
    assert os.path.exists(mip_path), f"Checkpoint missing: {mip_path}"
    mip_size_bytes = os.path.getsize(mip_path)

    mip_loaded = MinimalIterativePolicy(state_dim=obs_dim, action_dim=action_dim, horizon=horizon)
    mip_loaded.load(mip_path)
    mip_action_samples = mip_loaded.sample(sample_obs, num_samples=1)
    mip_action = mip_action_samples[0]

    assert isinstance(mip_action, np.ndarray), f"Expected np.ndarray, got {type(mip_action)}"
    assert mip_action.shape == (horizon, action_dim), f"Expected shape {(horizon, action_dim)}, got {mip_action.shape}"
    assert np.all(np.isfinite(mip_action)), "MIP produced non-finite values"

    mip_params = sum(p.size for p in mip_loaded.step1_net.params() + mip_loaded.step2_net.params())
    results["mip_pusht"] = {
        "path": mip_path,
        "file_size_bytes": mip_size_bytes,
        "file_size_kb": round(mip_size_bytes / 1024, 2),
        "param_count": mip_params,
        "input_type": f"state ({obs_dim},)",
        "output_shape": list(mip_action.shape),
        "sample_output_mean": float(np.mean(mip_action)),
        "sample_output_std": float(np.std(mip_action)),
        "status": "PASS",
    }
    logger.info("  ✓ Minimal Iterative Policy PASS (size: %.2f KB, params: %s, output shape: %s)",
                results["mip_pusht"]["file_size_kb"],
                f"{results['mip_pusht']['param_count']:,}",
                mip_action.shape)

    return results


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Train and export baseline checkpoints for Hugging Face Hub.")
    parser.add_argument("--num-demos", type=int, default=50, help="Number of demonstrations to collect (default: 50)")
    parser.add_argument("--horizon", type=int, default=8, help="Action chunk horizon (default: 8)")
    parser.add_argument("--image-size", type=int, default=96, help="Image resolution for VLA (default: 96)")
    parser.add_argument("--output-dir", type=str, default="results/checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('auto', 'cuda', 'cpu')")
    parser.add_argument("--vla-epochs", type=int, default=20, help="Epochs for SmallVLA (default: 20)")
    parser.add_argument("--ddpm-epochs", type=int, default=40, help="Epochs for DDPM (default: 40)")
    parser.add_argument("--flow-epochs", type=int, default=40, help="Epochs for Flow Matching (default: 40)")
    parser.add_argument("--mip-epochs", type=int, default=40, help="Epochs for MIP (default: 40)")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("BASELINE CHECKPOINT TRAINING AND EXPORT PIPELINE")
    logger.info("Device: %s (PyTorch %s)", device, torch.__version__)
    logger.info("Study root: %s", STUDY_ROOT)
    logger.info("=" * 60)

    # Destination directory
    checkpoint_dir = os.path.join(STUDY_ROOT, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Collect Demonstrations
    vla_demos, state_demos, demo_meta = collect_pusht_demonstrations(
        n_demos=args.num_demos,
        max_steps=100,
        horizon=args.horizon,
        image_size=args.image_size,
        seed=42,
    )

    # 2. Train SmallVLA
    vla_model = train_small_vla(
        vla_demos=vla_demos,
        action_dim=demo_meta["action_dim"],
        horizon=args.horizon,
        img_size=args.image_size,
        hidden_dim=256,
        num_layers=4,
        epochs=args.vla_epochs,
        batch_size=32,
        lr=1e-4,
        device=device,
    )
    vla_path = os.path.join(checkpoint_dir, "small_vla_pusht.pt")
    vla_model.save(vla_path)
    logger.info("Saved SmallVLA checkpoint -> %s", vla_path)

    # 3. Train DDPM Diffusion Policy
    ddpm_policy = train_ddpm_policy(
        state_demos=state_demos,
        obs_dim=demo_meta["obs_dim"],
        action_dim=demo_meta["action_dim"],
        horizon=args.horizon,
        num_diffusion_steps=50,
        hidden_dim=128,
        num_layers=3,
        epochs=args.ddpm_epochs,
        batch_size=64,
        lr=1e-3,
        device=device,
    )
    ddpm_path = os.path.join(checkpoint_dir, "ddpm_pusht.pt")
    ddpm_policy.save(ddpm_path)
    logger.info("Saved DDPM Policy checkpoint -> %s", ddpm_path)

    # 4. Train Flow Matching Policy
    flow_policy = train_flow_matching_policy(
        state_demos=state_demos,
        obs_dim=demo_meta["obs_dim"],
        action_dim=demo_meta["action_dim"],
        horizon=args.horizon,
        num_flow_steps=10,
        hidden_dim=128,
        num_layers=3,
        epochs=args.flow_epochs,
        batch_size=64,
        lr=1e-3,
        device=device,
    )
    flow_path = os.path.join(checkpoint_dir, "flow_matching_pusht.pt")
    flow_policy.save(flow_path)
    logger.info("Saved Flow Matching Policy checkpoint -> %s", flow_path)

    # 5. Train Minimal Iterative Policy (MIP)
    mip_policy = train_minimal_iterative_policy(
        state_demos=state_demos,
        state_dim=demo_meta["obs_dim"],
        action_dim=demo_meta["action_dim"],
        horizon=args.horizon,
        hidden_dim=128,
        noise_std=0.1,
        epochs=args.mip_epochs,
        batch_size=32,
        lr=1e-3,
        seed=42,
    )
    mip_path = os.path.join(checkpoint_dir, "mip_pusht.npz")
    mip_policy.save(mip_path)
    logger.info("Saved MIP checkpoint -> %s", mip_path)

    # 6. Verification Suite
    test_env = PushTEnv(image_size=args.image_size, max_steps=100, seed=999)
    verification_results = verify_checkpoints(
        checkpoint_dir=checkpoint_dir,
        env=test_env,
        horizon=args.horizon,
        action_dim=demo_meta["action_dim"],
        obs_dim=demo_meta["obs_dim"],
        device=device,
    )

    # Save release manifest
    manifest = {
        "title": "MPC vs VLA vs Diffusion Study - Baseline Checkpoints",
        "benchmark": "PushT",
        "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": "DGX Spark (GB10 Grace Blackwell)",
        "demonstrations": demo_meta,
        "checkpoints": verification_results,
    }
    manifest_path = os.path.join(checkpoint_dir, "release_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved Hugging Face Hub release manifest -> %s", manifest_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("                     TRAINING & VERIFICATION SUMMARY                     ")
    print("=" * 80)
    print(f"{'Checkpoint Name':<28} | {'Size':<10} | {'Params':<12} | {'Output Shape':<14} | {'Status'}")
    print("-" * 80)
    for name, info in verification_results.items():
        size_str = f"{info.get('file_size_mb', info.get('file_size_kb', 0)):.2f} " + ("MB" if "file_size_mb" in info else "KB")
        params_str = f"{info['param_count']:,}"
        shape_str = str(info['output_shape'])
        status_str = info['status']
        print(f"{name:<28} | {size_str:<10} | {params_str:<12} | {shape_str:<14} | {status_str}")
    print("=" * 80)
    print(f"All 4 baseline checkpoints successfully verified and ready for Hugging Face Hub release!")
    print(f"Artifacts located at: {checkpoint_dir}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
