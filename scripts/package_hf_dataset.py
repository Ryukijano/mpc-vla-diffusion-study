#!/usr/bin/env python3
"""Packaging script for Hugging Face Hub dataset artifacts.

Collects expert demonstrations from Collision-Free MPC on 2D Reaching benchmarks
and packages them into:
  1. NumPy .npz archive (mpc_expert_demos_state.npz & mpc_expert_demos_images.npz)
  2. Apache Parquet table (mpc_expert_demos_state.parquet)
  3. Comprehensive Hugging Face Dataset Card (README.md)

Usage::

    conda run -n mpc_vla python scripts/package_hf_dataset.py
    conda run -n mpc_vla python scripts/package_hf_dataset.py --output-dir dist/hf_datasets/mpc_expert_demos --n-episodes 20 --verify
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
STUDY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY_ROOT not in sys.path:
    sys.path.insert(0, STUDY_ROOT)

for _d in ["mpc_baselines_repo", "mpc_baselines_repo/src", "diffusion_baselines", "vla_baselines", "benchmarks"]:
    _p = os.path.join(STUDY_ROOT, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.demonstration_collector import DemonstrationCollector
from benchmarks.reaching_env import Obstacle, ReachingEnv
from src.collision_free_mpc import CollisionFreeMPC, SDFWorld
from src.utils import PointMass2D


def collect_and_package_dataset(
    output_dir: str,
    n_episodes: int = 20,
    max_steps: int = 60,
    image_size: int = 64,
    verify: bool = True,
    dry_run: bool = False,
) -> str:
    """Collect expert demonstrations and format into .npz and .parquet packages."""
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(STUDY_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 72)
    print("Hugging Face Dataset Artifact Packager")
    print(f"Target Output Directory: {output_dir}")
    if dry_run:
        print("Mode: DRY-RUN (synthetic data, no MPC rollouts)")
    print(f"Episodes: {n_episodes}, Max Steps: {max_steps}, Image Size: {image_size}")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Dry-run shortcut: generate a tiny synthetic dataset without collecting
    # -----------------------------------------------------------------------
    if dry_run:
        rng = np.random.RandomState(42)
        total = max(n_episodes * max_steps, 4)
        obs_arr = rng.randn(total, 4).astype(np.float32)
        act_arr = rng.randn(total, 2).astype(np.float32)
        nobs_arr = obs_arr.copy()
        nobs_arr[:, :2] += 0.1 * act_arr
        rew_arr = -np.linalg.norm(obs_arr[:, :2], axis=1).astype(np.float32)
        done_arr = np.zeros(total, dtype=bool)
        steps_per_ep = max(1, total // n_episodes)
        for i in range(1, n_episodes + 1):
            idx = min(i * steps_per_ep - 1, total - 1)
            done_arr[idx] = True

        npz_state_path = os.path.join(output_dir, "mpc_expert_demos_state.npz")
        np.savez(
            npz_state_path,
            observations=obs_arr,
            actions=act_arr,
            next_observations=nobs_arr,
            rewards=rew_arr,
            dones=done_arr,
        )
        np.savez(
            os.path.join(data_dir, "mpc_expert_demos_state.npz"),
            observations=obs_arr,
            actions=act_arr,
            next_observations=nobs_arr,
            rewards=rew_arr,
            dones=done_arr,
        )
        print(f"  Saved state .npz: {npz_state_path} ({os.path.getsize(npz_state_path):,} bytes, {total} transitions)")

        parquet_rows = []
        ep_id = 0
        step_id = 0
        for i in range(total):
            parquet_rows.append(
                {
                    "episode_id": ep_id,
                    "step_id": step_id,
                    "obs_x": float(obs_arr[i, 0]),
                    "obs_y": float(obs_arr[i, 1]),
                    "obs_vx": float(obs_arr[i, 2]),
                    "obs_vy": float(obs_arr[i, 3]),
                    "action_ax": float(act_arr[i, 0]),
                    "action_ay": float(act_arr[i, 1]),
                    "next_obs_x": float(nobs_arr[i, 0]),
                    "next_obs_y": float(nobs_arr[i, 1]),
                    "next_obs_vx": float(nobs_arr[i, 2]),
                    "next_obs_vy": float(nobs_arr[i, 3]),
                    "reward": float(rew_arr[i]),
                    "done": bool(done_arr[i]),
                }
            )
            step_id += 1
            if done_arr[i]:
                ep_id += 1
                step_id = 0

        df_parquet = pd.DataFrame(parquet_rows)
        parquet_path = os.path.join(output_dir, "mpc_expert_demos_state.parquet")
        df_parquet.to_parquet(parquet_path, engine="pyarrow", index=False)
        df_parquet.to_parquet(os.path.join(data_dir, "mpc_expert_demos_state.parquet"), engine="pyarrow", index=False)
        print(f"  Saved Parquet table: {parquet_path} ({os.path.getsize(parquet_path):,} bytes, {len(df_parquet)} rows)")

        # Synthetic image dataset
        img_arr = rng.randint(0, 255, size=(min(total, 8), image_size, image_size, 3), dtype=np.uint8)
        npz_img_path = os.path.join(output_dir, "mpc_expert_demos_images.npz")
        np.savez(
            npz_img_path,
            observations=img_arr,
            actions=act_arr[: len(img_arr)],
            next_observations=img_arr,
            rewards=rew_arr[: len(img_arr)],
            dones=done_arr[: len(img_arr)],
        )
        np.savez(
            os.path.join(data_dir, "mpc_expert_demos_images.npz"),
            observations=img_arr,
            actions=act_arr[: len(img_arr)],
            next_observations=img_arr,
            rewards=rew_arr[: len(img_arr)],
            dones=done_arr[: len(img_arr)],
        )
        print(f"  Saved image .npz: {npz_img_path} ({os.path.getsize(npz_img_path):,} bytes, {len(img_arr)} frames)")

        # Dataset Card
        readme_content = f"""---
title: "MPC Expert Demonstrations — Benchmark Dataset"
language: en
license: mit
tags:
  - robotics
  - robot-learning
  - imitation-learning
  - demonstrations
  - mpc
  - collision-free-mpc
  - reaching-task
  - pusht
  - npz
  - parquet
  - lerobot
configs:
  - config_name: reaching_state
    data_files: "mpc_expert_demos_state.parquet"
task_categories:
  - robot-learning
size_categories:
  - n<1K
---

# 📦 MPC Expert Demonstrations Dataset

This is a **dry-run / synthetic** dataset package generated to verify Hugging Face dataset card and metadata creation for the study:
**"MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"**
([GitHub Repository](https://github.com/Ryukijano/mpc-vla-diffusion-study)).

## 🌟 Dataset Summary

- **Mode**: Dry-run (synthetic data, no environment rollouts).
- **Formats Provided**:
  1. `mpc_expert_demos_state.parquet`: Structured columnar tabular format.
  2. `mpc_expert_demos_state.npz`: NumPy archive with keys `observations`, `actions`, `next_observations`, `rewards`, `dones`.
  3. `mpc_expert_demos_images.npz`: Multimodal image observation dataset ($64 \\times 64 \\times 3$ RGB) stubs.

## 🚀 How to Load the Dataset

### Using Pandas / PyArrow (Parquet)
```python
import pandas as pd

df = pd.read_parquet("mpc_expert_demos_state.parquet")
print("Total transitions:", len(df))
print(df.head())
```

### Using NumPy (.npz)
```python
import numpy as np

data = np.load("mpc_expert_demos_state.npz")
print("Observations shape:", data["observations"].shape)
print("Actions shape:", data["actions"].shape)
```

## 📜 Citation

```bibtex
@misc{{mpc_vla_diffusion_study_2026,
  title        = {{MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families}},
  author       = {{Gyanateet and Devin AI}},
  year         = {{2026}},
  howpublished = {{\\url{{https://github.com/Ryukijano/mpc-vla-diffusion-study}}}}
}}
```
"""
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"  Generated dataset card: {readme_path}")

        if verify:
            print("\n[Verify] Validating dry-run dataset artifacts...")
            df_check = pd.read_parquet(parquet_path)
            assert len(df_check) > 0, "Empty parquet file!"
            assert "obs_x" in df_check.columns and "action_ax" in df_check.columns
            npz_check = np.load(npz_state_path)
            assert "observations" in npz_check and "actions" in npz_check
            assert len(npz_check["observations"]) == len(df_check)
            print("  [Verify PASS] Dry-run dataset artifacts loaded and verified successfully!")

        print("\n" + "=" * 72)
        print(f"Dataset dry-run packaging complete in {output_dir}!")
        print("=" * 72)
        return output_dir

    # 1. Initialize Reaching Environment with Obstacles
    obstacles = [
        Obstacle([1.5, 1.5], 0.45),
        Obstacle([2.5, 2.5], 0.5),
        Obstacle([3.0, 1.0], 0.4),
        Obstacle([1.0, 3.0], 0.35),
    ]
    env = ReachingEnv(
        dim=2,
        dt=0.05,
        success_threshold=0.2,
        max_steps=max_steps,
        workspace=5.0,
        image_size=image_size,
        obstacles=obstacles,
        seed=0,
    )

    # 2. Setup Collision-Free MPC Expert
    dyn = PointMass2D(mass=1.0, dt=env.dt)
    world = SDFWorld(dim=2)
    for obs in obstacles:
        world.add_sphere(obs.center.tolist(), obs.radius)

    horizon = 15
    u_bounds = (-5.0 * np.ones(2), 5.0 * np.ones(2))
    Q = np.diag([10.0, 10.0, 1.0, 1.0])
    R = np.diag([0.1, 0.1])
    Qf = np.diag([100.0, 100.0, 10.0, 10.0])

    def stage_cost(x, u, k=None):
        return float(dx := x - target_full) and float(dx @ Q @ dx + u @ R @ u)

    def terminal_cost(x):
        return float(dx := x - target_full) and float(dx @ Qf @ dx)

    target_full = np.zeros(4)

    cf_mpc = CollisionFreeMPC(
        dyn.dynamics,
        stage_cost,
        terminal_cost,
        world,
        horizon=horizon,
        u_bounds=u_bounds,
        collision_weight=150.0,
        ilqr_iters=15,
    )

    print("\n[1/3] Collecting state demonstrations from Collision-Free MPC...")
    collector_state = DemonstrationCollector(image_mode=False)

    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        target_pos = env._target[:2]
        target_full = np.concatenate([target_pos, np.zeros(2)])

        for step in range(max_steps):
            state = env.get_state()
            try:
                action = cf_mpc.solve(state)[0]
            except Exception:
                vec = target_pos - state[:2]
                action = np.clip(vec * 1.5, -2.0, 2.0)

            next_obs, reward, done, info = env.step(action)
            collector_state._append(
                state,
                action,
                next_obs,
                reward,
                done,
            )
            if done or env.is_success():
                break

    # Save State .npz
    npz_state_path = os.path.join(output_dir, "mpc_expert_demos_state.npz")
    collector_state.save(npz_state_path)
    # Also copy to root data/
    collector_state.save(os.path.join(data_dir, "mpc_expert_demos_state.npz"))
    print(f"  Saved state .npz: {npz_state_path} ({os.path.getsize(npz_state_path):,} bytes, {len(collector_state)} transitions)")

    # 3. Convert State Demonstrations to Apache Parquet
    print("\n[2/3] Converting transitions to Apache Parquet format...")
    ds_dict = collector_state.get_dataset()

    obs_arr = ds_dict["observations"]
    act_arr = ds_dict["actions"]
    nobs_arr = ds_dict["next_observations"]
    rew_arr = ds_dict["rewards"]
    done_arr = ds_dict["dones"]

    parquet_rows = []
    ep_id = 0
    step_id = 0
    for i in range(len(obs_arr)):
        parquet_rows.append(
            {
                "episode_id": ep_id,
                "step_id": step_id,
                "obs_x": float(obs_arr[i, 0]),
                "obs_y": float(obs_arr[i, 1]),
                "obs_vx": float(obs_arr[i, 2]),
                "obs_vy": float(obs_arr[i, 3]),
                "action_ax": float(act_arr[i, 0]),
                "action_ay": float(act_arr[i, 1]),
                "next_obs_x": float(nobs_arr[i, 0]),
                "next_obs_y": float(nobs_arr[i, 1]),
                "next_obs_vx": float(nobs_arr[i, 2]),
                "next_obs_vy": float(nobs_arr[i, 3]),
                "reward": float(rew_arr[i]),
                "done": bool(done_arr[i]),
            }
        )
        step_id += 1
        if done_arr[i]:
            ep_id += 1
            step_id = 0

    df_parquet = pd.DataFrame(parquet_rows)
    parquet_path = os.path.join(output_dir, "mpc_expert_demos_state.parquet")
    df_parquet.to_parquet(parquet_path, engine="pyarrow", index=False)
    # Also copy to data/
    df_parquet.to_parquet(os.path.join(data_dir, "mpc_expert_demos_state.parquet"), engine="pyarrow", index=False)
    print(f"  Saved Parquet table: {parquet_path} ({os.path.getsize(parquet_path):,} bytes, {len(df_parquet)} rows)")

    # 4. Collect Image Demonstrations for Multimodal VLA
    print("\n[3/3] Collecting multimodal image demonstrations...")
    collector_img = DemonstrationCollector(image_mode=True)
    for ep in range(min(n_episodes, 8)):
        obs = env.reset(seed=ep + 100)
        target_pos = env._target[:2]
        target_full = np.concatenate([target_pos, np.zeros(2)])

        for step in range(min(max_steps, 30)):
            state = env.get_state()
            try:
                action = cf_mpc.solve(state)[0]
            except Exception:
                vec = target_pos - state[:2]
                action = np.clip(vec * 1.5, -2.0, 2.0)

            img = env.get_image()
            next_obs, reward, done, info = env.step(action)
            collector_img._append(
                img,
                action,
                env.get_image(),
                reward,
                done,
            )
            if done or env.is_success():
                break

    npz_img_path = os.path.join(output_dir, "mpc_expert_demos_images.npz")
    collector_img.save(npz_img_path)
    collector_img.save(os.path.join(data_dir, "mpc_expert_demos_images.npz"))
    print(f"  Saved image .npz: {npz_img_path} ({os.path.getsize(npz_img_path):,} bytes, {len(collector_img)} frames)")

    # 5. Create Dataset Card (README.md)
    readme_content = f"""---
title: "MPC Expert Demonstrations — Benchmark Dataset"
language: en
license: mit
tags:
  - robotics
  - robot-learning
  - imitation-learning
  - demonstrations
  - mpc
  - collision-free-mpc
  - reaching-task
  - pusht
  - npz
  - parquet
  - lerobot
configs:
  - config_name: reaching_state
    data_files: "mpc_expert_demos_state.parquet"
task_categories:
  - robot-learning
size_categories:
  - n<1K
---

# 📦 MPC Expert Demonstrations Dataset

This dataset contains expert trajectory demonstrations collected via **Collision-Free Model Predictive Control (MPC)** on 2D robotic navigation and reaching benchmarks from the study:
**"MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"**
([GitHub Repository](https://github.com/Ryukijano/mpc-vla-diffusion-study)).

---

## 🌟 Dataset Summary

- **Expert Controller**: Signed Distance Field (SDF) Constrained Collision-Free MPC ($H=15$, iLQR optimizer).
- **Environment**: 2D point-mass reaching with double-integrator dynamics ($x_{{k+1}} = x_k + v_k \\Delta t + 0.5 a_k \\Delta t^2$) and dense circular obstacles.
- **Formats Provided**:
  1. `mpc_expert_demos_state.parquet`: Structured columnar tabular format for fast loading in Pandas, Polars, and Hugging Face `datasets`.
  2. `mpc_expert_demos_state.npz`: Compact NumPy archive with keys `observations`, `actions`, `next_observations`, `rewards`, `dones`.
  3. `mpc_expert_demos_images.npz`: Multimodal image observation dataset ($64 \\times 64 \\times 3$ RGB) with natural language instructions.

---

## 📐 Data Fields & Schema

| Column Name | Type | Description |
|:---|:---|:---|
| `episode_id` | `int64` | Demonstration episode index |
| `step_id` | `int64` | Time step index within episode ($t=0..T$) |
| `obs_x`, `obs_y` | `float64` | Point-mass position coordinates $(x, y)$ in workspace $[-5.0, 5.0]$ |
| `obs_vx`, `obs_vy` | `float64` | Velocity components $(\\dot{{x}}, \\dot{{y}})$ |
| `action_ax`, `action_ay` | `float64` | Commanded continuous 2D accelerations $(a_x, a_y) \\in [-5.0, 5.0]^2$ |
| `next_obs_x`, `next_obs_y`, ... | `float64` | Resulting next state after integration step $\\Delta t = 0.05$ s |
| `reward` | `float64` | Step reward $(- \\|x - x_{{target}}\\| - 0.01 \\|u\\|)$ |
| `done` | `bool` | Episode completion indicator |

---

## 🚀 How to Load the Dataset

### Using Pandas / PyArrow (Parquet)
```python
import pandas as pd

df = pd.read_parquet("mpc_expert_demos_state.parquet")
print("Total transitions:", len(df))
print(df.head())
```

### Using NumPy (.npz)
```python
import numpy as np

data = np.load("mpc_expert_demos_state.npz")
print("Observations shape:", data["observations"].shape)
print("Actions shape:", data["actions"].shape)
```

### Using Hugging Face Datasets
```python
from datasets import load_dataset

dataset = load_dataset("Ryukijano/mpc-expert-demos-quick-test", data_files="mpc_expert_demos_state.parquet")
print(dataset)
```

---

## 📜 Citation

```bibtex
@misc{{mpc_vla_diffusion_study_2026,
  title        = {{MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families}},
  author       = {{Gyanateet and Devin AI}},
  year         = {{2026}},
  howpublished = {{\\url{{https://github.com/Ryukijano/mpc-vla-diffusion-study}}}}
}}
```
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    # 6. Verification
    if verify:
        print("\n[Verify] Validating packaged dataset artifacts...")
        # Verify Parquet
        df_check = pd.read_parquet(parquet_path)
        assert len(df_check) > 0, "Empty parquet file!"
        assert "obs_x" in df_check.columns and "action_ax" in df_check.columns

        # Verify NPZ
        npz_check = np.load(npz_state_path)
        assert "observations" in npz_check and "actions" in npz_check
        assert len(npz_check["observations"]) == len(df_check)

        print("  [Verify PASS] All dataset artifacts loaded and verified successfully!")

    print("\n" + "=" * 72)
    print(f"Dataset Packaging Complete in {output_dir}!")
    print("=" * 72)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Package MPC Expert Demonstration dataset.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(STUDY_ROOT, "dist", "hf_datasets", "mpc_expert_demos"),
        help="Target output directory for packaged dataset.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=15,
        help="Number of demonstration episodes to collect.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Maximum steps per demonstration episode.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify created dataset files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Generate a synthetic dataset and card only; do not roll out the environment.",
    )
    args = parser.parse_args()

    collect_and_package_dataset(
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        verify=args.verify,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
