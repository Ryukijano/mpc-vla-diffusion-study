---
language: en
title: "MPC Expert Demonstrations — quick test set"
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
  - multimodal
task_categories:
  - robotics
size_categories:
  - n<1K
---

# MPC Expert Demonstrations — quick test set

This dataset contains quick-test expert trajectory demonstrations for the open-source study _"MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"_.
It is intended as a small, reproducible starting point for imitation-learning baselines (VLA, diffusion, flow, MIP) before the full large-scale dataset is collected.

## Dataset summary

The quick-test set covers two benchmark environments used in the study:

- **2-D Reaching** (`reaching_2d` / `reaching_2d_cluttered`) — a point-mass double-integrator must reach a target while optionally avoiding circular obstacles.
- **PushT** — a T-shaped block on a 2-D plane must be pushed by a point agent onto a target T outline (multi-modal solutions).

For both tasks, demonstrations are collected from a strong expert:

- **Reaching**: a **Collision-Free Model Predictive Control (MPC)** expert based on SDF-constrained iLQR, with signed-distance-field collision avoidance.
- **PushT**: a **geometric push heuristic** that positions the agent behind the block relative to the target and pushes forward, clipped to control bounds.

## Files and format

| File | Description | Format |
|------|-------------|--------|
| `mpc_expert_demos_state.parquet` | Tabular state-action transitions | Apache Parquet |
| `mpc_expert_demos_state.npz` | NumPy archive of state-only transitions | NumPy `.npz` |
| `mpc_expert_demos_images.npz` | Multimodal image observations + language instructions | NumPy `.npz` |

### Parquet schema (reaching state)

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | `int64` | Episode index |
| `step_id` | `int64` | Time step within episode |
| `obs_x`, `obs_y` | `float64` | Point-mass position |
| `obs_vx`, `obs_vy` | `float64` | Point-mass velocity |
| `action_ax`, `action_ay` | `float64` | Commanded 2-D accelerations |
| `next_obs_x`, `next_obs_y`, ... | `float64` | Resulting next state |
| `reward` | `float64` | Step reward |
| `done` | `bool` | Episode completion flag |

### NumPy archive keys

Both `.npz` files contain at least:

- `observations`: state (or image) at time `t`
- `actions`: action taken at time `t`
- `next_observations`: resulting observation at time `t+1`
- `rewards`: scalar step reward
- `dones`: episode termination flag

## Collection method

The demonstrations are produced by the study's built-in expert controllers, not by human teleoperation:

1. **Reaching / cluttered reaching**: at every timestep, the current state is passed to the `CollisionFreeMPC` solver (horizon 15, iLQR iterations 15, control bounds `[-5.0, 5.0]`, signed-distance collision weight 150.0). If the solver fails, a simple proportional-to-target fallback is used.
2. **PushT**: the scripted `pusht_expert_action` computes the desired agent position behind the block relative to the target, moves toward it, and pushes, with all actions clipped to `[-1.0, 1.0]`.

The resulting trajectories are chunked into horizon-length action sequences (`horizon=8` by default) for sequence-modelling baselines.

## Usage

### Load with Pandas (Parquet)

```python
import pandas as pd

df = pd.read_parquet("mpc_expert_demos_state.parquet")
print(df.head())
print("Transitions:", len(df))
```

### Load with NumPy

```python
import numpy as np

data = np.load("mpc_expert_demos_state.npz")
print("Observations shape:", data["observations"].shape)
print("Actions shape:", data["actions"].shape)
```

### Load with Hugging Face `datasets`

```python
from datasets import load_dataset

dataset = load_dataset(
    "Ryukijano/mpc-expert-demos-quick-test",
    data_files="mpc_expert_demos_state.parquet",
)
print(dataset)
```

### Use with the study baselines

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/path/to/mpc-vla-diffusion-study").resolve()))

from benchmarks.demonstration_collector import DemonstrationCollector

# Load state-only demonstrations for DDPM / Flow / MIP
collector = DemonstrationCollector(image_mode=False)
collector.load("mpc_expert_demos_state.npz")
demos = collector.get_demos()  # list of (obs, action_seq)

# Load image+language demonstrations for SmallVLA
img_collector = DemonstrationCollector(image_mode=True)
img_collector.load("mpc_expert_demos_images.npz")
vla_demos = img_collector.get_vla_demos()  # list of {"image", "instruction", "action"}
```

## Citation

```bibtex
@misc{mpc_vla_diffusion_study_2026,
  title        = {MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families},
  author       = {Gyanateet and Devin AI},
  year         = {2026},
  howpublished = {\url{https://github.com/Ryukijano/mpc-vla-diffusion-study}}
}
```
