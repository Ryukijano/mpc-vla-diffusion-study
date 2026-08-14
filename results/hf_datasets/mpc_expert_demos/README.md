---
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
- **Environment**: 2D point-mass reaching with double-integrator dynamics ($x_{k+1} = x_k + v_k \Delta t + 0.5 a_k \Delta t^2$) and dense circular obstacles.
- **Formats Provided**:
  1. `mpc_expert_demos_state.parquet`: Structured columnar tabular format for fast loading in Pandas, Polars, and Hugging Face `datasets`.
  2. `mpc_expert_demos_state.npz`: Compact NumPy archive with keys `observations`, `actions`, `next_observations`, `rewards`, `dones`.
  3. `mpc_expert_demos_images.npz`: Multimodal image observation dataset ($64 \times 64 \times 3$ RGB) with natural language instructions.

---

## 📐 Data Fields & Schema

| Column Name | Type | Description |
|:---|:---|:---|
| `episode_id` | `int64` | Demonstration episode index |
| `step_id` | `int64` | Time step index within episode ($t=0..T$) |
| `obs_x`, `obs_y` | `float64` | Point-mass position coordinates $(x, y)$ in workspace $[-5.0, 5.0]$ |
| `obs_vx`, `obs_vy` | `float64` | Velocity components $(\dot{x}, \dot{y})$ |
| `action_ax`, `action_ay` | `float64` | Commanded continuous 2D accelerations $(a_x, a_y) \in [-5.0, 5.0]^2$ |
| `next_obs_x`, `next_obs_y`, ... | `float64` | Resulting next state after integration step $\Delta t = 0.05$ s |
| `reward` | `float64` | Step reward $(- \|x - x_{target}\| - 0.01 \|u\|)$ |
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
@misc{mpc_vla_diffusion_study_2026,
  title        = {MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families},
  author       = {Gyanateet and Devin AI},
  year         = {2026},
  howpublished = {\url{https://github.com/Ryukijano/mpc-vla-diffusion-study}}
}
```
