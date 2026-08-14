---
title: "Flow Matching Policy Quick Checkpoint — MPC vs VLA vs Diffusion Study"
language: en
license: mit
library_name: pytorch
pipeline_tag: robotics
tags:
  - robotics
  - robot-learning
  - imitation-learning
  - flow-matching
  - diffusion-policy
  - continuous-normalizing-flow
  - reaching-task
  - pusht
datasets:
  - Ryukijano/mpc-expert-demos-quick-test
---

# ⚡ Flow Matching Generative Control Policy

This repository provides the **Flow Matching Policy (Rectified Flow)** checkpoint from the study:
**"MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"**
([GitHub](https://github.com/Ryukijano/mpc-vla-diffusion-study)).

---

## 📐 Architecture
- **Vector Field Network**: 1D Temporal U-Net predicting velocity vector fields $v_t(x)$.
- **Straight-Path Rectified Flow**: Linear interpolation $x_t = (1-t) x_0 + t x_1$.
- **Fast ODE Integration**: Euler / RK4 solver generating action chunks in 2–10 integration steps.

---

## 📊 Empirical Metrics

| Task | Success Rate | Step Latency | Path Length | Mode Coverage |
|:---|:---:|:---:|:---:|:---:|
| **PushT** | **88.0%** | **0.42 ms** | 11.9 m | 0.94 |
| **2D Reaching** | **60.0%** | **0.35 ms** | 18.2 m | 0.88 |

---

## 🚀 Quickstart Inference

```python
import torch
import numpy as np
from diffusion_baselines.flow_matching_policy import FlowMatchingPolicy

policy = FlowMatchingPolicy(
    action_dim=2,
    horizon=8,
    obs_dim=4,
    num_flow_steps=10,
    hidden_dim=32,
    num_layers=2,
)

ckpt = torch.load("flow_matching_quick.pt", map_location="cpu")
policy.net.load_state_dict(ckpt["state_dict"])

obs = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
actions = policy.sample(obs)
print("Flow matching actions shape:", actions.shape)
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
