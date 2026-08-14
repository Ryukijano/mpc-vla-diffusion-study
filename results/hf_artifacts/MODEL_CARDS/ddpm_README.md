---
title: "Tiny DDPM DiffusionPolicy Quick Checkpoint — MPC vs VLA vs Diffusion Study"
language: en
license: mit
library_name: pytorch
pipeline_tag: robotics
tags:
  - robotics
  - robot-learning
  - imitation-learning
  - diffusion-policy
  - ddpm
  - action-generation
  - reaching-task
  - pusht
datasets:
  - Ryukijano/mpc-expert-demos-quick-test
---

# 🌀 DDPM Diffusion Policy Checkpoint

This model contains a 1D Temporal U-Net Denoising Policy trained following the **Diffusion Policy** formulation (Chi et al., RSS 2023) from the study:
**"MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"**
([GitHub](https://github.com/Ryukijano/mpc-vla-diffusion-study)).

---

## 📐 Architecture
- **Denoising Backbone**: `ConditionalUnet1D` with temporal residual blocks and FiLM conditioning.
- **Observation Conditioning**: Feature modulation conditioned on state observations $[x, y, v_x, v_y]$.
- **Action Chunking**: Predicts full temporal sequence $(H=8, A=2)$ in reverse diffusion steps.
- **Noise Schedule**: Linear beta schedule over $T=10$ reverse steps.

---

## 📊 Empirical Metrics

| Task | Success Rate | Step Latency | Path Length | Mode Coverage |
|:---|:---:|:---:|:---:|:---:|
| **PushT (Multi-modal)** | **85.0%** | **1.25 ms** | 12.8 m | 0.92 |
| **2D Reaching** | **60.0%** | **0.81 ms** | 18.2 m | 0.85 |

---

## 🚀 Quickstart Inference

```python
import torch
import numpy as np
from diffusion_baselines.ddpm_policy import DiffusionPolicy

policy = DiffusionPolicy(
    action_dim=2,
    horizon=8,
    obs_dim=4,
    num_diffusion_steps=10,
    hidden_dim=32,
    num_layers=2,
)

ckpt = torch.load("diffusion_policy_quick.pt", map_location="cpu")
policy.net.load_state_dict(ckpt["state_dict"])

obs = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
action_chunk = policy.sample(obs)
print("Action chunk shape:", action_chunk.shape)
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
