---
title: "Minimal Iterative Policy (MIP) — MPC vs VLA vs Diffusion Study"
language: en
license: mit
library_name: pytorch
pipeline_tag: robotics
tags:
  - robotics
  - robot-learning
  - imitation-learning
  - minimal-iterative-policy
  - mip
  - supervised-iterative-compute
  - stochasticity-injection
  - mpc
  - reaching-task
  - pusht
datasets:
  - Ryukijano/mpc-expert-demos-quick-test
---

# ⚡ Minimal Iterative Policy (MIP)

This repository hosts the **Minimal Iterative Policy (MIP)** baseline from the study:
**"MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"**
([GitHub](https://github.com/Ryukijano/mpc-vla-diffusion-study)), implementing the theoretical framework of **Simchowitz et al. (2026)** (*"Much Ado About Noising"*, [arXiv:2512.01809](https://arxiv.org/abs/2512.01809)).

---

## 💡 Key Conceptual Insight

Simchowitz et al. demonstrate that the performance gain of diffusion models in robotics is primarily driven by:
1. **Supervised iterative compute**: Refining actions over $K=2$ steps rather than a single step.
2. **Stochasticity injection**: Injecting Gaussian noise between iterations to enhance manifold adherence under out-of-distribution drift.

MIP replaces 100-step reverse diffusion chains with a **2-step regression + noise** policy, cutting latency from milliseconds to microseconds.

---

## 📊 Empirical Metrics

| Metric | Standalone MIP | Full DDPM (T=100) | Pure Regression (RCP) |
|:---|:---:|:---:|:---:|
| **Inference Latency** | **0.0068 ms (6.8 µs)** | 0.812 ms (812 µs) | 0.0056 ms (5.6 µs) |
| **Speedup vs DDPM** | **119x Faster** | 1x (Baseline) | 145x Faster |
| **Success Rate (PushT)** | **65.0%** | 85.0% | 20.0% |

---

## 🚀 Quickstart Inference

```python
import numpy as np
from src.diffusion_warm_start import MinimalIterativePolicy

mip = MinimalIterativePolicy(
    state_dim=4,
    action_dim=2,
    horizon=15,
    hidden_dim=32,
    noise_std=0.1,
)

state = np.array([2.0, -1.0, 0.0, 0.0])
actions = mip.sample(state, num_samples=1)[0]
print("MIP action trajectory:", actions.shape)
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
