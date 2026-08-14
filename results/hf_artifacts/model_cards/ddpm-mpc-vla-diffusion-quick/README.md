---
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
  - mpc
  - reaching-task
  - pusht
datasets:
  - Ryukijano/mpc-expert-demos-quick-test
---

# DDPM Diffusion Policy quick checkpoint — MPC vs VLA vs Diffusion study

This is a small **DDPM Diffusion Policy** baseline for the open-source study _"MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"_.
It learns to map an observation vector to an 8-step action chunk by training a 1D temporal U-Net to predict and remove Gaussian noise conditioned on the state.

## Model architecture

| Component | Details |
|-----------|---------|
| Backbone | Conditional 1D Temporal U-Net (`ConditionalUnet1D`) |
| Conditioning | Global state conditioning via FiLM; sinusoidal timestep embedding |
| Action dim / horizon | 2 / 8 |
| Observation dim | 5 (PushT: block_x, block_y, block_angle, agent_x, agent_y) |
| Diffusion steps | 50 denoising steps |
| Hidden dims | `[128, 256, 512]` (3 U-Net levels) |
| Conv kernel | 3 |
| Activation | Mish + GroupNorm |
| Total parameters | ~11.1 M |
| Output shape | `(8, 2)` action chunk |

## Training details

| Setting | Value |
|---------|-------|
| Task | PushT (state-only action-sequence demonstrations) |
| Demonstrations | 50 PushT expert episodes, chunked into `(obs, action_seq)` pairs |
| Loss | DDPM simplified objective: MSE between predicted and true noise |
| Optimizer | Adam, lr = 1.0e-3 |
| Epochs | 40 |
| Batch size | 64 |
| Diffusion steps (training sampling) | 50 |
| Hardware | DGX Spark / GB10 (Grace Blackwell), `cuda` |
| Training time | ~61 s |
| Final epoch loss | 0.023778 |
| Checkpoint size | ~42 MB (`ddpm_pusht.pt`) |

## Evaluation results (quick-test placeholders)

The numbers below are taken from `results/quick_test/report/master_comparison_table.csv` for the **DDPM Policy** row on the 2-D `reaching` quick test (1 seed, 5 episodes). They are provided as placeholders only; full PushT results will be added once the complete experimental matrix is run.

| Metric | Quick-test value |
|--------|------------------|
| `success_rate_mean` | **0.0** |
| `success_rate_std` | **0.0** |
| `path_length_mean` | **4.377** |
| `collision_rate_mean` | **0.030** |
| `latency_ms_mean` | **16.25** |
| `mode_coverage_mean` | **0.0** |
| `n_seeds` | 1 |
| `n_episodes` | 5 |

> **Disclaimer:** These quick-test numbers are intentionally limited and should not be interpreted as final performance. The full benchmark suite will evaluate this checkpoint on PushT and other tasks.

## Usage

```python
import sys
from pathlib import Path

# Add the study root to your Python path
sys.path.insert(0, str(Path("/path/to/mpc-vla-diffusion-study").resolve()))

import numpy as np
import torch
from diffusion_baselines.ddpm_policy import DiffusionPolicy

# Load from the saved checkpoint
policy = DiffusionPolicy.from_checkpoint("ddpm_pusht.pt", device="cuda")
policy.net.eval()

# Provide a single PushT observation (5-dim state)
obs = np.array([
    0.0,  # block_x
    0.0,  # block_y
    0.0,  # block_angle
    0.0,  # agent_x
    0.0,  # agent_y
], dtype=np.float32)

with torch.no_grad():
    action_chunk = policy.sample(obs, num_samples=1).cpu().numpy().squeeze(0)

print("DDPM action chunk:", action_chunk)  # (8, 2)
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
