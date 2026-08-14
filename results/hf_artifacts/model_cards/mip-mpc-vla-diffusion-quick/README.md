---
language: en
license: mit
library_name: numpy
pipeline_tag: robotics
tags:
  - robotics
  - robot-learning
  - imitation-learning
  - mip
  - minimal-iterative-policy
  - ablation
  - action-generation
  - mpc
  - reaching-task
  - pusht
datasets:
  - Ryukijano/mpc-expert-demos-quick-test
---

# Minimal Iterative Policy (MIP) quick checkpoint — MPC vs VLA vs Diffusion study

This is the **Minimal Iterative Policy (MIP)** ablation baseline for the open-source study _"MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"_.
It is a deliberately simple two-step regression network with Gaussian noise injection between the steps, designed to isolate the "iterative compute + stochasticity" ingredients hypothesized to explain much of diffusion/flow policy performance.

## Model architecture

| Component | Details |
|-----------|---------|
| Step 1 network | Single-hidden-layer ReLU MLP: `state_dim` -> 128 hidden -> flattened action |
| Step 2 network | Single-hidden-layer ReLU MLP: flattened action -> 128 hidden -> flattened action |
| Action dim / horizon | 2 / 8 (flattened dim = 16) |
| Observation (state) dim | 5 (PushT: block_x, block_y, block_angle, agent_x, agent_y) |
| Noise injection | Gaussian noise with `noise_std = 0.1` between step 1 and step 2 |
| Implementation | Pure NumPy (no PyTorch required at inference) |
| Total parameters | ~7,072 |
| Output shape | `(8, 2)` action chunk |

## Training details

| Setting | Value |
|---------|-------|
| Task | PushT (state-only action-sequence demonstrations) |
| Demonstrations | 50 PushT expert episodes, chunked into `(obs, action_seq)` pairs |
| Step 1 target | Noisy version of the expert action sequence |
| Step 2 target | Clean expert action sequence, conditioned on noisy step-1 output |
| Optimizer | Adam (manual NumPy implementation) |
| Learning rate | 1.0e-3 |
| Epochs | 40 per step |
| Batch size | 32 |
| Noise std | 0.1 |
| Hardware | CPU (NumPy) |
| Training time | ~5 s |
| Final step 1 loss | 0.019436 |
| Final step 2 loss | 0.000548 |
| Checkpoint size | ~59 KB (`mip_pusht.npz`) |

## Evaluation results (quick-test placeholders)

The numbers below are taken from `results/quick_test/report/master_comparison_table.csv` for the **MIP (standalone)** row on the 2-D `reaching` quick test (1 seed, 5 episodes). They are provided as placeholders only; full PushT results will be added once the complete experimental matrix is run.

| Metric | Quick-test value |
|--------|------------------|
| `success_rate_mean` | **0.4** |
| `success_rate_std` | **0.0** |
| `path_length_mean` | **66.824** |
| `collision_rate_mean` | **0.045** |
| `latency_ms_mean` | **0.007** |
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
from mpc_baselines_repo.src.diffusion_warm_start.minimal_iterative_policy import MinimalIterativePolicy

# Construct with the same dimensions used during training
mip = MinimalIterativePolicy(
    state_dim=5,
    action_dim=2,
    horizon=8,
    hidden_dim=128,
    noise_std=0.1,
    seed=42,
)

# Load the saved NumPy checkpoint
mip.load("mip_pusht.npz")

# Provide a single PushT observation (5-dim state)
obs = np.array([
    0.0,  # block_x
    0.0,  # block_y
    0.0,  # block_angle
    0.0,  # agent_x
    0.0,  # agent_y
], dtype=np.float32)

action_chunk = mip.sample(obs, num_samples=1)[0]
print("MIP action chunk:", action_chunk)  # (8, 2)
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
