---
language: en
license: mit
library_name: pytorch
pipeline_tag: robotics
tags:
  - robotics
  - robot-learning
  - imitation-learning
  - vla
  - vision-language-action
  - smallvla
  - diffusion-policy
  - mpc
  - reaching-task
  - pusht
  - action-generation
datasets:
  - Ryukijano/mpc-expert-demos-quick-test
---

# SmallVLA quick checkpoint — MPC vs VLA vs Diffusion study

This is a small, compute-matched **Vision-Language-Action (VLA)** model trained as one of the baselines for the open-source study _"MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"_.
It is designed to be comparable in parameter count to the small diffusion/flow policies in the same study, so that differences in performance are attributable to the method rather than raw model capacity.

## Model architecture

| Component | Details |
|-----------|---------|
| Vision encoder | Small ViT-Base (patch 16, 96×96 RGB, 12 layers, 768 hidden dim, 12 heads, no pretrained weights) |
| Language encoder | CLIP `openai/clip-vit-base-patch32` (fallback to BoW) projecting to 256-dim |
| Fusion | Concatenate vision (768) + language (256) features, MLP project to 256-dim with ReLU + Dropout(0.1) |
| Action head | 4-layer MLP, hidden dim 256, output `(horizon, action_dim)` = `(8, 2)` |
| Total parameters | ~149.4 M |
| Output shape | `(8, 2)` action chunk |

## Training details

| Setting | Value |
|---------|-------|
| Task | PushT (T-shaped block pushing, image + language) |
| Demonstrations | 50 expert episodes (~5,000 transitions), instruction: *"push the T block to the target area"* |
| Image size | 96 × 96 × 3 RGB |
| Action dim / horizon | 2 / 8 |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 1.0e-4 (Adam, weight decay 1.0e-5) |
| Hardware | DGX Spark / GB10 (Grace Blackwell), `cuda` |
| Training time | ~556 s |
| Final epoch loss | 0.012850 |
| Checkpoint size | ~570 MB (`small_vla_pusht.pt`) |

> **Note:** This model is trained on **PushT image+language demonstrations**. It is not evaluated in the state-only 2-D reaching quick test.

## Evaluation results (quick-test placeholders)

The quick-test sweep used `run_experiments.py --quick` on the `reaching_2d` benchmark with one seed and five episodes. Because SmallVLA requires image-based demonstrations, it was skipped in that quick test.

| Metric | Quick-test value | Notes |
|--------|------------------|-------|
| `success_rate_mean` | **N/A** | Not evaluated in reaching quick test (image required) |
| `path_length_mean` | **TBD** | Full run pending |
| `collision_rate_mean` | **TBD** | Full run pending |
| `latency_ms_mean` | **TBD** | Full run pending |
| `mode_coverage_mean` | **TBD** | Full run pending |

These numbers are placeholders from `results/quick_test/report/master_comparison_table.csv` and will be replaced by the full PushT / reaching-with-image benchmark results once available.

## Usage

```python
import numpy as np
import sys
from pathlib import Path

# Add the study root to your Python path
sys.path.insert(0, str(Path("/path/to/mpc-vla-diffusion-study").resolve()))

from vla_baselines.small_vla import SmallVLA

# Load the model (use text_backend="auto" to load CLIP; "bow" for CPU/no-internet)
vla = SmallVLA.load(
    "small_vla_pusht.pt",
    device="cuda",
    text_backend="auto",
)

# Predict an action chunk from an RGB image + language instruction
image = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
instruction = "Push the T-shaped block onto the target T outline."

action = vla.predict_action(image, instruction)  # (8, 2)
print("Predicted action chunk:", action)
```

The `predict_action` helper normalizes the image to ImageNet statistics, runs the forward pass, and returns a numpy array of shape `(horizon, action_dim)`.

## Citation

```bibtex
@misc{mpc_vla_diffusion_study_2026,
  title        = {MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families},
  author       = {Gyanateet and Devin AI},
  year         = {2026},
  howpublished = {\url{https://github.com/Ryukijano/mpc-vla-diffusion-study}}
}
```
