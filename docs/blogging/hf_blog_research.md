# Hugging Face Blog Research for MPC vs VLA vs Diffusion

**Date:** August 2026  
**Purpose:** Landscape analysis and format guide for a HF blog on the comparison study

---

## 1. HF Blog Landscape for Robotics/Control Research

| # | Title | URL | Date | Angle | Key Elements |
|---|-------|-----|------|-------|--------------|
| 1 | LeRobot v0.6.0: Imagine, Evaluate, Improve | https://huggingface.co/blog/lerobot-release-v060 | 2026 | World model policies (VLA-JEPA, FastWAM, LingBot-VA) | Code, benchmarks, CLI, gifs |
| 2 | π0 and π0-FAST | https://huggingface.co/blog/pi0 | 2025 | VLA flow models on HF LeRobot | Code, flow matching, robot demos |
| 3 | Asynchronous Robot Inference | https://huggingface.co/blog/async-robot-inference | 2025 | Decoupling action prediction and execution | Code (gRPC), architecture, gifs |
| 4 | NVIDIA Cosmos Policy | https://huggingface.co/blog/nvidia/cosmos-policy-for-robot-control | 2025 | World foundation model for manipulation | Checkpoints, LIBERO/RoboCasa benchmarks |
| 5 | SmolVLA | https://huggingface.co/blog/smolvla | 2025 | Compact open-source VLA | Checkpoints, training recipes, real demos |
| 6 | Open-source AI Robotics | https://huggingface.co/blog/clem/opensourceairobotics | 2025 | Open models, datasets, hardware | SO-100 tutorial, ecosystem |
| 7 | Generalist Robot Policy Evaluation in Isaac Lab-Arena | https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot | 2025 | VLA evaluation in sim | Isaac Lab-Arena integration |

### Key observations
- World models and VLAs are major 2025-2026 themes.
- Most successful posts link to HF Hub models, datasets, or Spaces.
- Collaboration posts (NVIDIA, Physical Intelligence, AllenAI) are common.
- Community posts from individual researchers are also featured.

---

## 2. HF Blog Format and Style

### Frontmatter

```yaml
---
title: "Your Title"
thumbnail: /blog/assets/your-post/thumbnail.png
authors:
- user: Ryukijano
- user: devin-ai
---
```

### Markdown conventions
- **Images:** upload body images to `huggingface/documentation-images` dataset under `blog/{post-slug}/`.
- **Thumbnails:** 1300x650 px, store in blog repo `assets/` (official) or upload with post (community).
- **References:** use code format for repo IDs: `` `lerobot/act_aloha` ``.
- **Code blocks:** syntax-highlighted.

### Length
- Typical technical post: 1,400–2,500 words.
- Robotics posts often 1,500–2,000 words with multiple gifs/diagrams.

### Tone
- Technical but accessible.
- Practical, code-first, reproducible.
- Community-oriented.
- TL;DR section strongly recommended.

### Community vs Official Blog
- **Community blog** (`huggingface.co/new-blog`) — recommended for external contributors.
- **Official blog** (`huggingface/blog` repo) — for HF collaborations, requires PR.
- HF docs explicitly suggest community blog for external contributors.

---

## 3. What Makes a Successful HF Robotics Blog?

- **Open-source artifact:** model, dataset, or code repo on HF Hub/GitHub.
- **Visuals:** gifs, architecture diagrams, benchmark charts, robot videos.
- **Code snippets:** installation, training, inference.
- **Reproducibility:** configs, seeds, environment details.
- **Benchmarks:** clear metrics and comparisons.

### Typical artifact set
1. GitHub repo with training/inference code.
2. HF Hub model checkpoint(s).
3. HF Hub dataset (e.g. LeRobotDataset format).
4. HF Space demo (optional but high impact).
5. Collection tying artifacts together.

---

## 4. Recommended Path for This Study

### Blog type
**Community blog article** at `huggingface.co/new-blog` is the best first step because:
- External contributor, not an HF collaboration.
- Fastest publication.
- Can update as research progresses.
- No PR review required.

**Note:** the GitHub repo `github.com/Ryukijano/mpc-vla-diffusion-study` exists and is public; the subagent search may have missed it due to cache.

### Artifacts to create before/at blog launch

1. **GitHub repo** (already done): `github.com/Ryukijano/mpc-vla-diffusion-study`
2. **HF Model repo(s)** for trained checkpoints (SmallVLA, DiffusionPolicy, MIP).
3. **HF Dataset repo** for expert demonstrations (LeRobotDataset or npz format).
4. **HF Space** with an interactive comparison demo or figure gallery.
5. **HF Collection** grouping the repo + models + dataset + Space.

### Minimum viable blog
- Can link to GitHub only, but HF Hub artifacts increase discoverability.
- Recommended hybrid: GitHub for code + HF Hub for one model + one dataset + one Space.

### Blog title suggestions
1. "Comparing MPC, VLA, and Diffusion Policies for Robot Control: An Open-Source Study"
2. "Do You Need Diffusion? A Controlled Comparison of Robot Control Architectures"
3. "MPC vs VLA vs Diffusion: An Open-Source Study on the DGX Spark"
4. "Beyond the Hype: MPC, VLA, Diffusion, and World Action Models Compared"
5. "The Robot Control Pareto Frontier: Latency, Generalization, and Safety"

### Section template
- TL;DR
- Introduction / Hook
- Background (families)
- Study design
- Quick test results
- Ablation study
- Pareto analysis
- Roadmap to real robots
- Artifacts & reproducibility
- Call to action

---

## 5. Sources

- https://huggingface.co/blog
- https://huggingface.co/blog-explorers
- https://huggingface.co/docs/hub/blog-articles
- https://huggingface.co/blog/lerobot-release-v060
- https://huggingface.co/blog/async-robot-inference
- https://huggingface.co/blog/smolvla
- https://huggingface.co/blog/pi0
- https://huggingface.co/blog/clem/opensourceairobotics
