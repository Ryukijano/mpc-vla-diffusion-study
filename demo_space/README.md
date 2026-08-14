---
title: MPC vs VLA vs Diffusion Policy Arena
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 MPC vs VLA vs Diffusion Policy Arena

An interactive benchmarking playground and empirical explorer comparing **Classical Model Predictive Control (MPC)**, **Vision-Language-Action (VLA)** models, **Generative Diffusion / Flow Control Policies (GCPs)**, and their hybrids on robotic control tasks.

🔗 **GitHub Repository:** [Ryukijano/mpc-vla-diffusion-study](https://github.com/Ryukijano/mpc-vla-diffusion-study)  
📄 **Pre-registered Study:** 12 Research Questions, 8 Comprehensive Experiment Protocols (EXP-001 through EXP-008).

---

## 🌟 Interactive Features

### 1. 🎮 Interactive Policy Arena
- **Live closed-loop simulation** on:
  - **2D Reaching** (point-mass double integrator)
  - **2D Reaching (Cluttered)** (multi-obstacle navigation with SDF obstacle collision checks)
  - **PushT** (canonical multi-modal T-block pushing task from Chi et al., 2023)
- **Select any controller family**:
  - `Linear MPC`: QP-based receding-horizon control
  - `Nonlinear MPC (iLQR)`: Iterative LQR with cost-based obstacle avoidance
  - `Collision-Free MPC`: Signed Distance Field (SDF) potential-constrained MPC
  - `Diffusion Warm-Start MPC`: Diffusion trajectory proposal + MPC refinement hybrid
  - `Minimal Iterative Policy (MIP)`: 2-step iterative regression with noise injection (Simchowitz et al., 2026)
  - `Diffusion Policy (DDPM)`: 1D temporal U-Net denoising policy
  - `Flow Matching Policy`: Continuous flow-matching rectified-flow policy
  - `SmallVLA`: Vision-Language-Action model (ViT + BoW text conditioning)
- **Adjust parameters**: Random seed, simulation horizon, diffusion steps, stochastic noise std, language prompt.
- **Inspect rich outputs**: 2D animated trajectory paths, time-series telemetry (actions, distance-to-goal, step solve latencies), and collision audits.

### 2. ⚡ Interactive Pareto Explorer
- Explores the multi-dimensional trade-off frontier between **Inference Latency** and **Task Success Rate**.
- Interactive Plotly scatter plot with log-scale latency toggle, controller family color coding, and customizable filters (task, minimum success rate, maximum latency).
- Hover over any controller point to view exact latency, success rate, path length, collision rate, compute demands, and empirical takeaways.

### 3. 📊 Results & Ablation Viewer
- **Master Comparison Table**: Aggregated evaluation across all baselines.
- **EXP-001 GCP Component Ablation**: Tests the central thesis of Simchowitz et al. (*"Do we need diffusion in robotics?"*, 2026):
  - Iterative compute vs. single-step regression (RCP)
  - Noise injection vs. deterministic reverse sampling
  - Full reverse chain ($T=100$) vs. Minimal Iterative Policy ($T=2$)
- **High-Resolution Empirical Figures**: Interactive gallery of publication-ready figures.

### 4. 📚 About & Methodology
- Formal problem formulations, safety and constraint guarantees, hardware setup (NVIDIA DGX Spark / GB10 Grace Blackwell), and recommendations for roboticists.

---

## 🚀 Running Locally

Clone the repository and install requirements:

```bash
git clone https://github.com/Ryukijano/mpc-vla-diffusion-study.git
cd mpc-vla-diffusion-study/demo_space

pip install -r requirements.txt
python app.py
```

Open your browser at `http://localhost:7860`.

---

## 📜 Citation

If you use this benchmark or interactive arena in your research, please cite:

```bibtex
@misc{mpc_vla_diffusion_study_2026,
  title        = {MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families},
  author       = {Gyanateet and Devin AI},
  year         = {2026},
  howpublished = {\url{https://github.com/Ryukijano/mpc-vla-diffusion-study}}
}
```

---
*Developed as part of the MPC vs VLA vs Diffusion Robotics Benchmark Suite.*
