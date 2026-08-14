# Master Experiment Roadmap: Classical MPC vs VLA vs Diffusion vs World Action Models

**Study Title:** Generative Control Policies vs Optimization-Based Control vs Vision-Language-Action Models in Robotics: Mechanisms, Niches, Latency-Performance Pareto, and Real-Physics Scalability  
**Repository:** `https://github.com/Ryukijano/mpc-vla_diffusion_study`  
**Author / Lead Researcher:** Gyanateet  
**Compute Platform:** NVIDIA DGX Spark (`spark-1240`), GB10 Grace Blackwell Superchip (sm_121), 128 GB Unified LPDDR5X Memory, CUDA 12.8, PyTorch 2.12 nightly  
**Target Publication Venues:**  
- **Stage 1 (Horizon 1):** Hugging Face Community Blog + Open-Source Hub Artifact Suite (Models, Datasets, Gradio Spaces)  
- **Stage 2 (Horizon 2):** Premier Robotics/AI Conference (CoRL / NeurIPS / RSS / ICRA 2026/2027) + Open Benchmark Release (`robocontrol-bench`)  
**Status:** Canonical Master Roadmap  
**Date:** August 2026  

---

## Table of Contents

1. [Executive Summary & Core Scientific Debate](#1-executive-summary--core-scientific-debate)
2. [Two-Stage Shipping Roadmap](#2-two-stage-shipping-roadmap)
   - [2.1 Horizon 1 (Weeks 1–3): The Hugging Face Community Release](#21-horizon-1-weeks-13-the-hugging-face-community-release)
   - [2.2 Horizon 2 (Months 1–3): The Conference Paper & Benchmark Suite](#22-horizon-2-months-13-the-conference-paper--benchmark-suite)
3. [Comprehensive Experiment Matrix (EXP-001 through EXP-010)](#3-comprehensive-experiment-matrix-exp-001-through-exp-010)
   - [EXP-001: GCP Mechanism Ablation (Distribution Fitting vs Iterative Compute + Noise)](#exp-001-gcp-mechanism-ablation)
   - [EXP-002: Three-Family Head-to-Head Comparison (MPC vs VLA vs Diffusion)](#exp-002-three-family-head-to-head-comparison)
   - [EXP-003: In-Distribution vs OOD Robustness & Manifold Adherence](#exp-003-in-distribution-vs-ood-robustness--manifold-adherence)
   - [EXP-004: Latency-Performance Pareto Optimization & Step Sweeps](#exp-004-latency-performance-pareto-optimization--step-sweeps)
   - [EXP-005: World Action Models (WAM) vs Reactive Policies on Long-Horizon Tasks](#exp-005-world-action-models-wam-vs-reactive-policies)
   - [EXP-006: Sim-to-Real Visual Domain Gap & Rendering/DR Mitigation](#exp-006-sim-to-real-visual-domain-gap)
   - [EXP-007: Whole-Body Real-Robot MPC vs Learned Visuomotor Policies](#exp-007-whole-body-real-robot-mpc)
   - [EXP-008: Real-Robotic Vision Stress Test (Lighting, Occlusion, Distractors, Background)](#exp-008-real-robotic-vision-stress-test)
   - [EXP-009: Contact-Rich Multi-Stage Manipulation & Dynamic Interaction](#exp-009-contact-rich-multi-stage-manipulation)
   - [EXP-010: Multi-Modal Mode Recovery & Action Manifold Collapse Stress Test](#exp-010-multi-modal-mode-recovery)
4. [Hardware Compute Budget & Allocation Matrix (DGX Spark GB10)](#4-hardware-compute-budget--allocation-matrix-dgx-spark-gb10)
5. [Step-by-Step Execution Timeline & Dependency DAG](#5-step-by-step-execution-timeline--dependency-dag)
6. [Statistical Rigor, Stopping Rules & Protocol Governance](#6-statistical-rigor-stopping-rules--protocol-governance)
7. [Artifact Release & Hugging Face Hub Integration Plan](#7-artifact-release--hugging-face-hub-integration-plan)

---

## 1. Executive Summary & Core Scientific Debate

The robotics community is experiencing a massive paradigm shift. Three distinct control philosophies currently compete for dominance in robot manipulation and locomotion:

```
                  ┌─────────────────────────────────────────────────┐
                  │          Classical MPC (Optimization)           │
                  │  - Explicit physics, hard safety constraints    │
                  │  - Sub-millisecond QP/SQP, zero demo data       │
                  │  - Poor visual/semantic generalization           │
                  └───────────────┬─────────────────┬───────────────┘
                                  │                 │
                        Hybrid Warm-Start     Constraint-Guided
                                  │                 │
                                  ▼                 ▼
  ┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
  │   Generative Control Policies       │     │   Vision-Language-Action (VLA)      │
  │   (Diffusion / Flow / MIP)          │     │   (Autoregressive / Pretrained)     │
  │  - Trajectory distribution fitting  │◄───►│  - Internet-scale visual semantics  │
  │  - Iterative denoising & noise      │     │  - Natural language task routing    │
  │  - 10-50 Hz inference rate          │     │  - High latency (50-300 ms)         │
  └──────────────────┬──────────────────┘     └──────────────────┬──────────────────┘
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           ▼
                  ┌─────────────────────────────────────────────────┐
                  │       World Action Models (WAM / Latent Dyn)    │
                  │  - Joint future state prediction + action chunk │
                  │  - Explicit rollouts in learned latent space    │
                  │  - Contact-rich & multi-stage reasoning         │
                  └─────────────────────────────────────────────────┘
```

### The Central Controversies Under Investigation

1. **The Mechanism Controversy (Simchowitz vs Generative Orthodoxy — RQ1, RQ3):**  
   Does Diffusion Policy (Chi et al.) succeed because it models complex multi-modal distributions (score matching / generative modeling), or is the performance driven almost entirely by **iterative test-time computation and noise injection** during training (Simchowitz et al., *Much Ado About Noising*, 2025)? If a 2-step regression head with noise (Minimal Iterative Policy, MIP) matches full 100-step DDPM, the foundational premise of generative modeling in robotics must be re-evaluated.

2. **The Multi-Modality Reality Check (RQ2):**  
   Do diffusion policies actually recover distinct operational modes on true multi-modal human demonstrations (RoboMimic / LIBERO), or do they collapse onto dominant trajectories while regression policies fail only due to lack of iterative smoothing?

3. **The Generalization vs Precision Trade-Off (RQ5, RQ6, RQ8):**  
   Classical MPC dominates high-rate (>100 Hz), constraint-critical contact tasks where dynamics are known. VLAs dominate zero-shot semantic and cross-embodiment task instructions but struggle with millimetric precision and real-time control. Can hybrid frameworks (e.g., Diffusion Warm-Start MPC or DiVLA) occupy the true Pareto frontier?

4. **The Latent World Model Hypothesis (RQ9):**  
   Do World Action Models (WAMs) that explicitly roll out dynamics in latent space surpass reactive policies on long-horizon, contact-rich manipulation tasks without the extreme computational overhead of full online trajectory optimization?

5. **Sim-to-Real and Real-Vision Vulnerabilities (RQ10, RQ11, RQ12):**  
   How do these control families degrade under real-world visual perturbations (lighting, occlusions, background distractors) and dynamic loco-manipulation contacts?

This Master Experiment Roadmap provides the complete, pre-registered blueprint to systematically evaluate these questions across 10 controlled experiments, structured into two actionable shipping horizons.

---

## 2. Two-Stage Shipping Roadmap

To balance rapid scientific visibility, community contribution, and top-tier academic publication standards, the project executes across two sequential horizons:

```
═══════════════════════════════════════════════════════════════════════════════════════════════
  HORIZON 1: THE HUGGING FACE COMMUNITY PACKAGE (Weeks 1–3)
  Goal: Ship an empirically robust, artifact-backed Hugging Face Community Blog + Hub Repos
═══════════════════════════════════════════════════════════════════════════════════════════════
  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
  │    EXP-001    │     │    EXP-002    │     │    EXP-003    │     │    EXP-004    │
  │   Mechanism   │────►│  Three-Family │────►│      OOD      │────►│    Latency    │
  │   Ablation    │     │  Head-to-Head │     │  Robustness   │     │    Pareto     │
  └───────┬───────┘     └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
          │                     │                     │                     │
          ▼                     ▼                     ▼                     ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  HF Hub Artifacts: Checkpoints (SmallVLA, DDPM, Flow, MIP), Expert Demos (NPZ/  │
  │  LeRobot), Interactive Gradio Spaces (Plot Gallery + Controller Arena)          │
  └────────────────────────────────────────┬────────────────────────────────────────┘
                                           │
                                           ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  HF Community Blog: "Do We Actually Need Diffusion in Robotics? An Empirical    │
  │  Reappraisal of MPC, VLA, and Generative Control Policies"                      │
  └─────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════
  HORIZON 2: PREMIER CONFERENCE PAPER & BENCHMARK SUITE (Months 1–3)
  Goal: Submit full treatise to CoRL / NeurIPS / RSS / ICRA + Launch `robocontrol-bench`
═══════════════════════════════════════════════════════════════════════════════════════════════
  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
  │    EXP-005    │     │    EXP-006    │     │    EXP-007    │     │    EXP-008    │
  │ World Action  │────►│  Sim-to-Real  │────►│   Whole-Body  │────►│  Real-Vision  │
  │    Models     │     │  Domain Gap   │     │   Robot MPC   │     │      OOD      │
  └───────┬───────┘     └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
          │                     │                     │                     │
          ├─────────────────────┴─────────────────────┘                     │
          ▼                                                                 ▼
  ┌───────────────┐                                                 ┌───────────────┐
  │    EXP-009    │                                                 │    EXP-010    │
  │ Contact-Rich  │                                                 │  Multi-Modal  │
  │ Manipulation  │                                                 │ Mode Recovery │
  └───────┬───────┘                                                 └───────┬───────┘
          │                                                                 │
          └────────────────────────────────┬────────────────────────────────┘
                                           │
                                           ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  Premier Conference Submission (CoRL / NeurIPS Track / RSS / ICRA)              │
  │  + Public Benchmark Suite & Evaluator (`robocontrol-bench` on HF & GitHub)      │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Horizon 1 (Weeks 1–3): The Hugging Face Community Release

#### Strategic Rationale & Quality Gate Compliance
Hugging Face strictly enforces quality standards: low-quality, purely theoretical, or obviously LLM-generated articles lacking empirical depth are hidden from the front page. To ensure maximum reach and high community standing, Horizon 1 executes the core simulation experiments (EXP-001 to EXP-004) to provide undeniable empirical data, paired with public model checkpoints, expert datasets, and interactive Gradio Spaces.

#### Core Deliverables for Horizon 1
1. **Empirical Core:** Complete 5-seed runs across EXP-001 (Mechanism Ablation), EXP-002 (Three-Family Comparison), EXP-003 (OOD Degradation), and EXP-004 (Latency-Performance Pareto).
2. **Four Model Repositories on HF Hub:**
   - `Ryukijano/smallvla-mpc-vla-diffusion-quick`: Compute-matched SmallVLA (~86M params, ViT backbone + action regression head, ~340 MB checkpoint).
   - `Ryukijano/diffusion-policy-mpc-vla-diffusion-quick`: 1D Temporal UNet DDPM policy (~5.6 MB quick, ~641 MB full).
   - `Ryukijano/flow-matching-policy-quick`: Rectified Flow Matching controller with ODE velocity field.
   - `Ryukijano/mip-policy-quick`: Minimal Iterative Policy (2-step regression with noise injection).
3. **Open Demonstration Dataset:**
   - `Ryukijano/mpc-expert-demos-quick-test`: Multi-modal state and RGB image trajectories generated by Collision-Free MPC on 2D Reaching and PushT, formatted in NumPy `.npz` and LeRobot v2 Parquet schema.
4. **Two Interactive Gradio Spaces:**
   - `Ryukijano/mpc-vla-diffusion-plot-gallery`: Interactive exploration of Pareto frontiers, degradation curves, and trajectory rollouts.
   - `Ryukijano/mpc-vla-diffusion-arena`: Live browser-based comparison tool allowing users to select tasks, tweak controller parameters, and inspect action rollouts in real time.
5. **HF Collection:** `Ryukijano/mpc-vla-diffusion-study` grouping all artifacts.
6. **Canonical Blog Post:** Drafted in Markdown with interactive widgets, clear citations, and reproducible CLI instructions.

---

### 2.2 Horizon 2 (Months 1–3): The Conference Paper & Benchmark Suite

#### Strategic Rationale
While Horizon 1 establishes the mechanism baseline on canonical 2D/planar tasks (PushT, Reaching), a premier robotics conference paper (CoRL / NeurIPS / RSS / ICRA) requires testing under full 3D physics, contact discontinuities, language conditioning, high-DoF whole-body dynamics, and photorealistic vision. Horizon 2 transitions to ManiSkill3, robosuite, Isaac Sim, and MuJoCo Menagerie quadruped/humanoid dynamics.

#### Core Deliverables for Horizon 2
1. **Empirical Core:** EXP-005 (World Action Models), EXP-006 (Sim-to-Real Transfer), EXP-007 (Whole-Body MPC), EXP-008 (Real-Robotic Vision OOD), EXP-009 (Contact-Rich Manipulation), and EXP-010 (Multi-Modal Mode Recovery).
2. **Benchmark Release (`robocontrol-bench`):** A standardized, pip-installable benchmark suite containing unified wrappers for MPC (OSQP/CasADi/Pinocchio), VLA (OpenVLA/Octo), Diffusion (DDPM/Flow), and WAM controllers across ManiSkill3, robosuite, and MuJoCo.
3. **Full-Scale Model Zoo:** Checkpoints for OpenVLA fine-tunes (7B INT8/FP8), 100M-parameter WAMs, Full-Order Quadruped MPC solvers, and 3D Diffusion Policies.
4. **Conference Manuscript:** An 8-page paper titled *"Do We Need Diffusion in Robotics? An Empirical Reappraisal of Optimization, Foundation Models, and Generative Policies across Manipulation and Locomotion"*.

---

## 3. Comprehensive Experiment Matrix (EXP-001 through EXP-010)

Below is the complete specification for each experiment in the study, linking research questions, benchmark tasks, conditions, sample sizes, compute requirements, falsification criteria, and release artifacts.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RESEARCH QUESTIONS INDEX                                       │
├────────────────────────────────┬────────────────────────────────┬────────────────────────────────┤
│ RQ1: Primary GCP Advantage     │ RQ5: Classical MPC Niche       │ RQ9: World Action Models       │
│ RQ2: Multi-Modality Recovery   │ RQ6: VLA Niche & Latency       │ RQ10: Sim-to-Real Domain Gap   │
│ RQ3: Mechanism (Iter vs Dist)  │ RQ7: Hybrid Architecture Wins  │ RQ11: Real-Robot Dynamic MPC   │
│ RQ4: OOD Manifold Adherence    │ RQ8: Latency-Performance Pareto│ RQ12: Real-Robotic Vision OOD  │
└────────────────────────────────┴────────────────────────────────┴────────────────────────────────┘
```

---

### EXP-001: GCP Mechanism Ablation

* **Exact Purpose & Research Hypothesis:**  
  Tests **RQ1** and **RQ3** (H3a vs H3b). Resolves whether the success of Generative Control Policies stems from multi-modal distribution fitting (score matching / generative loss) or simply from **iterative compute + noise injection** during training (Simchowitz hypothesis).
  - *H3a (Generative Orthodoxy):* Distribution fitting is essential; removing it causes severe performance drops.
  - *H3b (Simchowitz Debunking):* Iterative compute + noise is the sole necessary mechanism; Minimal Iterative Policy (MIP, 2-step regression + noise) matches full DDPM ($T=100$).
* **Benchmark Tasks:**  
  - PushT (planar multi-modal pushing, state + image obs, horizon 15, max steps 300).  
  - 2D Reaching (uni-modal point-mass reaching, 4D state, max steps 60).
* **Conditions / Baselines Compared (5 compute-matched conditions):**  
  1. `C1: Full DDPM` ($T=100$, score matching, UNet1D backbone).  
  2. `C2: DDPM No-Noise` ($T=100$, score matching, deterministic sampling).  
  3. `C3: DDPM Single-Step` ($T=1$, score matching with 1 step, removes iterative compute).  
  4. `C4: MIP` (2-step regression head + noise injection during training, no generative loss).  
  5. `C5: Pure Regression` ($T=1$, standard MSE loss, single-step, no noise floor).
* **Metrics & Sample Sizes:**  
  - *Metrics:* Task Success Rate (%), Inference Latency (ms), Mode Coverage (k-means cluster hit %), Action KL Divergence.  
  - *Sample Size:* 100 evaluation episodes $\times$ 5 random seeds (`[0, 1, 2, 42, 123]`) $\times$ 5 conditions $\times$ 2 benchmarks = **5,000 episodes**.  
  - *Latency Harness:* 100 warmup calls, 1,000 timed inferences on GB10.
* **Compute Budget Estimate (DGX Spark GB10):**  
  - *Training:* 5 conditions $\times$ 5 seeds $\times$ 2 benchmarks $\times$ 100 epochs $\approx$ **2.5 hours**.  
  - *Evaluation & Latency:* 5,000 episodes + 10k timing calls $\approx$ **0.5 hours**.  
  - *Total Wall-Clock:* **~3.0 hours** (Quick smoke test: ~10 minutes).
* **Falsification Criteria:**  
  - *H3b falsified if:* $\text{Success}(\text{Full DDPM}) - \text{Success}(\text{MIP}) > 3.0\text{ pp}$ with Bonferroni-corrected Wilcoxon signed-rank $p < 0.05$ on **both** benchmarks.  
  - *H3a falsified if:* $\text{Success}(\text{MIP}) \ge \text{Success}(\text{Full DDPM}) - 3.0\text{ pp}$ on both benchmarks ($p \ge 0.05$).  
  - *Replication Gate:* Full DDPM PushT success must match Simchowitz et al. reported baseline within $\pm 5.0\text{ pp}$.
* **Artifacts Generated:**  
  - Model Checkpoint: `Ryukijano/diffusion-policy-mpc-vla-diffusion-quick`  
  - Model Checkpoint: `Ryukijano/mip-policy-quick`  
  - Analysis Output: `experiments/EXP-001-mechanism-ablation/outputs/analysis/comparison_table.csv`, `mechanism_ablation.pdf`.

---

### EXP-002: Three-Family Head-to-Head Comparison

* **Exact Purpose & Research Hypothesis:**  
  Tests **RQ5** (H5, MPC niche), **RQ6** (H6, VLA niche), **RQ7** (H7, hybrid superiority), and **RQ8** (Pareto placement). Establishes head-to-head empirical boundaries across Classical MPC, VLA, Diffusion, and Hybrid controllers across tasks spanning distinct difficulty zones.
* **Benchmark Tasks:**  
  - 2D Reaching (Clear) — Uni-modal unconstrained state-based control.  
  - 2D Reaching (Cluttered) — Multi-obstacle non-convex navigation with hard safety bounds.  
  - PushT (Image + Language conditioned) — Semantic visual manipulation.
* **Conditions / Baselines Compared (8 conditions):**  
  1. `C1: Linear MPC` (OSQP quadratic program solver, linearized dynamics).  
  2. `C2: Nonlinear MPC` (CasADi / iLQR interior-point solver).  
  3. `C3: Collision-Free MPC` (Signed Distance Field obstacle constraints).  
  4. `C4: SmallVLA` (Pretrained ViT + language token projection + action regression head).  
  5. `C5: DDPM Policy` ($T=100$, Conditional UNet1D).  
  6. `C6: Flow Matching Policy` (Rectified flow vector field, Euler solver).  
  7. `C7: MIP` (Minimal Iterative Policy from EXP-001).  
  8. `C8: Diffusion Warm-Start MPC` (Diffusion trajectory initial guess + SQP refinement).
* **Metrics & Sample Sizes:**  
  - *Metrics:* Success Rate (%), Median & P95 Latency (ms), Trajectory Path Length, Collision Rate (%), Constraint Violation Rate (%).  
  - *Sample Size:* 100 episodes $\times$ 5 seeds $\times$ 8 conditions $\times$ 3 benchmarks = **12,000 episodes**.
* **Compute Budget Estimate (DGX Spark GB10):**  
  - *Training (C4-C8):* 5 learning methods $\times$ 5 seeds $\times$ 3 benchmarks $\approx$ **4.5 hours**.  
  - *Rollouts & MPC Optimization:* 12,000 episodes $\approx$ **1.5 hours**.  
  - *Total Wall-Clock:* **~6.0 hours** (Quick smoke test: ~15 minutes).
* **Falsification Criteria:**  
  - *H5 (MPC Niche) falsified if:* Collision-Free MPC fails to beat DDPM and SmallVLA on Cluttered Reaching by $\ge 5.0\text{ pp}$ ($p < 0.05$).  
  - *H6 (VLA Niche) falsified if:* SmallVLA does not lead on language-conditioned PushT by $\ge 5.0\text{ pp}$ over pure DDPM.  
  - *H7 (Hybrid Win) falsified if:* Diffusion Warm-Start MPC does not achieve $\text{Success} > \max(\text{Collision-Free MPC}, \text{DDPM}) + 3.0\text{ pp}$ on Cluttered Reaching.
* **Artifacts Generated:**  
  - Model Checkpoints: `Ryukijano/smallvla-mpc-vla-diffusion-quick`, `Ryukijano/flow-matching-policy-quick`  
  - Dataset: `Ryukijano/mpc-expert-demos-quick-test`  
  - Analysis Output: `experiments/EXP-002-family-comparison/outputs/analysis/comparison_table.csv`, radar performance plots.

---

### EXP-003: In-Distribution vs OOD Robustness & Manifold Adherence

* **Exact Purpose & Research Hypothesis:**  
  Tests **RQ4** (H4). Evaluates whether iterative compute and noise injection act as implicit regularizers preventing out-of-distribution observation collapse by maintaining *action manifold adherence*.
* **Benchmark Tasks:**  
  - PushT (Image observation; training strictly on clean in-distribution frames; evaluation under 5 controlled visual/spatial perturbation levels).
* **Conditions / Baselines Compared (4 conditions $\times$ 5 perturbation levels):**  
  - *Controllers:* `C1: Pure Regression`, `C2: MIP (T=2)`, `C3: Full DDPM (T=100)`, `C4: Flow Matching (T=10)`.  
  - *Perturbation Levels:*  
    - Level 0: In-Distribution (Clean baseline).  
    - Level 1: Object Color / Texture jitter ($\pm 40\%$).  
    - Level 2: Initial Object Spatial Translation offset ($\pm 5\text{ cm}$).  
    - Level 3: Camera Viewpoint / Perspective rotation ($\pm 10^\circ$).  
    - Level 4: Combined Worst-Case (Levels 1 + 2 + 3 simultaneous).
* **Metrics & Sample Sizes:**  
  - *Metrics:* Success Rate per Level (%), Degradation Slope ($\Delta\text{Success}/\Delta\text{Level}$), Action Manifold Adherence Score (mean 5-NN distance to expert action manifold: lower = superior adherence).  
  - *Sample Size:* 100 episodes $\times$ 5 seeds $\times$ 4 conditions $\times$ 5 levels = **10,000 episodes**.
* **Compute Budget Estimate (DGX Spark GB10):**  
  - Reuses trained checkpoints from EXP-001/002.  
  - *Evaluation Rollouts:* 10,000 episodes with perturbed rendering $\approx$ **2.0 hours**.  
  - *Total Wall-Clock:* **~2.0 hours** (Quick smoke test: ~5 minutes).
* **Falsification Criteria:**  
  - *H4 falsified if:* Two-way ANOVA (Condition $\times$ Level) fails to show a significant interaction ($p \ge 0.05$), OR if Regression degradation slope is less steep than MIP/DDPM, OR if MIP fails to maintain manifold adherence within 10% of Full DDPM at Level 4.
* **Artifacts Generated:**  
  - Analysis Output: `experiments/EXP-003-ood-robustness/outputs/analysis/anova_results.json`, `degradation_slopes.csv`, `manifold_adherence_curves.pdf`.

---

### EXP-004: Latency-Performance Pareto Optimization & Step Sweeps

* **Exact Purpose & Research Hypothesis:**  
  Tests **RQ8** (H8). Maps the empirical Pareto frontier across control families in the 2D plane of $\text{Latency (ms)}$ versus $\text{Task Success Rate (\%)}$, identifying dominated architectures and step-efficiency limits.
* **Benchmark Tasks:**  
  - PushT and 2D Reaching.
* **Conditions / Baselines Compared (21 distinct configurations):**  
  - *Classical MPC:* Linear MPC (OSQP), Nonlinear MPC (iLQR).  
  - *DDPM Step Sweep:* $T \in \{1, 2, 4, 8, 16, 32, 64, 100\}$.  
  - *Flow Matching Step Sweep:* $T \in \{1, 2, 4, 8, 10\}$.  
  - *MIP Iteration Sweep:* $\text{Iterations} \in \{1, 2, 3, 5\}$.  
  - *Single-Step RCP:* Pure Regression ($T=1$).  
  - *VLA:* SmallVLA (single forward pass).
* **Metrics & Sample Sizes:**  
  - *Metrics:* Median and P95 Latency (ms), Task Success Rate (%), Pareto Dominance Count, Marginal Success Gain per Step ($\Delta\text{Success}/\Delta T$).  
  - *Sample Size:* 100 episodes $\times$ 5 seeds $\times$ 21 conditions $\times$ 2 benchmarks = **21,000 episodes**; plus 1,000 timed latency calls $\times$ 5 seeds $\times$ 21 conditions $\times$ 2 benchmarks = **210,000 timing measurements**.
* **Compute Budget Estimate (DGX Spark GB10):**  
  - *Training:* Reuses models from EXP-001/002 (only inference $T$ is varied).  
  - *Rollouts & High-Precision Timing:* 21,000 rollouts + 210,000 timed forward calls on locked GPU clocks $\approx$ **5.0 hours**.  
  - *Total Wall-Clock:* **~5.0 hours** (Quick smoke test: ~15 minutes).
* **Falsification Criteria:**  
  - *H8 falsified if:* Fewer than 2 distinct controller families lie on the non-dominated Pareto frontier, OR if MIP ($T=2$) is Pareto-dominated on both benchmarks, OR if Flow Matching ($T=4$) fails to match DDPM ($T=32$) within $3.0\text{ pp}$ at $\le 50\%$ latency.
* **Artifacts Generated:**  
  - Interactive Space Asset: `Ryukijano/mpc-vla-diffusion-plot-gallery`  
  - Interactive Space Asset: `Ryukijano/mpc-vla-diffusion-arena`  
  - Analysis Output: `pareto_frontier.csv`, `pareto_pusht.pdf`, `pareto_reaching.pdf`, `step_saturation_curves.pdf`.

---

### EXP-005: World Action Models (WAM) vs Reactive Policies on Long-Horizon Tasks

* **Exact Purpose & Research Hypothesis:**  
  Tests **RQ9** (H9, H9b). Evaluates whether a World Action Model (jointly predicting future latent dynamics and outputting action sequences) provides a decisive advantage over reactive VLAs, diffusion policies, and MPC on multi-stage, contact-rich manipulation.
* **Benchmark Tasks:**  
  - robosuite: `Lift` (single-stage precision), `Square` (multi-stage contact-rich peg-in-hole).  
  - ManiSkill3: `PickCube` (single-stage rigid), `StackCube` (long-horizon contact assembly).
* **Conditions / Baselines Compared (5 compute-matched conditions):**  
  1. `C1: WAM + Action Policy` (Latent dynamics rollout $\le 8$ steps + action decoder, $\le 100\text{M}$ params).  
  2. `C2: SmallVLA` (Compute-matched VLM backbone + autoregressive/regression chunking).  
  3. `C3: DDPM Policy` ($T=50$, compute-matched backbone).  
  4. `C4: MIP` (2-step regression + noise).  
  5. `C5: Nonlinear MPC` (Full analytical Franka dynamics with obstacle/contact constraints).
* **Metrics & Sample Sizes:**  
  - *Metrics:* End-to-End Success Rate (%), Sub-Goal Stage Completion Rate (Reach, Grasp, Align, Insert/Stack %), Latency per Step, Contact Violation Rate.  
  - *Sample Size:* 100 episodes $\times$ 5 seeds $\times$ 5 conditions $\times$ 4 benchmarks = **10,000 episodes**.
* **Compute Budget Estimate (DGX Spark GB10):**  
  - *Training (C1-C4):* 4 methods $\times$ 5 seeds $\times$ 4 tasks $\times$ 150 epochs on SAPIEN/robosuite $\approx$ **14.0 hours**.  
  - *Evaluation Rollouts:* 10,000 episodes with GPU physics $\approx$ **2.0 hours**.  
  - *Total Wall-Clock:* **~16.0 hours** (Quick smoke test: ~30 minutes).
* **Falsification Criteria:**  
  - *H9 falsified if:* WAM fails to beat SmallVLA by $\ge 5.0\text{ pp}$ and MIP by $\ge 3.0\text{ pp}$ on `Square` and `StackCube` ($p < 0.05$), OR if WAM rollout ablation (disabling future latent loss) matches full WAM within $1.0\text{ pp}$.
* **Artifacts Generated:**  
  - Model Checkpoint: `Ryukijano/world-action-model-franka-100m`  
  - Analysis Output: `experiments/EXP-005-world-models/outputs/analysis/subgoal_completion.csv`, `wam_long_horizon_breakdown.pdf`.

---

### EXP-006: Sim-to-Real Visual Domain Gap & Rendering/DR Mitigation

* **Exact Purpose & Research Hypothesis:**  
  Tests **RQ10** (H10, H10b). Quantifies the visual transfer gap between synthetic simulation and real-world image distributions, testing whether Domain Randomization (DR), RTX Photorealistic PBR rendering, or Synthetic-to-Real Image Adaptation closes the gap for learned policies.
* **Benchmark Tasks:**  
  - ManiSkill3 / Isaac Sim: `PickCube` and `PushCube` evaluated across Paired Source Sim vs Real/Synthetic-Real Target Test Splits (using `VLA-REPLICA` / `RoboDojo-Real` image backgrounds and lighting).
* **Conditions / Baselines Compared (5 training pipelines):**  
  1. `C1: Sim-Only Baseline` (Default rasterizer, no DR).  
  2. `C2: Domain Randomization (DR)` (Randomized lighting, textures, background, camera extrinsics).  
  3. `C3: Photorealistic PBR` (NVIDIA RTX / SAPIEN ray-traced materials + subtle DR).  
  4. `C4: DR + Synthetic-to-Real Transfer` (CycleGAN / AdaIN feature alignment layer).  
  5. `C5: Real-Pretrained VLA` (OpenVLA-7B pretrained on Open X-Embodiment zero-shot / fine-tuned).
* **Metrics & Sample Sizes:**  
  - *Metrics:* Sim-to-Real Success Gap ($\Delta = \text{Success}_{\text{sim}} - \text{Success}_{\text{real}}$, in pp), Target Domain Success Rate (%), Fréchet Inception Distance (FID), Perceptual LPIPS distance.  
  - *Sample Size:* 100 episodes $\times$ 5 seeds $\times$ 5 conditions $\times$ 2 domains (Sim vs Real) $\times$ 2 tasks = **10,000 episodes**.
* **Compute Budget Estimate (DGX Spark GB10):**  
  - *Training & Ray-Tracing:* 5 conditions $\times$ 5 seeds $\times$ 2 tasks $\approx$ **12.0 hours**.  
  - *Evaluation:* 10,000 paired rollouts $\approx$ **2.0 hours**.  
  - *Total Wall-Clock:* **~14.0 hours** (Quick smoke test: ~25 minutes).
* **Falsification Criteria:**  
  - *H10 falsified if:* Neither Photorealistic PBR (C3) nor DR+Transfer (C4) reduces the Sim-to-Real gap by $\ge 8.0\text{ pp}$ relative to C1 ($p < 0.05$).
* **Artifacts Generated:**  
  - Dataset Split: `Ryukijano/sim-to-real-transfer-benchmark`  
  - Analysis Output: `sim_to_real_gaps.csv`, `fid_vs_gap_correlation.pdf`.

---

### EXP-007: Whole-Body Real-Robot MPC vs Learned Visuomotor Policies

* **Exact Purpose & Research Hypothesis:**  
  Tests **RQ11** (H11). Investigates whether optimization-based Whole-Body MPC maintains unassailable superiority over learned policies on dynamic, high-rate legged locomotion and loco-manipulation where contact dynamics and torque constraints dictate physical stability.
* **Benchmark Tasks:**  
  - MuJoCo Menagerie Unitree Go2 Quadruped & Humanoid-v4 / RoboDojo:  
    1. Dynamic Walk / Trotting over uneven terrain.  
    2. Whole-Body Stand & Balance under external impulse perturbations ($10\text{ N}\cdot\text{s}$).  
    3. Mobile Manipulation Reach (quadruped-mounted arm reaching target while maintaining base stability).
* **Conditions / Baselines Compared (4 controller paradigms):**  
  1. `C1: Centroidal / Single-Rigid-Body (SRB) MPC` (Simplified CoM dynamics, convex QP @ 100 Hz).  
  2. `C2: Full-Order Nonlinear Whole-Body MPC` (Pinocchio/CasADi multi-contact SQP @ 25 Hz).  
  3. `C3: Diffusion Warm-Start Whole-Body MPC` (Diffusion trajectory initialization + 2-iteration SQP @ 50 Hz).  
  4. `C4: End-to-End Learned Visuomotor Policy` (Diffusion Policy trained on expert MPC rollouts @ 20 Hz).
* **Metrics & Sample Sizes:**  
  - *Metrics:* Task Success Rate (%), Fall / Loss-of-Balance Rate (%), Mean Control Frequency (Hz), Joint/Torque Constraint Violation Rate (%), Cumulative Mechanical Energy ($J$).  
  - *Sample Size:* 100 episodes $\times$ 5 seeds $\times$ 4 conditions $\times$ 3 tasks = **6,000 episodes**.
* **Compute Budget Estimate (DGX Spark GB10):**  
  - *Policy Training (C4 & C3 Prior):* 5 seeds $\times$ 3 tasks $\approx$ **4.0 hours**.  
  - *Physics & Non-Convex MPC Rollouts:* 6,000 episodes $\times$ 500 steps @ high sub-stepping $\approx$ **4.0 hours**.  
  - *Total Wall-Clock:* **~8.0 hours** (Quick smoke test: ~20 minutes).
* **Falsification Criteria:**  
  - *H11 falsified if:* Learned Policy (C4) achieves equal or lower fall rate than Full-Order MPC (C2) under impulse perturbations, OR if Diffusion Warm-Start MPC (C3) fails to run at $\ge 30\text{ Hz}$ while matching C2 success within $3.0\text{ pp}$.
* **Artifacts Generated:**  
  - Code & Solvers: `src/real_robotics/whole_body_mpc/`  
  - Video Rollouts & Analysis: `fall_rate_comparison.pdf`, `control_frequency_pareto.pdf`.

---

### EXP-008: Real-Robotic Vision Stress Test

* **Exact Purpose & Research Hypothesis:**  
  Tests **RQ12** (H12). Stresses vision-conditioned controllers (SmallVLA, DDPM, MIP, WAM) against realistic, complex 3D visual corruptions (dynamic lighting shifts, physical occlusions, visual clutter/distractors, and unmodeled background swaps).
* **Benchmark Tasks:**  
  - ManiSkill3 `PickCube` and robosuite `Lift` rendered with realistic physical camera optics.
* **Conditions / Baselines Compared (4 controllers $\times$ 5 visual corruption environments):**  
  - *Controllers:* `C1: SmallVLA`, `C2: DDPM Policy`, `C3: MIP (T=2)`, `C4: World Action Model (WAM)`.  
  - *Corruption Environments:*  
    - Level 0: Clean RGB (ID Reference).  
    - Level 1: Extreme Lighting Fluctuations (intensity $\pm 50\%$, dynamic shadow casting).  
    - Level 2: Visual Occlusions ($1-3$ random volumetric occluders masking up to $35\%$ of workspace).  
    - Level 3: Scene Clutter / Distractors ($1-5$ unmodeled household objects placed around target).  
    - Level 4: Background Swap (Photorealistic real-room HDRIs replacing clean sim backdrop).
* **Metrics & Sample Sizes:**  
  - *Metrics:* Success Rate per Corruption (%), Manifold Adherence Score (5-NN distance to clean expert actions), Success Drop $\Delta_{\text{drop}} = \text{Success}_0 - \text{Success}_k$.  
  - *Sample Size:* 100 episodes $\times$ 5 seeds $\times$ 4 conditions $\times$ 5 levels $\times$ 2 tasks = **20,000 episodes** (Quick evaluation option: 50 episodes $\times$ 5 seeds = 5,000 episodes).
* **Compute Budget Estimate (DGX Spark GB10):**  
  - Reuses trained checkpoints from EXP-005.  
  - *Evaluation Rollouts:* 20,000 episodes $\approx$ **5.0 hours**.  
  - *Total Wall-Clock:* **~5.0 hours** (Quick option: ~1.5 hours).
* **Falsification Criteria:**  
  - *H12 falsified if:* SmallVLA does not exhibit the steepest degradation slope across occlusions and distractors, OR if WAM fails to match or exceed DDPM/MIP manifold adherence under Level 4 background swaps ($p < 0.05$).
* **Artifacts Generated:**  
  - Analysis Output: `experiments/EXP-008-real-vision/outputs/analysis/perturbation_heatmap.pdf`, `manifold_degradation.csv`.

---

### EXP-009: Contact-Rich Multi-Stage Manipulation

* **Exact Purpose & Research Hypothesis:**  
  Tests **RQ9** and **RQ5** on high-dimensional physical interaction. Tests the hypothesis ($H_{\text{CONTACT}}$) that classical MPC with exact contact models or WAMs with predictive latent rollouts outperform purely reactive diffusion and VLA policies on tasks dominated by contact mode switches, friction transitions, and jamming avoidance.
* **Benchmark Tasks:**  
  - robosuite: `NutAssembly` (high-precision threading with contact jamming) and `Can` (tight-tolerance pick-and-place).  
  - ManiSkill3: `TurnFaucet` (articulated contact mechanism) and `OpenCabinet` (multi-body door pull).
* **Conditions / Baselines Compared (5 conditions):**  
  1. `C1: Contact-Aware Nonlinear MPC` (SDF contact formulation with friction cone constraints).  
  2. `C2: SmallVLA` (Pretrained VLM + continuous action regression).  
  3. `C3: Full DDPM Policy` ($T=100$).  
  4. `C4: MIP` (2-step iterative regression + noise).  
  5. `C5: World Action Model (WAM)` (Predictive latent contact model).
* **Metrics & Sample Sizes:**  
  - *Metrics:* Assembly Success Rate (%), Contact Jamming / Force Spike Count ($>20\text{ N}$), Completion Time (s), Cumulative Contact Impulse ($\int \|F_c\|\,dt$).  
  - *Sample Size:* 100 episodes $\times$ 5 seeds $\times$ 5 conditions $\times$ 4 benchmarks = **10,000 episodes**.
* **Compute Budget Estimate (DGX Spark GB10):**  
  - *Training (C2-C5):* 4 methods $\times$ 5 seeds $\times$ 4 benchmarks $\approx$ **14.0 hours**.  
  - *Physics Rollouts:* 10,000 episodes @ high contact resolution $\approx$ **3.0 hours**.  
  - *Total Wall-Clock:* **~17.0 hours** (Quick smoke test: ~35 minutes).
* **Falsification Criteria:**  
  - *$H_{\text{CONTACT}}$ falsified if:* Pure DDPM or SmallVLA achieves lower jamming rates and higher assembly success than both Contact-Aware MPC and WAM on `NutAssembly`.
* **Artifacts Generated:**  
  - Model Checkpoints: `Ryukijano/contact-rich-wam-manipulation`  
  - Analysis Output: `contact_jamming_analysis.csv`, `force_profile_comparison.pdf`.

---

### EXP-010: Multi-Modal Mode Recovery & Action Manifold Collapse Stress Test

* **Exact Purpose & Research Hypothesis:**  
  Directly tests **RQ2** (H2a vs H2b). Evaluates whether diffusion policies truly recover and sample from multiple distinct human demonstration modes when presented with an explicitly multi-modal dataset, or whether they suffer from mode averaging / mode collapse similar to regression baselines.
* **Benchmark Tasks:**  
  - RoboMimic Multi-Modal Demonstration Splits:  
    1. `Lift-MultiModal` (Two distinct grasping strategies: top grasp vs side grasp).  
    2. `Can-MultiModal` (Left-hand path vs Right-hand path obstacle avoidance).  
    3. `Square-MultiModal` (Clockwise vs Counter-Clockwise insertion trajectories).  
  - Synthetic Bi-Modal 2D Navigation (Exact analytical ground-truth bi-modal distribution).
* **Conditions / Baselines Compared (6 conditions):**  
  1. `C1: Pure Regression (RCP)` (MSE loss — predicted to produce mode averaging / collision).  
  2. `C2: Mixture Density Network (MDN)` (Parametric multi-modal baseline, $K=5$ Gaussian mixture).  
  3. `C3: Full DDPM Policy` ($T=100$, score matching).  
  4. `C4: Flow Matching Policy` (Rectified flow matching with optimal transport).  
  5. `C5: MIP (T=2)` (Simchowitz minimal iterative regression).  
  6. `C6: SmallVLA` (Autoregressive tokenized action head).
* **Metrics & Sample Sizes:**  
  - *Metrics:* Mode Recovery Entropy ($H_{\text{modes}} = -\sum p_k \log p_k$), Mode Coverage % (percentage of dataset modes visited $\ge 10\%$ of rollouts), In-Between Mode Collapse Rate (% of trajectories falling into invalid intermediate space / obstacle zone), Wasserstein-1 Distance to Expert Distribution.  
  - *Sample Size:* 200 evaluation rollouts per seed $\times$ 5 seeds $\times$ 6 conditions $\times$ 4 benchmarks = **24,000 episodes**.
* **Compute Budget Estimate (DGX Spark GB10):**  
  - *Training:* 6 conditions $\times$ 5 seeds $\times$ 4 tasks $\approx$ **10.0 hours**.  
  - *Rollouts & Density Estimation:* 24,000 episodes $\approx$ **3.0 hours**.  
  - *Total Wall-Clock:* **~13.0 hours** (Quick smoke test: ~30 minutes).
* **Falsification Criteria:**  
  - *H2b (Simchowitz) falsified if:* Full DDPM (C3) and Flow Matching (C4) achieve Mode Coverage $\ge 85\%$ across all benchmarks while MIP (C5) and Regression (C1) achieve $\le 50\%$ with mode collapse into obstacle space ($p < 0.001$).  
  - *H2a (Pro-Diffusion) falsified if:* MIP matches DDPM mode coverage within $\pm 5.0\text{ pp}$ across all multi-modal tasks.
* **Artifacts Generated:**  
  - Analysis Output: `mode_recovery_distribution.csv`, `trajectory_density_kde_plots.pdf`, `mode_collapse_breakdown.pdf`.

---

## 4. Hardware Compute Budget & Allocation Matrix (DGX Spark GB10)

All experiments are engineered and optimized specifically for the NVIDIA DGX Spark workstation (`spark-1240`) equipped with the **GB10 Grace Blackwell Superchip (sm_121)** and **128 GB Unified LPDDR5X Memory**.

### Master Resource Allocation Breakdown

| Exp ID | Experiment Focus | Total Rollout Episodes | Learning Training Hours | Eval / Timing Hours | Total GB10 Wall-Clock | Quick Smoke Test Time | Horizon |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP-001** | Mechanism Ablation (MIP vs DDPM) | 5,000 | 2.5 h | 0.5 h | **3.0 h** | 10 min | Horizon 1 |
| **EXP-002** | Three-Family Head-to-Head | 12,000 | 4.5 h | 1.5 h | **6.0 h** | 15 min | Horizon 1 |
| **EXP-003** | OOD Perturbation Degradation | 10,000 | 0.0 h *(Reused)* | 2.0 h | **2.0 h** | 5 min | Horizon 1 |
| **EXP-004** | Latency-Performance Pareto | 21,000 *(+210k tim)* | 0.0 h *(Reused)* | 5.0 h | **5.0 h** | 15 min | Horizon 1 |
| **EXP-005** | World Action Models (WAM) | 10,000 | 14.0 h | 2.0 h | **16.0 h** | 30 min | Horizon 2 |
| **EXP-006** | Sim-to-Real Domain Gap | 10,000 | 12.0 h | 2.0 h | **14.0 h** | 25 min | Horizon 2 |
| **EXP-007** | Whole-Body Real-Robot MPC | 6,000 | 4.0 h | 4.0 h | **8.0 h** | 20 min | Horizon 2 |
| **EXP-008** | Real-Robotic Vision Stress Test | 20,000 | 0.0 h *(Reused)* | 5.0 h | **5.0 h** | 25 min | Horizon 2 |
| **EXP-009** | Contact-Rich Manipulation | 10,000 | 14.0 h | 3.0 h | **17.0 h** | 35 min | Horizon 2 |
| **EXP-010** | Multi-Modal Mode Recovery | 24,000 | 10.0 h | 3.0 h | **13.0 h** | 30 min | Horizon 2 |
| **TOTALS** | **Full 10-Experiment Package** | **118,000 eps** | **61.0 h** | **28.0 h** | **89.0 h (~3.7 days)** | **~3.5 h** | **H1 + H2** |

```
Summary Statistics:
- Horizon 1 Subtotal (EXP-001 to EXP-004): 48,000 episodes | 16.0 GB10 compute hours
- Horizon 2 Subtotal (EXP-005 to EXP-010): 70,000 episodes | 73.0 GB10 compute hours
- Full Project Grand Total: 118,000 rollouts + 210,000 latency measurements | 89.0 compute hours
```

### Compute Optimization & Memory Management on GB10
1. **Unified Memory Utilization:** Utilizing the 128 GB unified memory enables zero-copy tensor passing between CPU (OSQP, CasADi, Pinocchio kinematics) and GPU (PyTorch diffusion backbones, ViT encoders) without host-to-device PCIe bottlenecks.
2. **CUDA 12.8 & PyTorch 2.12 Nightly Features:** Utilizing `torch.compile(mode="max-autotune")` with Triton FlashAttention kernels for VLA vision backbones and 1D temporal convolutions, yielding an estimated $1.8\times$ speedup over eager mode.
3. **Fixed GPU Clock Gating:** All latency timing sweeps in EXP-004 and EXP-007 lock the GPU base clock (`nvidia-smi -lgc 1800,1800`) to eliminate dynamic thermal throttling variance.

---

## 5. Step-by-Step Execution Timeline & Dependency DAG

### Experiment Dependency Graph (DAG)

```
                            ┌──────────────────────────────────────┐
                            │ EXP-001: GCP Mechanism Ablation      │
                            │ (MIP vs DDPM vs Regression Baseline) │
                            └──────────────────┬───────────────────┘
                                               │ [Replication Gate Passed]
                     ┌─────────────────────────┼─────────────────────────┐
                     │                         │                         │
                     ▼                         ▼                         ▼
         ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
         │ EXP-002: Three-Family │ │ EXP-003: OOD          │ │ EXP-004: Latency-     │
         │ Head-to-Head Compare  │ │ Robustness Degradation│ │ Performance Pareto    │
         └───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
                     │                         │                         │
                     └─────────────────────────┼─────────────────────────┘
                                               │
                                               ▼
                     ═════════════════════════════════════════════════════
                       MILESTONE 1: HORIZON 1 RELEASE (End of Week 3)
                       - Hugging Face Community Blog Published
                       - Hub Artifacts (4 Models, 1 Dataset, 2 Spaces)
                     ═════════════════════════════════════════════════════
                                               │
                     ┌─────────────────────────┴─────────────────────────┐
                     │                                                   │
                     ▼                                                   ▼
         ┌───────────────────────┐                           ┌───────────────────────┐
         │ EXP-005: World Action │                           │ EXP-010: Multi-Modal  │
         │ Models (ManiSkill3)   │                           │ Mode Recovery Stress  │
         └───────────┬───────────┘                           └───────────┬───────────┘
                     │                                                   │
         ┌───────────┴───────────┬───────────────────┐                   │
         │                       │                   │                   │
         ▼                       ▼                   ▼                   │
┌─────────────────┐     ┌─────────────────┐ ┌─────────────────┐          │
│ EXP-006: Sim-to-│     │ EXP-007: Whole- │ │ EXP-008: Real-  │          │
│ Real Transfer   │     │ Body Robot MPC  │ │ Vision Stress   │          │
└────────┬────────┘     └────────┬────────┘ └────────┬────────┘          │
         │                       │                   │                   │
         └───────────────────────┼───────────────────┴───────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ EXP-009: Contact-Rich │
                     │ Dynamic Manipulation  │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ═════════════════════════════════════════════════════
                       MILESTONE 2: HORIZON 2 RELEASE (End of Month 3)
                       - CoRL / NeurIPS / RSS Conference Paper Submission
                       - `robocontrol-bench` Open Benchmark & Model Zoo
                     ═════════════════════════════════════════════════════
```

---

### Concrete Week-by-Week Execution Timeline

```
Week  1: [EXP-001 Execute] -> [EXP-002 Execute] -> Verify Replication & MIP Convergence
Week  2: [EXP-003 Execute] -> [EXP-004 Latency Sweep] -> Aggregate H1 CSVs & Figures
Week  3: Package HF Hub Models/Data -> Build Gradio Spaces -> Publish HF Community Blog (H1 COMPLETE)
Week  4: Set up ManiSkill3 & robosuite Sim Stack -> Implement WAM Baseline architecture
Week  5: [EXP-005 Execute] -> [EXP-010 Multi-Modal Mode Recovery Execute]
Week  6: [EXP-006 Sim-to-Real Execute] (PBR / Domain Randomization / Real Splits)
Week  7: [EXP-007 Whole-Body MPC Execute] (Quadruped Go2 & Humanoid Dynamics)
Week  8: [EXP-008 Real-Vision Stress Execute] -> [EXP-009 Contact-Rich Manipulation Execute]
Week  9: Full Data Aggregation -> Statistical Audits -> Master Comparison Table Synthesis
Week 10: Draft Conference Paper Manuscript -> Build `robocontrol-bench` Evaluator Package
Week 11: Internal Red-Team Review -> Verify Determinism Spot-Checks -> Finalize Codebase
Week 12: Submit Paper to Conference (CoRL/NeurIPS/RSS/ICRA) & Launch Open Benchmark (H2 COMPLETE)
```

---

## 6. Statistical Rigor, Stopping Rules & Protocol Governance

To prevent p-hacking, confirmation bias, and irreproducible claims, the study strictly enforces pre-registered protocol governance:

### 6.1 Pre-Registration Locking
- All experiment parameters, random seed lists (`[0, 1, 2, 42, 123]`), model dimensions, solver tolerances, evaluation predicates, and statistical tests are locked prior to rollout execution.
- Any change in hyperparameters or environment configuration requires a dated entry in `outputs/protocol_amendments.md` explaining the scientific justification.

### 6.2 Statistical Testing Protocols
1. **Paired Comparisons:** Since all conditions are evaluated on identical episode initial states (`eval_seeds_exp*.json`), paired comparisons use the two-sided **Wilcoxon signed-rank test**.
2. **Multiple Comparison Corrections:** For family-wide comparisons involving $K$ conditions, Bonferroni corrections are applied:
   $$\alpha_{\text{corrected}} = \frac{0.05}{C(K, 2)}$$
3. **Factorial Interactions:** Two-way repeated-measures **ANOVA** ($F$-statistic, partial $\eta^2$) evaluates Condition $\times$ Perturbation Level interactions in EXP-003, EXP-006, and EXP-008.
4. **Effect Size Reporting:** All primary findings report Rank-Biserial Correlation ($r$) or Cohen's $d$ alongside 95% bootstrap confidence intervals ($B=10,000$ resamples).

### 6.3 Mandatory Stopping & Gate Rules
1. **Replication Stop Gate (EXP-001):** If Full DDPM on PushT deviates $> 5.0\text{ pp}$ from Simchowitz et al. published baseline, execution is halted immediately to diagnose training schedules before any mechanism conclusions are drawn.
2. **Solver Convergence Stop Gate (EXP-002 / EXP-007):** If MPC solvers fail to converge on $> 15\%$ of unconstrained clear episodes, solver tuning (SQP line-search, trust region) must be debugged and re-run.
3. **No Peeking Rule:** Hypothesis tests are strictly executed only after all 5 seeds $\times$ 100 episodes are completely rolled out and serialized to disk.
4. **Futility Rule:** If after 3 completed seeds a condition is dominated by $> 25.0\text{ pp}$ across all benchmarks, early termination is permissible provided it is documented in the final report.

---

## 7. Artifact Release & Hugging Face Hub Integration Plan

All generated assets are structured for direct publication on the Hugging Face Hub to maximize developer adoption and scientific reproducibility:

### 7.1 Hugging Face Hub Repositories

```
  Hugging Face Collection: Ryukijano/mpc-vla-diffusion-study
  │
  ├── Models:
  │   ├── Ryukijano/smallvla-mpc-vla-diffusion-quick  (~340 MB) [SmallVLA PyTorch checkpoint]
  │   ├── Ryukijano/diffusion-policy-quick            (~5.6 MB) [ConditionalUNet1D DDPM]
  │   ├── Ryukijano/flow-matching-policy-quick        (~5.6 MB) [Rectified Flow Policy]
  │   ├── Ryukijano/mip-policy-quick                  (~4.8 MB) [Minimal Iterative Policy]
  │   └── Ryukijano/world-action-model-franka-100m    (~420 MB) [Horizon 2 Latent WAM]
  │
  ├── Datasets:
  │   ├── Ryukijano/mpc-expert-demos-quick-test       [NumPy NPZ + LeRobot v2 Parquet]
  │   └── Ryukijano/robocontrol-bench-multimodal      [RoboMimic & ManiSkill3 Multi-Modal]
  │
  └── Spaces (Gradio / WebXR):
      ├── Ryukijano/mpc-vla-diffusion-plot-gallery    [Interactive Pareto & Metric Explorer]
      └── Ryukijano/mpc-vla-diffusion-arena           [Live Browser Controller Rollout Arena]
```

### 7.2 Directory Tree of Master Study Repository

```
/home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study/
├── benchmarks/                           # Standardized gym/gymnasium environments
│   ├── base_env.py
│   ├── reaching_env.py                   # 2D Reaching (Clear & Cluttered)
│   ├── pusht_env.py                      # PushT (State, Image, Language)
│   ├── metaworld_wrapper.py
│   └── demonstration_collector.py        # MPC trajectory recorder (NPZ / LeRobot)
├── mpc_baselines_repo/                   # Classical optimization solvers
│   ├── src/linear_mpc/                   # OSQP Quadratic Programming
│   ├── src/nonlinear_mpc/                # CasADi / iLQR Interior-Point SQP
│   ├── src/collision_free_mpc/           # SDF Obstacle Avoidance
│   └── src/diffusion_warm_start/         # Diffusion Prior + MPC Refinement
├── vla_baselines/                        # Vision-Language-Action Models
│   ├── small_vla.py                      # Compute-matched SmallVLA (~86M params)
│   ├── openvla_interface.py              # OpenVLA-7B adapter
│   └── vla_trainer.py
├── diffusion_baselines/                  # Generative Control Policies
│   ├── ddpm_policy.py                    # Score Matching DDPM (Chi et al.)
│   ├── flow_matching_policy.py           # Rectified Flow Matching
│   ├── iterative_regression_policy.py    # Minimal Iterative Policy (Simchowitz MIP)
│   └── conditional_unet1d.py
├── docs/                                 # Documentation & Protocols
│   ├── research_questions.md             # Primary RQ1..RQ12 specifications
│   ├── comparison_plan.md
│   ├── methodology.md
│   ├── blogging/                         # HF Community Blog drafts & audits
│   └── experiments/
│       └── master_experiment_roadmap.md  # THIS MASTER DOCUMENT
└── experiments/                          # Pre-registered experiment suites
    ├── EXP-001-mechanism-ablation/
    ├── EXP-002-family-comparison/
    ├── EXP-003-ood-robustness/
    ├── EXP-004-latency-pareto/
    ├── EXP-005-world-models/
    ├── EXP-006-sim-to-real/
    ├── EXP-007-real-robot-mpc/
    ├── EXP-008-real-vision/
    ├── environment.lock                  # Pinned dependency freeze
    └── run_all.sh                        # Master automated execution harness
```

---

*This master document stands as the canonical execution blueprint for the entire study. All subsequent experiment runs, data serializations, and publications must cross-reference this roadmap.*

---

## 8. Current Execution Status

> **Live tracker:** See `docs/STATUS.md` for the full, up-to-date execution state. The summary below is current as of 2026-08-14.

### 8.1 Overall State

| Milestone | Status | Notes |
|-----------|--------|-------|
| EXP-001 — GCP Mechanism Ablation | **In-progress** | Medium 25-episode × 5-seed run complete (`results/exp001/`, PushT + Reaching). Full 100-episode pre-registered run still pending. |
| EXP-002 — Three-Family Comparison | **Pending** | No full run; only quick smoke on `reaching` (`results/quick_test/`, no VLA). |
| EXP-003 — OOD Robustness | **Pending** | No outputs. |
| EXP-004 — Latency-Performance Pareto | **In-progress (smoke)** | CPU-only low-latency smoke complete (`results/exp004_cpu_low_latency_smoke/`). Canonical GPU Pareto sweep not started. |
| EXP-005..010 — Horizon 2 | **Pending** | Protocol dirs for EXP-005–008 exist; EXP-009 and EXP-010 not yet created. |
| Model Checkpoints | **Partial** | Four PushT baseline checkpoints verified in `results/checkpoints/`; full per-condition checkpoint trees not yet produced. |
| Hugging Face Hub Artifacts | **Packaged locally / upload pending** | `dist/hf_models/` and `dist/hf_datasets/` are populated; `results/hf_artifacts/` card templates and upload checklist exist. No Hub repos created or uploaded. |
| HF Community Blog | **Draft / blocked** | Draft at `docs/blogging/hf_blog_draft.md`; publication waiting on full Horizon 1 results and actual Hub uploads. |

### 8.2 What Is Already on Disk

- **EXP-001 medium run:** `results/exp001/` contains 5-seed, 25-episode ablation outputs for PushT and 2-D Reaching (`ablation_results.json`, `ablation_aggregated.csv`, `ablation_comparison.csv`, 3 PNGs). This is the run from `scripts/run_horizon1.sh:32–38`.
- **EXP-004 CPU smoke:** `results/exp004_cpu_low_latency_smoke/` contains 1-seed, 2-episode, 17-condition Pareto data on 2-D Reaching from `scripts/run_pareto_cpu_low_latency.py`.
- **Quick smoke-test results:** `results/quick_test/` (1 seed, 5 episodes, `reaching` only, `mpc` + `diffusion` controllers).
- **Baseline checkpoints:** `results/checkpoints/{small_vla_pusht.pt, ddpm_pusht.pt, flow_matching_pusht.pt, mip_pusht.npz}` with `release_manifest.json`.
- **HF-ready packages:** `dist/hf_models/{small_vla,ddpm,flow_matching,mip}/` and `dist/hf_datasets/mpc_expert_demos/`.
- **HF card templates:** `results/hf_artifacts/` including `UPLOAD_CHECKLIST.md` and model/dataset card READMEs.
- **Earlier failure log:** `results/horizon1_runbook.log` documents an attempt that failed due to a now-fixed CLI mismatch (`--benchmarks` plural). The current `scripts/run_horizon1.sh` uses the correct singular `--benchmark` flag.
- **Environment provenance:** `results/env_info.json`.

### 8.3 Key Blockers

1. **Full 100-episode EXP-001 not yet run.** The existing 25-episode run is below the pre-registered sample size and is stored in `results/exp001/` rather than the canonical `experiments/EXP-001-mechanism-ablation/outputs/`.
2. **Missing fixed evaluation seed files.** `data/eval_seeds_exp{001..004}.json` are required by the protocols and are currently absent.
3. **EXP-002 not started.** It depends on EXP-001 and requires VLA to be exercised end-to-end on PushT.
4. **Canonical EXP-004 GPU Pareto sweep not started.** The CPU smoke is a placeholder/low-latency probe only.
5. **Canonical runner ambiguity for EXP-004.** `experiments/README.md` points to `run_experiments.py`, while `scripts/run_horizon1.sh` points to `scripts/run_pareto_sweep.py`.
6. **VLA end-to-end validation** on PushT has not been completed; a local SmallVLA checkpoint exists but the quick smoke test excluded VLA because it used state-only `reaching`.

### 8.4 Next Steps

1. Run full 100-episode × 5-seed EXP-001 to the canonical `experiments/EXP-001-mechanism-ablation/outputs/`.
2. Generate/restore `data/eval_seeds_exp{001..004}.json`.
3. Run EXP-002 three-family comparison (including SmallVLA on PushT).
4. Run canonical EXP-004 GPU Pareto sweep and reconcile the runner choice.
5. Upload HF Hub model and dataset repos (remove the dry-run `dist/hf_models/test-model/` first).
6. Create protocol directories for EXP-009 and EXP-010.
7. Update `docs/STATUS.md` and this section after each completed run.

