# Seed Papers: MPC vs VLA vs Diffusion MPC

## The Core Debate

### 1. Much Ado About Noising: Dispelling the Myths of Generative Robotic Control
- **Authors:** Pan, Anantharaman, Huang, Jin, Pfrommer, Yuan, Permenter, Qu, Boffi, Shi, Simchowitz
- **arXiv:** 2512.01809
- **URL:** https://arxiv.org/abs/2512.01809
- **Code:** https://github.com/simchowitzlabpublic/much-ado-about-noising
- **Project:** https://simchowitzlabpublic.github.io/much-ado-about-noising-project/
- **Key claim:** GCPs' success is from iterative compute + noise injection, not multi-modal distribution fitting. MIP (2-step regression) matches flow GCPs.
- **Relevance:** THE paper this study is built around. Provides the counter-narrative to diffusion policy hype.

### 2. Do we need diffusion in robotics? (Talk)
- **Speaker:** Max Simchowitz (CMU)
- **Venue:** Simons Institute, "Diffusion Generative Modeling: Progress and Next Steps" workshop
- **Date:** Aug 7, 2026
- **URL:** https://live-simons-institute.pantheon.berkeley.edu/talks/max-simchowitz-carnegie-mellon-university-2026-08-07
- **YouTube:** https://www.youtube.com/live/q0lmXek-x_k
- **Relevance:** The talk that motivated this study. Presents the "Much Ado About Noising" findings.

### 3. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- **Authors:** Chi, Feng, Du, Xu, Cousineau, Russell, Song
- **arXiv:** 2303.04137
- **Key claim:** Diffusion models for robot action generation capture multi-modal action distributions.
- **Relevance:** The foundational diffusion policy paper that Simchowitz et al. challenge.

## Classical MPC

### 4. Warm-Starting Collision-Free MPC with Object-Centric Diffusion
- **arXiv:** 2601.02873
- **Key claim:** Diffusion warm-start + collision-aware MPC = reliable real-time motion in clutter.
- **Relevance:** Hybrid approach — uses diffusion to warm-start classical MPC. Direct comparison point.

### 5. Model Predictive Control: Theory and Design
- **Author:** James B. Rawlings, David Q. Mayne, Moritz M. Diehl
- **Relevance:** The canonical MPC textbook. For baseline MPC formulations.

## VLA (Vision-Language-Action)

### 6. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- **Authors:** Brohan et al. (Google DeepMind)
- **arXiv:** 2307.15818
- **Key claim:** VLM → action head, transfer web knowledge to robot control.
- **Relevance:** Foundational VLA paper.

### 7. OpenVLA: An Open-Source Vision-Language-Action Model
- **Authors:** Kim et al. (Stanford/Berkeley)
- **arXiv:** 2406.09246
- **Key claim:** Open-source 7B VLA with competitive performance.
- **Relevance:** Practical open-source VLA baseline we can actually run.

### 8. What Matters in Building Vision-Language-Action Models for Generalist Robots
- **arXiv:** 2412.14058
- **Key claim:** Systematic study of VLA design choices.
- **Relevance:** Meta-study of VLA design — informs our VLA baseline choices.

### 9. π0 / π0-flow (Physical Intelligence)
- **Key claim:** Flow-matching VLA for generalist robot control.
- **Relevance:** State-of-the-art VLA with diffusion/flow action head — hybrid of VLA + GCP.

## Diffusion MPC / GCPs

### 10. Diffusion-Based Approximate MPC: Fast and Consistent Imitation of Multi-Modal Action Distributions
- **arXiv:** 2504.04603
- **Key claim:** Diffusion models approximate multi-modal MPC solution distributions at kHz rates.
- **Relevance:** Direct "diffusion MPC" paper — uses diffusion to imitate MPC expert. Tests the multi-modality claim.

### 11. VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation
- **arXiv:** 2407.09829
- **Key claim:** VLM samples candidate actions + video prediction model + hierarchical cost = VLMPC.
- **Relevance:** Hybrid VLM + MPC — bridges VLA and MPC worlds.

### 12. DiffusionVLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression
- **Authors:** Wen et al.
- **Venue:** MLR 2026
- **Key claim:** Autoregressive reasoning (VLM) + diffusion action policy = DiVLA.
- **Relevance:** Hybrid VLA + diffusion — the convergence point.

## Recent / Hybrid / Streaming

### 13. StreamingVLA: Streaming VLA with Action Flow Matching and Adaptive Early Observation
- **arXiv:** 2603.28565
- **Key claim:** Asynchronous streaming across VLA stages, 2.4x latency speedup.
- **Relevance:** Addresses VLA latency problem — relevant for latency comparison.

### 14. Realtime-VLA FLASH: Speculative Inference for Diffusion-based VLAs
- **Date:** May 2026
- **Key claim:** Speculative decoding for DVLA: 58ms → 7.8ms, 3x speedup.
- **Relevance:** Latency optimization for diffusion VLAs — relevant for real-time comparison.

### 15. DAWN: Pixel Motion Diffusion is What We Need for Robot Control
- **Venue:** CVPR 2026
- **Key claim:** Unified diffusion framework for language-conditioned manipulation (high + low level).
- **Relevance:** Pro-diffusion counterpoint to Simchowitz — claims diffusion IS what we need.

## To Be Added (via literature search)

Use `literature_search_arxiv` and `literature_search_openalex` skills to find:
- Flow matching for control (beyond diffusion)
- MPC + learning hybrids (learning-based MPC, AMPC)
- Behavior cloning theory (when does BC work / fail?)
- Multi-modality in robot demonstrations (evidence for/against)
- Constraint-aware generative policies
- Real-time diffusion inference optimization
