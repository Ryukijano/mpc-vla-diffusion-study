# Recent SOTA Digest — MPC vs VLA vs Diffusion Study

**Generated:** 2026-08-14
**Scope:** Recent (2025-2026) papers directly relevant to the study's scientific claims.

---

## Key Findings Summary

| Paper | Topic | Year | Relevance to H3 (Simchowitz) | Citation Recommendation |
|-------|-------|------|------------------------------|--------------------------|
| Much Ado About Noising | Diffusion mechanism | 2025 | **Core paper** - directly validates H3 | Both blog & conference |
| FlowMPC | MPC + learned policies | 2026 | Supports hybrid approach | Conference |
| World Action Models are Zero-shot Policies | WAMs | 2026 | Contrasts diffusion with dynamics | Conference |
| OpenVLA | VLA generalization | 2025 | Baseline for comparison | Conference |
| πR²: Reactive Real-time Flow Policies | VLA latency | 2026 | Addresses real-time deployment | Conference |
| Factored Diffusion Policies | Diffusion mechanism | 2026 | Partially supports H3 | Conference |
| BIFROST: Sim-to-real transfer | OOD robustness | 2026 | Addresses domain gap | Conference |

---

## Detailed Paper Summaries

### 1. Much Ado About Noising: Dispelling the Myths of Generative Robotic Control
- **Authors:** Chaoyi Pan, Giri Anantharaman, Nai-Chieh Huang, Claire Jin, Daniel Pfrommer, Chenyang Yuan, Frank Permenter, Guannan Qu, Nicholas Boffi, Guanya Shi, Max Simchowitz
- **Year:** 2025
- **arXiv ID:** 2512.01809
- **URL:** https://arxiv.org/abs/2512.01809

**Key Contribution:** Comprehensive evaluation of generative control policies (GCPs) on 28 behavior cloning benchmarks. GCPs do **not** owe their success to capturing multi-modal action distributions. Instead, the advantage stems from **iterative computation** combined with **stochastic injection**, provided intermediate steps are supervised during training. Introduces Minimal Iterative Policy (MIP), a lightweight two-step regression-based policy that matches flow-based GCP performance.

**Relation to H3:** **Core paper validating H3**. Contradicts the common belief that diffusion/flow policies succeed due to multi-modal distribution fitting; identifies iterative compute + noise injection as the key mechanism.

**Citation:** Both blog and conference.

---

### 2. FlowMPC: Improving Flow Matching Policies with World Models
- **Author:** Chandon Hamel
- **Year:** 2026
- **arXiv ID:** 2606.16286
- **URL:** https://arxiv.org/abs/2606.16286

**Key Contribution:** Combines Flow Matching (FM) policies with a learned world model for test-time MPPI planning. Uses the FM policy to propose action sequences and evaluates them with a world model without modifying FM training. On ManiSkill tasks, improves end-of-episode success.

**Relation to H3:** Supports a **hybrid approach** — learned priors plus explicit planning yield additional gains, suggesting action distribution fitting is less critical than planning.

**Citation:** Conference paper.

---

### 3. World Action Models are Zero-shot Policies
- **Authors:** Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, et al.
- **Year:** 2026
- **arXiv ID:** 2602.15922
- **URL:** https://arxiv.org/abs/2602.15922

**Key Contribution:** Introduces DreamZero, a 14B World Action Model (WAM) built on a pretrained video diffusion backbone. WAMs jointly predict future video states and actions, learning physical dynamics from heterogeneous robot data. Achieves 2× improvement in generalization over state-of-the-art VLAs in real robot experiments.

**Relation to H3:** **Contrasts** Simchowitz — while action-space distribution fitting may not matter, **state-space dynamics modeling** can be valuable for generalization.

**Citation:** Conference paper.

---

### 4. OpenVLA: An Open-Source Vision-Language-Action Model
- **Authors:** Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, et al.
- **Year:** 2025 (CoRL 2025)
- **arXiv ID:** 2406.09246
- **URL:** https://arxiv.org/abs/2406.09246

**Key Contribution:** 7B-parameter open-source VLA trained on 970k real-world robot demonstrations. Outperforms closed RT-2-X (55B) by 16.5% absolute success rate across 29 tasks. Demonstrates effective fine-tuning via LoRA and quantization.

**Relation to H3:** **Counterpoint** — VLA can outperform Diffusion Policy by 20.4%, suggesting scale/pretraining/vision-language alignment matter beyond the generative mechanism.

**Citation:** Conference paper.

---

### 5. πR²: Reactive Real-time Flow Policies
- **Year:** 2026
- **arXiv ID:** 2607.26055
- **URL:** https://arxiv.org/abs/2607.26055

**Key Contribution:** Addresses latency in action-chunking flow policies by splitting conditioning into fast (proprioception) and slow (vision-language) channels. Uses latency-adaptive flow schedule. Achieves 4× faster replanning (25 Hz on A5000 GPU). Improves success rate by up to 23% in simulation and 30% in real world.

**Relation to H3:** The iterative compute mechanism introduces latency; this work shows architectural mitigations, supporting the study's latency-as-differentiator analysis.

**Citation:** Conference paper.

---

### 6. Factored Diffusion Policies: Compositionally Generalized Robot Control with a Single Score Network
- **Year:** 2026
- **arXiv ID:** 2605.22596
- **URL:** https://arxiv.org/abs/2605.22596

**Key Contribution:** Decomposes robot tasks into factors and trains a single shared diffusion network with per-factor null-token dropout. Score decomposes additively at inference. On drone racing, matches oracle and improves zero-shot venue transfer.

**Relation to H3:** **Partially supports** Simchowitz — structured iterative compute yields better generalization, but architecture also matters, suggesting iterative compute is necessary but not sufficient.

**Citation:** Conference paper.

---

### 7. BIFROST: Bridging Invariant Feature Representation for Observation-space Sim2Real Transfer
- **Year:** 2026
- **arXiv ID:** 2607.01410
- **URL:** https://arxiv.org/abs/2607.01410

**Key Contribution:** Learns a shared history encoder on paired cross-domain data via cross-domain bisimulation objective. Policies trained on these latent states in simulation transfer zero-shot to reality.

**Relation to H3:** Addresses OOD robustness. Aligns with the idea that staying on the "manifold of expert data" is key, and invariant representations can extend that manifold across domains.

**Citation:** Conference paper.

---

## Synthesis for the Study

### Support for H3 (Simchowitz)
- **Strong support:** "Much Ado About Noising" provides empirical evidence that iterative compute + stochasticity, not multi-modal distribution fitting, drives diffusion policy success.
- **Nuanced support:** FlowMPC shows that adding explicit planning (world model) to flow policies yields gains, suggesting distribution fitting alone is insufficient.
- **Partial support:** Factored Diffusion Policies shows that structured iterative compute works better, but still relies on the iterative mechanism.

### Contrasts / Refinements to H3
- **WAMs** suggest distribution learning over *state space* (future video prediction) is valuable for generalization, even if action-space distribution fitting is not.
- **πR²** shows the iterative compute mechanism introduces latency challenges requiring architectural solutions.
- **OpenVLA** shows scale, pretraining, and vision-language alignment can outweigh the generative mechanism.

### Key Takeaways
1. **Diffusion mechanism:** Iterative compute + noise is key, not multi-modal action distribution fitting.
2. **VLA vs Diffusion:** VLAs can outperform Diffusion Policy when scaled and pretrained on vision-language data.
3. **MPC vs Learned:** Hybrid approaches (FlowMPC, Residual MPC) show promise by combining learned priors with explicit planning.
4. **WAMs:** World dynamics modeling (state-space prediction) enables better generalization than action-space distribution fitting.
5. **Latency:** Diffusion/flow policies face real-time challenges; πR² addresses this.
6. **Sim-to-real:** Invariant representation learning (BIFROST) is more effective than domain adaptation for OOD robustness.
