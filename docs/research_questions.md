# Research Questions & Hypotheses

## Primary Research Question

**RQ1:** Do diffusion-based generative control policies (GCPs) provide measurable advantages
over classical MPC and VLA policies for robotic manipulation, and what is the mechanism?

## Sub-Questions

### RQ2: Multi-Modality
Do diffusion policies actually capture multi-modal action distributions in practice?
- **H2a (pro-diffusion):** Diffusion policies recover multiple demonstration modes on
  multi-modal benchmarks (RoboMimic multi-modal), while regression and MPC recover only one.
- **H2b (Simchowitz):** Diffusion policies do NOT meaningfully capture multi-modality;
  with proper architecture, regression ≈ flow on most tasks. Flow wins only on
  high-precision tasks.

### RQ3: Mechanism
Is the advantage of GCPs due to distribution fitting or iterative compute + noise?
- **H3a (conventional):** The generative (distribution fitting) component is essential.
- **H3b (Simchowitz):** Iterative compute + noise injection is the key; distribution
  fitting is not. MIP (2-step regression + noise) matches full flow GCPs.

### RQ4: OOD Robustness
Does noise injection + iterative compute improve out-of-distribution robustness?
- **H4:** GCPs and MIP exhibit better manifold adherence (stay close to expert action
  manifold) under OOD observations than single-step regression.

### RQ5: Classical MPC Niche
Where does classical MPC still dominate?
- **H5:** Classical MPC dominates on: (a) tasks with hard safety constraints,
  (b) real-time control (>100Hz), (c) tasks with accurate analytical dynamics models.
- **H5b:** Classical MPC fails on: (a) novel objects/scenes without re-modeling,
  (b) language-conditioned tasks, (c) tasks requiring visual reasoning.

### RQ6: VLA Niche
Where does VLA dominate?
- **H6:** VLA dominates on: (a) language-conditioned tasks, (b) zero-shot transfer
  to novel objects, (c) tasks requiring semantic reasoning.
- **H6b:** VLA struggles on: (a) real-time control (latency), (b) high-precision
  manipulation, (c) tasks without language instructions.

### RQ7: Hybrid Approaches
Can hybrid approaches (diffusion warm-start + MPC, VLA + diffusion action) get the
best of multiple worlds?
- **H7:** Hybrid approaches outperform pure approaches on tasks that require both
  constraint satisfaction AND visual/language generalization.

### RQ8: Latency-Performance Pareto
What is the latency-performance Pareto frontier across all three families?
- **H8:** Classical MPC occupies the low-latency / moderate-performance region;
  VLA occupies the high-latency / high-generalization region;
  Diffusion policies occupy the middle, with MIP on the Pareto frontier.

## Evaluation Criteria for Each Hypothesis

| Hypothesis | Metric | Test |
|------------|--------|------|
| H2a/H2b | Mode coverage %, KL divergence to expert | Wilcoxon signed-rank |
| H3a/H3b | Success rate: full GCP vs MIP vs regression | Paired t-test |
| H4 | Manifold adherence score under OOD | ANOVA across perturbation levels |
| H5 | Success rate on constraint-heavy vs vision-heavy tasks | Chi-square |
| H6 | Success rate on language vs non-language tasks | Chi-square |
| H7 | Success rate: hybrid vs best pure | Paired t-test |
| H8 | Pareto dominance (latency, success) | Dominance counting |
