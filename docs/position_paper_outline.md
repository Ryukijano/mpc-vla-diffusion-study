# Position Paper Outline: "Do We Need Diffusion in Robotics? An Empirical Reappraisal"

## Target Venue
- Workshop: RSS 2026 "Diffusion for Robot Learning" (or similar)
- Or: Conference paper (CoRL, RSS, ICRA 2027)

## Outline

### Abstract (~200 words)
- The debate: are diffusion-based GCPs essential for robot control, or is their
  advantage an artifact of iterative compute + noise?
- Our approach: systematic comparison of classical MPC, VLA, and diffusion policies
  across 6 benchmarks, with component-level ablations
- Key finding: [TBD from experiments]
- Implication: [TBD]

### 1. Introduction (~1 page)
- The rise of generative control policies (diffusion policy, flow matching)
- The prevailing wisdom: GCPs capture multi-modal action distributions
- The challenge: Simchowitz et al. (2025) argue the mechanism is iterative compute + noise
- Our question: where does each family (MPC, VLA, diffusion) actually win?
- Contributions: (1) systematic comparison, (2) ablation replicating Simchowitz,
  (3) Pareto analysis, (4) hybrid evaluation

### 2. Background (~1 page)
- 2.1 Classical MPC: formulation, strengths, limitations
- 2.2 VLA models: architecture, training, strengths, limitations
- 2.3 Diffusion policies / GCPs: formulation, claimed advantages
- 2.4 The Simchowitz critique: MIP, iterative compute, noise injection

### 3. Experimental Setup (~1.5 pages)
- 3.1 Benchmarks (MetaWorld, RoboMimic, CALVIN, LIBERO, DMControl, custom)
- 3.2 Methods compared (table of all baselines)
- 3.3 Metrics (success, latency, mode coverage, manifold adherence, constraint violation)
- 3.4 Evaluation protocol (seeds, episodes, statistical tests)

### 4. Results (~3 pages)
- 4.1 Main comparison: success rate across benchmarks (table + bar chart)
- 4.2 Multi-modality analysis: do diffusion policies capture multiple modes? (H2)
- 4.3 Mechanism ablation: full GCP vs MIP vs regression (H3) — replicate Simchowitz
- 4.4 OOD robustness: manifold adherence under perturbation (H4)
- 4.5 Where MPC wins: constraint-heavy tasks, real-time (H5)
- 4.6 Where VLA wins: language tasks, generalization (H6)
- 4.7 Hybrid approaches: diffusion warm-start + MPC, VLA + diffusion (H7)
- 4.8 Latency-performance Pareto frontier (H8)

### 5. Analysis (~1 page)
- 5.1 Reconciling our findings with Simchowitz et al.
- 5.2 When diffusion IS needed (high precision, OOD manifold adherence)
- 5.3 When classical MPC is sufficient (constraints, real-time, known dynamics)
- 5.4 When VLA is the right choice (language, generalization, zero-shot)
- 5.5 The case for hybrid architectures

### 6. Conclusion (~0.5 page)
- Answer to "Do we need diffusion in robotics?": nuanced yes — not for the reasons
  commonly believed, but for specific niches (OOD robustness, high precision)
- Future directions: better iterative policies without distribution fitting,
  constraint-aware generative policies, real-time VLA

### References

### Appendix
- A: Full results tables
- B: Per-task breakdown
- C: Architecture details
- D: Hyperparameter sensitivity
