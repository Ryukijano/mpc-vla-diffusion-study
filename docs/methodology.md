# Evaluation Methodology

## Benchmarks

### Benchmark Selection Rationale

We select benchmarks that span the strength zones of all three families:

| Benchmark | Why included | Primary family advantage |
|-----------|-------------|--------------------------|
| MetaWorld (50 tasks) | Standard ML manipulation benchmark | Diffusion policy, VLA |
| RoboMimic (multi-modal) | Tests multi-modality claim directly | Diffusion (claimed) |
| CALVIN (long-horizon, language) | Tests language conditioning | VLA |
| LIBERO (language-conditioned) | VLA standard benchmark | VLA |
| DMControl (classical control) | MPC-friendly, explicit dynamics | Classical MPC |
| Custom: cluttered reaching | Tests constraint handling + OOD | MPC, Hybrid |

### Observation Modalities

Each benchmark is evaluated with multiple observation modalities to enable fair comparison:
1. **State-only** (joint positions, velocities, object positions) — fair to MPC
2. **Image** (RGB camera) — fair to VLA and diffusion policy
3. **Image + Language** — fair to VLA
4. **Point cloud** — fair to diffusion policy (3D conditioning)

### Multi-Modal Demonstration Subsets

For RQ2 (multi-modality), we use RoboMimic's multi-modal demonstration sets:
- `Lift` (can be solved with multiple strategies)
- `Can` (multiple grasping approaches)
- `Square` (multiple insertion paths)

These have demonstrations collected from multiple human operators using different strategies,
creating genuine multi-modal action distributions.

## Metrics

### Primary Metrics

| Metric | Definition | How measured |
|--------|------------|--------------|
| **Success Rate** | % episodes achieving task goal | Binary per episode, averaged |
| **Return** | Cumulative reward | Sum of per-step rewards |
| **Inference Latency** | Time from observation to action | Wall-clock, median over 1000 calls |
| **Data Efficiency** | Success rate vs. # demonstrations | Curve, AUC |

### Secondary Metrics (Multi-Modality Debate)

| Metric | Definition | How measured |
|--------|------------|--------------|
| **Mode Coverage** | % of expert modes recovered by policy | Cluster expert actions, check policy samples hit each cluster |
| **Action KL** | KL(policy actions ‖ expert actions) | Kernel density estimate, KL divergence |
| **Manifold Adherence** | Distance of OOD actions to expert manifold | k-NN distance to expert action set |
| **Constraint Violation** | % timesteps violating safety constraints | Per-step binary, averaged |

### Latency Measurement Protocol

- Measure on DGX Spark (GB10) with fixed GPU clock
- Warm up 100 inferences before timing
- Time 1000 inferences, report median + p95
- For MPC: include solver time only (not model computation)
- For VLA: include full VLM forward pass
- For diffusion: include all denoising steps
- For hybrid: include both components

## Evaluation Protocol

### Episode Setup
- 100 evaluation episodes per (method, task, seed) combination
- 5 random seeds: 0, 1, 2, 42, 123
- Fixed episode seeds across methods (same initial conditions)
- Episode length: task-dependent (50-500 steps)
- Success criterion: task-specific (e.g., object in target zone)

### Statistical Testing
- **Paired comparisons** (same episodes): Wilcoxon signed-rank test
- **Unpaired comparisons**: Mann-Whitney U test
- **Multiple comparisons**: Bonferroni correction
- **Significance level**: p < 0.05
- Report: mean ± std, effect size (Cohen's d), p-value

### Fairness Controls
- Same observation preprocessing (resize, normalize) across methods
- Same action space (joint positions or end-effector poses)
- Same control frequency (10Hz for learning-based, native for MPC)
- Same training data (for learning-based methods)
- Same training budget (epochs or steps, compute-matched where possible)

## Ablation Protocol (Phase 4)

### GCP Component Ablation

Following Simchowitz et al., ablate these components one at a time:

| Variant | Iterative Steps | Noise Injection | Distribution Fitting | Description |
|---------|----------------|-----------------|---------------------|-------------|
| Full Flow GCP | T=16 | Yes | Yes (flow matching) | Complete generative policy |
| GCP no-noise | T=16 | No | Yes | Remove training stochasticity |
| GCP single-step | T=1 | Yes | Yes | Remove iterative compute |
| MIP (2-step) | T=2 | Yes | No (regression) | Minimal iterative policy |
| Pure RCP | T=1 | No | No (regression) | Regression control policy |

All variants use the same backbone architecture, same training data, same training budget.

### Observation Modality Ablation

| Configuration | Observation | Language | Point Cloud |
|---------------|-------------|----------|-------------|
| State-only | ✓ | ✗ | ✗ |
| Image | ✓ (RGB) | ✗ | ✗ |
| Image + Lang | ✓ (RGB) | ✓ | ✗ |
| Point Cloud | ✗ | ✗ | ✓ |

### OOD Perturbation Levels

| Level | Perturbation | Description |
|-------|-------------|-------------|
| 0 (ID) | None | In-distribution test |
| 1 | Object color | Change object texture/color |
| 2 | Object position | Shift object ±5cm |
| 3 | Camera viewpoint | Rotate camera ±10° |
| 4 | Lighting | Change scene lighting |
| 5 | Combined | All perturbations together |

Measure: success rate and manifold adherence at each level.

## Reproducibility Checklist

- [ ] All configs committed to `configs/`
- [ ] Random seeds fixed and documented
- [ ] Environment captured (conda env, pip freeze)
- [ ] Data manifests with hashes
- [ ] Evaluation episodes fixed (seed files in `data/`)
- [ ] Code tagged at each experiment milestone
- [ ] Results logged with config hash
- [ ] Figures generated from committed results (not manual)
