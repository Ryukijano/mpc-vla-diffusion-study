# Comparison Plan: Classical MPC vs VLA vs Diffusion MPC

## 1. Research Question

**Primary:** Do diffusion-based generative control policies (GCPs) provide measurable
advantages over classical MPC and VLA-style policies for robotic manipulation, and if so,
what is the mechanism — multi-modal distribution fitting, or iterative compute + noise
injection (as argued by Simchowitz et al.)?

**Secondary:**
- Under what conditions (task precision, OOD observations, multi-modal demonstrations)
  does each family excel?
- Can a minimal iterative policy (MIP) match full diffusion GCPs, as claimed?
- Where does classical MPC still dominate (constraints, safety, real-time)?
- Where does VLA dominate (language conditioning, generalization, zero-shot transfer)?

## 2. Three Families Under Comparison

### 2.1 Classical MPC (Optimization-Based Control)

**Paradigm:** At each timestep, solve an online optimization problem:
  min_{u_0..u_{N-1}}  cost(x_0, u, x_pred)
  subject to: dynamics, state constraints, input constraints

**Key properties:**
- Explicit dynamics model (robot + environment)
- Hard constraint handling (collision, joint limits, torque bounds)
- Single-modal solution (solver returns one trajectory)
- Real-time capable (QP/LP solvers at kHz rates)
- Limited generalization to novel scenes/objects without re-modeling
- Sensitive to local minima in non-convex scenes (cluttered environments)

**Representative methods:**
- Linear MPC (QP-based)
- Nonlinear MPC (SQP, interior point)
- Collision-free MPC (SDF-based constraints)
- Warm-started MPC (from sampling or learned priors)

### 2.2 VLA (Vision-Language-Action Models)

**Paradigm:** VLM backbone (pretrained on internet-scale vision+language) + action output head.
Map (image, language instruction, proprioception) → action chunk.

**Key properties:**
- Language-conditioned control (natural language instructions)
- Generalization via VLM pretraining (semantic + visual)
- Autoregressive action generation (token-by-token or chunk-by-chunk)
- High inference latency (VLM forward pass: 50-200ms+)
- Typically single-modal action output (regression head)
- No explicit constraint handling
- Open-loop action chunking common (execute N steps blind, then re-plan)

**Representative methods:**
- RT-2 (Google DeepMind)
- OpenVLA (Stanford/Berkeley)
- π0 (Physical Intelligence)
- Qwen-VLA (Alibaba)
- DiffusionVLA / DiVLA (hybrid: VLM reasoning + diffusion action head)

### 2.3 Diffusion MPC / Diffusion Policy (GCPs)

**Paradigm:** Generative control policy using diffusion/flow matching to generate action
trajectories conditioned on observations. Start from noise, iteratively denoise → action.

**Key properties:**
- Claimed multi-modal action distribution capture
- Iterative computation (T denoising steps)
- Noise injection during training (stochasticity)
- Conditioned on observation (image, state, point cloud, language)
- Moderate inference latency (10-100ms depending on steps)
- No explicit constraint handling (constraints baked into training data)
- Can be warm-start for MPC (diffusion warm-start + MPC refinement)

**Representative methods:**
- Diffusion Policy (Chi et al.)
- Flow Matching policies (e.g., π0-flow)
- Diffusion-based Approximate MPC (arXiv: 2504.04603)
- Warm-Starting Collision-Free MPC with Diffusion (arXiv: 2601.02873)
- VLMPC (Vision-Language MPC, arXiv: 2407.09829)

**Simchowitz et al. finding:** The advantage comes from **iterative compute + noise**,
not distribution fitting. A MIP (2-step regression) matches flow GCPs.

## 3. Comparison Dimensions

| Dimension | Classical MPC | VLA | Diffusion MPC/GCP |
|-----------|--------------|-----|-------------------|
| **Action representation** | Continuous optimization variables | Discrete tokens / regression | Iterative denoising trajectory |
| **Multi-modality** | Single solution (local opt) | Single (regression) or multi (if diffusion head) | Claimed multi-modal (disputed) |
| **Observation conditioning** | State vector | Image + language + proprioception | Image / state / point cloud / language |
| **Dynamics model** | Explicit (analytical or learned) | Implicit (in policy) | Implicit (in policy) |
| **Constraint handling** | Hard constraints (solver-enforced) | None (soft, from data) | None (soft, from data) |
| **Generalization** | Limited (needs re-modeling) | Strong (VLM pretraining) | Moderate (from demo data) |
| **Language conditioning** | None | Native | Via VLM conditioning or text embedding |
| **Inference latency** | <1ms (linear) to ~10ms (nonlinear) | 50-200ms+ | 10-100ms (T denoising steps) |
| **Data efficiency** | N/A (model-based) | Low (needs large demos) | Moderate (needs demos) |
| **Safety guarantees** | Yes (constraint enforcement) | No | No |
| **OOD robustness** | Poor (model mismatch) | Moderate (VLM helps) | Key finding: noise+iter helps manifold adherence |
| **Real-time** | Yes (kHz) | No (open-loop chunking) | Borderline (depends on steps) |
| **Training cost** | N/A | High (VLM finetune) | Moderate (policy training) |

## 4. Experimental Plan

### Phase 1: Literature Review (Weeks 1-2)

**Tools:** `deep-research`, `literature_search_arxiv`, `literature_search_openalex`, `autoresearch`

1. Collect seed papers (see `literature/seed_papers/`)
2. Citation chase via OpenAlex
3. Categorize papers by family (MPC / VLA / Diffusion / Hybrid)
4. Extract claimed advantages and evaluation metrics
5. Identify common benchmarks used across families
6. Write structured literature notes (per-paper in `literature/notes/`)

### Phase 2: Benchmark Selection (Week 2)

Identify benchmarks that all three families can be evaluated on:

**Candidate benchmarks:**
- **MetaWorld** (50 manipulation tasks) — ML benchmark, used by diffusion policy papers
- **CALVIN** (long-horizon language-conditioned) — used by VLA papers
- **RoboMimic** (multi-modal demonstrations) — designed to test multi-modality
- **LIBERO** (language-conditioned manipulation) — VLA standard
- **DMControl / MuJoCo** (classical control) — MPC-friendly
- **Real robot tasks** (if hardware available) — for real-world validation

**Selection criteria:**
- At least 2 benchmarks per family's strength zone
- Must support state-based AND image-based observations
- Must have multi-modal demonstration subsets (to test the multi-modality claim)
- Must have language-conditioned variants (to test VLA)

### Phase 3: Baseline Implementation (Weeks 3-5)

**3.1 Classical MPC baselines** (`src/mpc_baselines/`)
- Linear MPC with QP solver (OSQP)
- Nonlinear MPC with CasADi/iLQR
- Collision-free MPC with SDF constraints
- Configs in `configs/mpc/`

**3.2 VLA baselines** (`src/vla_baselines/`)
- OpenVLA (open-source, 7B)
- π0 or π0-flow (if weights available)
- Small VLA (e.g., Octo) for compute-matched comparison
- Configs in `configs/vla/`

**3.3 Diffusion policy baselines** (`src/diffusion_policies/`)
- Diffusion Policy (Chi et al., DDPM-based)
- Flow Matching policy (rectified flow)
- Minimal Iterative Policy (MIP) — 2-step regression (from Simchowitz)
- Diffusion-based Approximate MPC
- Configs in `configs/diffusion/`

**3.4 Hybrid baselines** (cross-family)
- Diffusion warm-start + MPC refinement
- VLMPC (VLM + MPC)
- DiVLA (VLM reasoning + diffusion action)

### Phase 4: Ablation Studies (Weeks 5-7)

**Tool:** `ablation-study` skill

**4.1 GCP component ablation** (following Simchowitz et al.)
- Full flow GCP (T denoising steps + noise injection)
- GCP without noise injection (deterministic iterative)
- GCP without iterative compute (single-step regression)
- MIP (2-step regression + noise)
- Pure regression (RCP baseline)

**4.2 Observation modality ablation**
- State-only → image → image+language → image+point cloud
- Compare all three families on each modality

**4.3 Multi-modality stress test**
- Use RoboMimic multi-modal demos (multiple solution strategies)
- Measure: does diffusion actually capture multiple modes? (mode coverage metric)
- Compare: MPC (single solution), VLA (single), Diffusion (claimed multi)

**4.4 OOD robustness test** (Simchowitz's key finding)
- Train on in-distribution, test on perturbed observations
- Measure manifold adherence (distance to expert action manifold)
- Compare: RCP vs MIP vs full flow GCP

**4.5 Latency vs. performance trade-off**
- Vary number of denoising steps (1, 2, 4, 8, 16, 32)
- Plot success rate vs. inference time
- Overlay MPC (fixed low latency) and VLA (fixed high latency)

### Phase 5: Evaluation & Metrics (Weeks 7-8)

**Tools:** `experiment-tracking`, `experiment-reproducibility`, `data-visualization`

**Primary metrics:**
- Task success rate (%)
- Return / cumulative reward
- Inference latency (ms per action)
- Training data efficiency (success vs. # demos)

**Secondary metrics (for the multi-modality debate):**
- Mode coverage (how many demonstration modes are recovered?)
- Action distribution KL divergence vs. expert distribution
- Manifold adherence score (OOD action distance to expert manifold)
- Constraint violation rate (for safety comparison)

**Evaluation protocol:**
- 100 episodes per task per method
- 5 random seeds per method
- Same observation preprocessing across methods
- Report mean ± std, plus significance tests (Wilcoxon)

### Phase 6: Analysis & Synthesis (Week 8+)

**Tools:** `academic-plotting`, `collaborative-research`

1. Aggregate results into comparison tables (`results/tables/`)
2. Generate comparison figures (`figures/`)
3. Write analysis document addressing:
   - Does diffusion actually capture multi-modality? (evidence for/against)
   - Is iterative compute + noise the real mechanism? (replicate Simchowitz)
   - Where does classical MPC still win? (constraints, safety, real-time)
   - Where does VLA win? (language, generalization, zero-shot)
   - Is there a principled hybrid? (diffusion warm-start + MPC, or VLA + diffusion action)
4. Identify open questions and future directions

## 5. Key Papers (Seed List)

See `literature/seed_papers/seed_papers.md` for the full annotated list.

**Core (the debate):**
- Pan et al., "Much Ado About Noising" (arXiv: 2512.01809) — Simchowitz's debunking
- Chi et al., "Diffusion Policy" — the original diffusion policy paper
- Simchowitz, "Do we need diffusion in robotics?" (Simons Institute talk, Aug 2026)

**Classical MPC:**
- Mayne, "Model Predictive Control: Theory and Design"
- Warm-Starting Collision-Free MPC with Object-Centric Diffusion (arXiv: 2601.02873)

**VLA:**
- RT-2 (Brohan et al.)
- OpenVLA (Kim et al.)
- "What Matters in Building VLA Models" (arXiv: 2412.14058)

**Diffusion MPC / GCP:**
- Diffusion-Based Approximate MPC (arXiv: 2504.04603)
- VLMPC (arXiv: 2407.09829)
- DiffusionVLA (Wen et al., MLR 2026)

**Hybrid / Recent:**
- StreamingVLA (arXiv: 2603.28565)
- Realtime-VLA FLASH (speculative inference for diffusion VLAs)
- DAWN / Pixel Motion Diffusion (CVPR 2026)

## 6. Compute Plan

**Hardware:** DGX Spark (GB10 Grace Blackwell, 128GB unified memory)

**Compute allocation:**
- Classical MPC: CPU-only (OSQP, CasADi) — negligible GPU
- VLA: 7B model inference on GB10 (quantized INT8/FP8)
- Diffusion policy: small diffusion models (10M-100M params) on GB10 GPU
- Ablations: ~50 runs × 100 episodes = manageable on single GPU

**Timeline:** ~8 weeks (part-time, single researcher)

## 7. Deliverables

1. `docs/comparison_report.md` — Full comparative analysis report
2. `results/tables/master_comparison.csv` — Aggregated metrics table
3. `figures/` — Publication-quality comparison plots
4. `src/` — Reproducible code for all baselines and ablations
5. `docs/position_paper.md` — Position paper draft: "Do we need diffusion in robotics? An empirical reappraisal"
