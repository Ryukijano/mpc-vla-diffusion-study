# EXP-006 — Sim-to-Real Domain Gap

**Status:** Pre-registered  
**Skill:** experiment-protocol (design → controls → pre-register → execute → verify → refine)  
**Hypothesis under test:** `H_S2R` (Domain randomization and photorealistic rendering reduce the sim-to-real success gap)  
**Owner:** Gyanateet  
**Created:** 2026-08-11  

---

## 1. Hypothesis & Prediction

### Hypothesis

**`H_S2R`:** Policies trained with **domain randomization (DR)** and/or **photorealistic rendering** narrow the sim-to-real visual gap compared to a baseline trained only in the default simulator. Additionally, a **synthetic-to-real image transfer** layer can further reduce the gap for pure vision-conditioned policies.

This experiment tests one of the central risks of the real-robotics phase: vision-conditioned methods (VLA, diffusion, WAM) may perform well in sim but collapse when visual statistics change in the real world.

### Prediction (pre-registered, quantitative)

1. **Baseline gap:** The no-DR condition (C1) has a sim-to-real success gap > 20 pp on the primary task.
2. **Domain randomization:** Training with DR (C2) reduces the gap to ≤ 15 pp.
3. **Photorealistic rendering:** Training with photorealistic / PBR rendering (C3) reduces the gap to ≤ 12 pp.
4. **Image transfer:** Adding a synthetic-to-real image transfer / adaptation module (C4) reduces the gap to ≤ 10 pp.
5. **Real-pretrained VLA:** A VLA pretrained on real internet-scale vision-language data (C5) has the smallest gap (≤ 8 pp) but may have lower absolute sim success.

### Falsification conditions

- `H_S2R` is falsified if none of C2–C4 reduce the sim-to-real gap by ≥ 5 pp relative to C1.
- If the real/synthetic-real test set cannot be constructed, the experiment is paused until `VLA-REPLICA`, `RoboDojo-Real`, or an equivalent real-image test split is available.

---

## 2. Design

| Element | Specification |
|---------|---------------|
| Design type | Two-way factorial, repeated measures: condition (5) × test domain (2: sim, real/synthetic-real) |
| Independent variables | Controller / training pipeline (5 levels), test domain (2 levels) |
| Benchmarks | ManiSkill3 `PickCube`, `PushCube`; Isaac Sim equivalent if photorealistic rendering is required; real/synthetic-real test split from `VLA-REPLICA` or `RoboDojo-Real` if available |
| Unit of analysis | One evaluation episode |
| Pairing | Same base episode seeds in sim and real/synthetic-real domains per condition |
| Blinding | Scripted task-success predicate (no human rating) |
| Randomization | Episode order shuffled per (seed, domain); condition order counterbalanced |

### Procedure (execute phase)

1. Set up the shared real-robot vision preprocessing pipeline and a **target real/synthetic-real visual test split**.
   - Primary source: `VLA-REPLICA` or `RoboDojo-Real` real-image split (if available).
   - Fallback: synthetic-to-real image transfer test set built by compositing real background/camera images onto ManiSkill3 / Isaac Sim scenes.
2. Train each condition once per seed in simulation:
   - C1: default renderer, no domain randomization.
   - C2: default renderer + domain randomization (lighting, texture, background, camera pose, object color).
   - C3: photorealistic renderer (RTX / PBR in Isaac Sim or ManiSkill3) + light DR.
   - C4: default renderer + DR + synthetic-to-real image adaptation module (e.g., CycleGAN / AdaIN / latent domain adversarial).
   - C5: real-pretrained VLA (OpenVLA or SmallVLA variant) used zero-shot, optionally with a small sim fine-tune.
3. Save checkpoints + config snapshots to `outputs/<condition>/seed_<s>/`.
4. For each (condition, seed), roll out 100 episodes in **sim** and 100 episodes in **real/synthetic-real** using paired episode seeds (`data/eval_seeds_exp006.json`).
5. Record per-episode: success, task metric, domain, inference latency, visual feature distribution stats, config hash, git commit.
6. Aggregate to per-(condition, domain, seed) means.

### Verify phase

- **Visual distribution check:** Compute FID / LPIPS between sim and real/synthetic-real image sets for each condition. Confirm the visual gap is reduced by C2–C4 relative to C1.
- **DR integrity:** Visually inspect 10 training frames per condition to confirm that the intended randomizations are active.
- **Determinism spot-check:** Re-run 1 seed of C3 and C5 on sim with fixed seeds; success must match within ±2 pp.
- **Sanity audit:** 5 random episodes per condition checked for NaN actions, valid episode length, and plausible task progression.

### Refine phase

- If the sim-to-real gap for C4 is not smaller than C2, inspect the image-transfer module for overfitting to the synthetic source domain.
- If the real-image split is too small (< 100 episodes), collect more real images or increase synthetic-real compositing diversity; log the amendment.

---

## 3. Primary Outcome

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Sim-to-real success gap** (primary) | `success_sim − success_real` in percentage points per (condition, seed) | Difference of mean success rate over 100 sim episodes and 100 real/synthetic-real episodes; reported as mean ± std over 5 seeds |

Task-success predicates:

- **`PickCube`:** cube grasped and lifted ≥ 5 cm; held for ≥ 5 steps.
- **`PushCube`:** cube pushed into the target zone and remains there for ≥ 5 steps.

The lower (more positive) the gap, the better the sim-to-real transfer.

---

## 4. Secondary Outcomes

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Sim success rate** | Fraction of episodes successful in the source sim domain | Binary per episode, averaged |
| **Real/synthetic-real success rate** | Fraction of episodes successful in the target domain | Binary per episode, averaged |
| **FID (Fréchet Inception Distance)** | Distance between sim and real/synthetic-real image distributions | Pre-trained Inception features; lower = closer distributions |
| **LPIPS (perceptual distance)** | Perceptual distance between paired sim and real frames | Alex/VGG feature distance; lower = more visually similar |
| **Per-perturbation robustness** | Success under each DR component (lighting, texture, background, camera) | Ablate DR components in C2; report success |
| **Inference latency** | Wall-clock time observation → action | Median + p95 over 1000 timed calls |
| **Training wall-clock** | Total training time per condition | Hours; for compute planning |

---

## 5. Conditions Table

| ID | Condition | Training Visual Domain | Image Transfer | Notes |
|----|-----------|------------------------|----------------|-------|
| C1 | **Sim-only baseline** | Default sim renderer, no DR | None | Upper-bound sim performance; expected large real gap |
| C2 | **Domain randomization (DR)** | Default renderer + lighting/texture/background/camera/object DR | None | Standard sim-to-real regularization |
| C3 | **Photorealistic rendering** | RTX/PBR photorealistic renderer (Isaac Sim / ManiSkill3) + light DR | None | Close the visual realism gap at the source |
| C4 | **DR + synthetic-to-real transfer** | Default renderer + DR | CycleGAN / AdaIN / latent domain adaptation at training or test time | Adapt visual features to real/synthetic-real target |
| C5 | **Real-pretrained VLA** | Real internet vision-language pretraining; optionally fine-tuned on sim | None (pretraining acts as implicit transfer) | Zero-shot or few-shot real performance |

**Held constant across conditions:**
- Same base policy architecture when possible (C1–C4 use the same SmallVLA or diffusion backbone; C5 is fixed by pretraining choice).
- Same training data size and number of gradient steps.
- Same real-robot vision preprocessing at test time.
- Same evaluation seeds in sim and real/synthetic-real.
- Same task-success predicates.

---

## 6. Controls & Ablations

### Controls (fairness)

- **Same seed pairing:** Each episode is evaluated in both sim and real/synthetic-real with the same initial object pose and robot state.
- **Same test-time preprocessing:** All conditions receive identical RGB/Depth preprocessing.
- **Same action space and control frequency:** 10 Hz for learning-based conditions.
- **Same DR budget:** C2–C4 use the same DR parameter ranges unless the condition is specifically ablating one component.
- **Same real-image split:** C1–C5 evaluated on identical real/synthetic-real episodes.

### Ablations embedded

- **DR component ablation:** Within C2, ablate lighting, texture, background, camera, and object randomization one at a time. Identify which component drives transfer.
- **Photoreal vs DR:** C2 vs C3 tests whether realism at the source (photoreal) beats training-time randomization.
- **Source DR + transfer vs photoreal:** C3 vs C4 tests whether explicit domain adaptation beats photorealistic source rendering.
- **Pretraining vs augmentation:** C5 vs C4 tests whether real VLM pretraining is more effective than sim DR + transfer.

### Negative / positive controls

- **C1 (sim-only):** negative control; expected to show the largest sim-to-real gap.
- **C3 (photoreal):** positive control for visual fidelity.
- **C5 (real-pretrained VLA):** positive control for real-world visual generalization.

---

## 7. Data

### Sample size

- 100 episodes per (condition, domain, seed) × 5 seeds × 5 conditions × 2 domains × 2 benchmarks = **10,000 episodes** total.

### Seeds

`[0, 1, 2, 42, 123]` — fixed across conditions, domains, and benchmarks. Episode initial states from `data/eval_seeds_exp006.json` (committed, hashed).

### Power analysis (pre-registered)

- Paired Wilcoxon on the sim-vs-real gap per condition, α=0.05 with Bonferroni over the 10 pairwise condition comparisons. Effective α ≈ 0.005.
- Minimal detectable gap difference ≈ 7 pp at power 0.80 given 5 seeds × 100 episodes and assumed within-seed correlation ρ=0.4.
- If observed gap std > 15 pp, flag as underpowered and record.

### Data storage

```
experiments/EXP-006-sim-to-real/outputs/
├── <condition>/
│   └── <benchmark>/
│       └── seed_<s>/
│           ├── config.yaml
│           ├── env_info.json
│           ├── checkpoint.pt
│           ├── sim/
│           │   └── episodes.jsonl
│           ├── real_or_synthetic_real/
│           │   └── episodes.jsonl
│           ├── sample_frames/           # sim/real frame pairs
│           └── metrics_summary.json
└── analysis/
    ├── sim_to_real_gaps.csv
    ├── success_by_domain.csv
    ├── fid_lpips.csv
    ├── dr_ablation.csv
    └── figures/
```

---

## 8. Analysis Plan

### Primary analysis

1. **Sim-to-real gap per condition:** Compute `gap = success_sim − success_real` for each (condition, seed); report mean ± std over seeds.
2. **Pairwise comparisons:** Wilcoxon signed-rank on the gap, comparing each condition to C1 and to each other, with Bonferroni correction across 10 pairs.
3. **Domain × condition interaction:** Two-way repeated-measures ANOVA on success rate (condition × domain) to confirm that conditions degrade differently across domains.

### Secondary analysis

1. **Correlation:** Pearson/Spearman correlation between FID/LPIPS visual gap and success gap across conditions.
2. **DR component ablation:** Report success gap for each DR component subset.
3. **Latency vs. gap:** Check whether image transfer modules (C4) add unacceptable latency.
4. **Pretraining effect:** Compare C5 zero-shot vs fine-tuned (if both are run).

### Figures

- `exp006_gap_by_condition.pdf` — sim-to-real gap per condition.
- `exp006_sim_vs_real.pdf` — paired scatter / bar plot of sim and real success.
- `exp006_fid_vs_gap.pdf` — FID vs. success gap.
- `exp006_dr_ablation.pdf` — gap for each DR component.

---

## 9. Links

- Real-robotics master plan: [`../../docs/real_robotics/README.md`](../../docs/real_robotics/README.md)
- Phase 2 roadmap: [`../../docs/real_robotics/phase2_roadmap.md`](../../docs/real_robotics/phase2_roadmap.md)
- Comparison plan: [`../../docs/comparison_plan.md`](../../docs/comparison_plan.md)
- Methodology: [`../../docs/methodology.md`](../../docs/methodology.md)
