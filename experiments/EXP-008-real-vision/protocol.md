# EXP-008 — Real-Vision OOD Robustness

**Status:** Pre-registered  
**Skill:** experiment-protocol (design → controls → pre-register → execute → verify → refine)  
**Hypothesis under test:** `H_RVV` (World action models and iterative policies are more robust than pure VLA under real-robot vision perturbations)  
**Owner:** Gyanateet  
**Created:** 2026-08-11  

---

## 1. Hypothesis & Prediction

### Hypothesis

**`H_RVV`:** Under real-robot visual perturbations — lighting changes, occlusions, distractors, and background changes — **world action models (WAMs)** and **iterative policies (DDPM / MIP)** maintain higher success and better manifold adherence than **pure VLA**. The explicit latent rollout in WAMs and the iterative-denoising / noise-injection mechanism in DDPM/MIP act as implicit regularizers that keep OOD actions close to the expert manifold.

This experiment is the real-vision counterpart to `EXP-003` (OOD robustness on PushT). It replaces synthetic 2D perturbations with realistic 3D visual perturbations on real-physics manipulation tasks.

### Prediction (pre-registered, quantitative)

1. **Success ordering at perturbation level ≥ 2:**
   `success(WAM) ≥ success(MIP) ≈ success(DDPM) > success(SmallVLA)`.
2. **Manifold adherence ordering at level ≥ 2** (lower is better):
   `ManifoldAdh(WAM) ≤ ManifoldAdh(MIP) ≤ ManifoldAdh(DDPM) < ManifoldAdh(SmallVLA)`.
3. **Degradation slope:** The slope of success vs. perturbation level is shallowest for WAM, then MIP/DDPM, and steepest for SmallVLA.
   `slope(SmallVLA) > slope(DDPM) > slope(MIP) ≥ slope(WAM)`.
4. **Perturbation type ranking:** Background changes and occlusions cause the largest VLA drop; WAM is most robust to lighting and background changes because the latent dynamics model can compensate for missing or altered visual features.
5. **Interaction:** A two-way ANOVA (condition × perturbation level) shows a significant interaction (p < 0.05), indicating that conditions degrade differently across levels.

### Falsification conditions

- `H_RVV` is falsified if the ANOVA interaction is non-significant.
- `H_RVV` is falsified if SmallVLA is not the worst-performing condition at perturbation level ≥ 2 on both success and manifold adherence.
- `H_RVV` is falsified if WAM does not match or beat MIP/DDPM at level ≥ 2.

---

## 2. Design

| Element | Specification |
|---------|---------------|
| Design type | Two-way factorial, repeated measures: condition (4) × perturbation level (5) |
| Independent variables | Controller (4 levels), real-vision perturbation level (5 levels) |
| Benchmarks | ManiSkill3 `PickCube`, `PushCube` and/or robosuite `Lift` rendered with a realistic camera model |
| Unit of analysis | One evaluation episode |
| Pairing | Same base episode seeds across conditions and perturbation levels |
| Blinding | Scripted success predicate; manifold score computed offline from logged actions |
| Randomization | Episode order shuffled per (seed, level); condition order counterbalanced |

### Procedure (execute phase)

1. Set up the real-physics benchmark with a realistic camera (intrinsics, noise, field-of-view, depth of field).
2. Train each condition once per seed on the **clean (ID)** training set:
   - C1: SmallVLA.
   - C2: DDPM policy.
   - C3: MIP (2-step + noise).
   - C4: World Action Model (small, from scratch) from `EXP-005`.
   - Perturbations are **not** present during training (true OOD test).
3. Save checkpoints + config snapshots to `outputs/<condition>/seed_<s>/`.
4. For each (condition, perturbation level, seed), roll out 100 evaluation episodes using the fixed episode seed file (`data/eval_seeds_exp008.json`) with the level's perturbation applied at render time.
5. Record per-episode: success, manifold adherence score, action trajectory, perturbation level, condition, seed, config hash, git commit.
6. Aggregate to per-(condition, level, seed) means.

### Verify phase

- **Perturbation integrity:** For each level, visually inspect 5 rendered frames to confirm the perturbation is actually applied (lighting, occlusion, distractor, background). Save reference images.
- **Manifold-score validation:** Recompute manifold adherence on the level-0 expert data; expert actions must score ≈ 0. If not, re-scale the k-NN metric before analysis.
- **Level-0 consistency:** At level 0 (clean), all conditions should be within 5 pp of their `EXP-005` success rate on the same task. If not, flag training/eval divergence.
- **Determinism spot-check:** Re-run 1 seed of WAM and SmallVLA at level 0 and level 4 with fixed seeds; success must match within ±2 pp.

### Refine phase

- If level-0 success deviates > 5 pp from `EXP-005` for a condition, re-check training data and evaluation pipeline.
- If perturbation integrity fails for a level, fix the renderer and re-run that level only.
- If WAM underperforms DDPM/MIP, inspect whether the WAM latent rollout is too short or whether the observation encoder is overfit to clean images.

---

## 3. Primary Outcome

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Success rate at each perturbation level** (primary) | Fraction of 100 episodes achieving the task goal per (condition, level, seed) | Binary per episode, averaged; mean ± std over 5 seeds, per level |

Task-success predicates:

- **`PickCube`:** cube grasped and lifted ≥ 5 cm; held for ≥ 5 steps.
- **`PushCube`:** cube pushed into the target zone and remains there for ≥ 5 steps.
- **`Lift`:** object lifted ≥ 5 cm above the table and held for ≥ 5 steps.

---

## 4. Secondary Outcomes

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Manifold adherence score** | Mean k-NN distance (k=5) from policy actions to the expert action set | Computed offline from logged action trajectories; lower = better |
| **Degradation slope** | Linear regression slope of success rate over perturbation level (0–4), per condition | Per (condition, seed); averaged over seeds |
| **Per-perturbation robustness** | Success drop from level 0 for each perturbation type | Δsuccess per condition per perturbation |
| **Inference latency** | Wall-clock time observation → action | Median + p95 over 1000 timed calls |
| **OOD detection score** (exploratory) | Classifier confidence / likelihood ratio for detecting perturbed frames | Per episode; higher = better at detecting distribution shift |

---

## 5. Conditions Table

| ID | Condition | Iterative? | Noise | World model? | Notes |
|----|-----------|-----------|-------|--------------|-------|
| C1 | **SmallVLA** | No (VLM forward + regression head) | No | No | Pure VLA; expected to degrade most under visual OOD |
| C2 | **DDPM Policy** | Yes (T=50 or T=100) | Yes (DDPM schedule) | No | Standard diffusion GCP from `EXP-001`/`EXP-005` |
| C3 | **MIP** | Yes (T=2) | Yes | No | Minimal iterative policy; tests whether iterative compute + noise is enough |
| C4 | **World Action Model** | Yes (latent rollout) | Optional | Yes | Small WAM from `EXP-005`; tests whether explicit future dynamics improves robustness |

**Held constant across conditions:**
- Same observation preprocessing (resize, normalize, real-robot vision pipeline).
- Same training data (ID only; no perturbation augmentation).
- Same training budget and backbone parameter budget.
- Same action space and control frequency (10 Hz).
- Same episode seeds across conditions and levels (paired factorial).
- Same success and manifold reference set (expert demonstrations, committed and hashed).

---

## 6. Conditions Table — Perturbation Levels

| Level | Name | Description |
|-------|------|-------------|
| 0 | **Clean RGB** | Default renderer, default lighting, no occluders/distractors/background change. In-distribution baseline. |
| 1 | **Lighting changes** | Randomize light intensity (±40%), color temperature (±30%), and direction (±30° azimuth). |
| 2 | **Occlusions** | Place 1–3 semi-transparent or opaque occluders between camera and scene, covering up to 30% of the object. |
| 3 | **Distractors** | Add 1–5 novel objects (not in training) to the workspace, with random pose and color. |
| 4 | **Background changes** | Replace the background with real laboratory / kitchen / workshop images or randomized textures. |

Perturbations are applied **at evaluation time only**. Training uses only clean level-0 data.

---

## 7. Controls & Ablations

### Controls (fairness)

- **Training on ID data only:** No perturbation augmentation at training time, so the test is a true OOD robustness test.
- **Same backbone budget:** C2, C3, and C4 share the same backbone family and parameter budget where possible.
- **Same VLM encoder:** C1 and C4 use the same vision-language encoder to isolate the world-model/action component.
- **Same episode seeds and perturbation parameters across conditions and levels.**
- **Same success and manifold reference set.**

### Ablations embedded

- **VLA vs iterative:** C1 (SmallVLA) vs C2 (DDPM) vs C3 (MIP) tests the effect of iterative compute + noise under real vision OOD.
- **World model benefit:** C4 (WAM) vs C3 (MIP) tests whether explicit future dynamics adds robustness beyond iterative denoising.
- **GCP variant:** C2 (DDPM) vs C3 (MIP) replicates `EXP-001` in a real-vision setting.

### Negative / positive controls

- **SmallVLA (C1):** negative control; predicted worst OOD robustness.
- **MIP (C3):** positive control for the iterative-compute mechanism.
- **Level 0 (clean):** positive control; all conditions should match `EXP-005` level-0 performance.

---

## 8. Data

### Sample size

- 100 episodes per (condition, level, seed) × 5 seeds × 4 conditions × 5 levels × 2 benchmarks = **20,000 episodes** total.
- Quick-run option: 50 episodes per (condition, level, seed) × 5 seeds × 4 conditions × 5 levels × 1 benchmark = **5,000 episodes**.

### Seeds

`[0, 1, 2, 42, 123]` — fixed across conditions, levels, and benchmarks. Episode initial states from `data/eval_seeds_exp008.json` (committed, hashed).

### Power analysis (pre-registered)

- Two-way ANOVA (condition × level) on per-seed success rates. Interaction test at α=0.05.
- With 4 conditions × 5 levels × 5 seeds, MDE for the interaction ≈ 5 pp at power 0.80.
- Pairwise condition comparisons at each level: Wilcoxon with Bonferroni over 4 conditions → 6 pairs × 5 levels = 30 tests (α ≈ 0.0017).
- If observed std > 12 pp, flag as underpowered and record.

### Data storage

```
experiments/EXP-008-real-vision/outputs/
├── <condition>/
│   └── level_<L>/
│       └── seed_<s>/
│           ├── config.yaml
│           ├── env_info.json
│           ├── checkpoint.pt
│           ├── episodes.jsonl          # success, manifold score, actions
│           ├── sample_frames/          # perturbation-integrity visual check
│           └── metrics_summary.json
└── analysis/
    ├── anova_results.json
    ├── degradation_slopes.csv
    ├── manifold_adherence.csv
    ├── per_perturbation_robustness.csv
    └── figures/
```

---

## 9. Analysis Plan

### Primary analysis

1. **Two-way ANOVA:** Condition × level on success rate; test the interaction effect (pre-registered).
2. **Pairwise comparisons:** Wilcoxon signed-rank at each perturbation level, with Bonferroni correction across the 6 pairwise condition comparisons per level.
3. **Degradation slope:** Linear regression of success vs. level (0–4) per condition; compare slopes across conditions.
4. **Manifold adherence:** Same two-way ANOVA and pairwise tests on the manifold adherence score.

### Secondary analysis

1. **Per-perturbation drops:** For each condition, compute `success_drop = success_level0 − success_levelX` for each perturbation type; rank conditions.
2. **Latency under perturbation:** Measure whether preprocessing for occlusions/distractors affects latency.
3. **OOD detection correlation:** If an OOD detector is implemented, correlate detector score with success drop.
4. **Generalization to real images:** If a real-image split exists, run the level-4 conditions on it and compare to synthetic background-change results.

### Figures

- `exp008_success_vs_level.pdf` — success rate vs. perturbation level, one line per condition.
- `exp008_manifold_vs_level.pdf` — manifold adherence vs. perturbation level.
- `exp008_degradation_slopes.pdf` — bar plot of degradation slopes per condition.
- `exp008_perturbation_heatmap.pdf` — heatmap of success drop by condition and perturbation type.

---

## 10. Links

- Real-robotics master plan: [`../../docs/real_robotics/README.md`](../../docs/real_robotics/README.md)
- Phase 2 roadmap: [`../../docs/real_robotics/phase2_roadmap.md`](../../docs/real_robotics/phase2_roadmap.md)
- EXP-005 WAM baseline: [`../EXP-005-world-models/protocol.md`](../EXP-005-world-models/protocol.md)
- EXP-003 OOD robustness: [`../EXP-003-ood-robustness/protocol.md`](../EXP-003-ood-robustness/protocol.md)
- Methodology: [`../../docs/methodology.md`](../../docs/methodology.md)
