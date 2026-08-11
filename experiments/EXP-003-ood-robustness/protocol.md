# EXP-003 — OOD Robustness

**Status:** Pre-registered
**Skill:** experiment-protocol (design → controls → pre-register → execute → verify → refine)
**Hypothesis under test:** H4 (noise + iterative compute → OOD robustness)
**Owner:** Gyanateet
**Created:** 2026-08-11

---

## 1. Hypothesis & Prediction

### Hypothesis
**H4:** Does noise injection + iterative compute improve out-of-distribution (OOD)
robustness? Specifically, do GCPs and MIP exhibit better *manifold adherence* (stay
close to the expert action manifold) under OOD observations than single-step
regression?

- **H4 (Simchowitz view):** Iterative compute + noise act as an implicit regularizer
  that keeps OOD actions near the expert manifold. Full DDPM and MIP should degrade
  *gracefully* under perturbation; Pure Regression should degrade *sharply*.
- **Null:** No difference in degradation profile across conditions.

### Prediction (pre-registered, quantitative)
1. **Manifold adherence ordering (predicted):** At perturbation level ≥ 2,
   `ManifoldAdh(Full DDPM) ≤ ManifoldAdh(MIP) < ManifoldAdh(Flow Matching) <
   ManifoldAdh(Regression)`, where lower = closer to expert manifold = better.
   (Full DDPM and MIP within 10% of each other.)
2. **Success-rate degradation slope:** The slope of success rate vs perturbation level
   is shallowest for Full DDPM and MIP, steepest for Regression. Predicted:
   `slope(Regression) > slope(Flow Matching) > slope(MIP) ≈ slope(Full DDPM)`.
3. **ANOVA:** A two-way ANOVA (condition × perturbation level) on success rate will
   show a significant interaction (p < 0.05), indicating conditions degrade
   differently across perturbation levels.
4. **Level-0 sanity:** At perturbation level 0 (ID), all conditions within 5 pp of
   their EXP-001 success rate (consistency check).

### Falsification conditions
- H4 falsified if the ANOVA interaction is non-significant AND the manifold-adherence
  ordering at level ≥ 2 does not hold (Regression not worst, or Full DDPM/MIP not
  best-two).
- H4 also falsified if `slope(Regression) ≤ slope(MIP)` (regression degrades no faster
  than the iterative+noise policy).

---

## 2. Design

| Element | Specification |
|---------|---------------|
| Design type | Two-way factorial, repeated measures: condition (4) × perturbation level (5) |
| Independent variables | Policy condition (4 levels), perturbation level (5 levels: 0–4) |
| Benchmark | PushT (image observation; perturbations applied to render) |
| Unit of analysis | One evaluation episode |
| Pairing | Same base episode seeds across conditions and perturbation levels |
| Blinding | Scripted success predicate; manifold score computed offline from logged actions |
| Randomization | Episode order shuffled per (seed, level); condition order counterbalanced |

### Procedure (execute phase)
1. Train each condition once per seed on the **in-distribution** training set (no
   perturbations during training). Save checkpoints to `outputs/<condition>/seed_<s>/`.
2. For each (condition, perturbation level, seed), roll out 100 evaluation episodes
   using `data/eval_seeds_exp003.json` with the level's perturbation applied at render
   time.
3. Record per-episode: success, manifold adherence score, action trajectory,
   perturbation level, config hash, git commit.
4. Aggregate per-(condition, level, seed).

### Verify phase
- **Perturbation integrity:** For each level, visually inspect 5 rendered frames to
  confirm the perturbation is actually applied (color shift, position offset, camera
  angle, combined). Log frames.
- **Manifold-score validation:** Recompute manifold adherence on the level-0 expert
  data; expert actions must score ≈ 0 (sanity). If not, the k-NN metric is mis-scaled;
  fix before analysis.
- Determinism spot-check on 1 seed of Full DDPM and Regression at level 0 and level 4.

### Refine phase
- If level-0 success deviates > 5 pp from EXP-001 for a condition, the training or eval
  pipeline diverged; re-check, re-train if needed, log amendment.
- If perturbation integrity fails for a level, fix the renderer and re-run that level
  only.

---

## 3. Primary Outcome

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Success rate at each perturbation level** (primary) | Fraction of 100 episodes achieving task goal per (condition, level, seed) | Binary per episode, averaged; mean ± std over 5 seeds, per level |

Task-success predicate (PushT): final pose within 0.05 m / 0.1 rad of target.

---

## 4. Secondary Outcomes

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Manifold adherence score** | Mean k-NN distance (k=5) from policy actions to the expert action set | Computed offline from logged action trajectories; lower = better |
| **Degradation slope** | Linear regression slope of success rate over perturbation level (0–4), per condition | Per (condition, seed); averaged over seeds |
| **Action KL** (exploratory) | KL(policy ‖ expert) per level | KDE estimate |

---

## 5. Conditions Table

| ID | Condition | Iterative? | Noise | Distribution fitting | Backbone | Role |
|----|-----------|-----------|-------|----------------------|----------|------|
| C1 | Regression | No (T=1) | No | No | Shared | H4 floor — should degrade worst |
| C2 | MIP | Yes (T=2) | Yes | No | Shared | H4 test — should match DDPM |
| C3 | Full DDPM | Yes (T=100) | Yes | Yes | Shared UNet | H4 test — should be best/tied |
| C4 | Flow Matching | Yes (T steps) | Yes | Yes (rectified flow) | Shared UNet | Comparison GCP variant |

**Held constant:** backbone (compute-matched), training data (ID only), training budget,
observation preprocessing, action space, control frequency (10 Hz), evaluation episodes.

---

## 6. Controls & Ablations

### Controls (fairness)
- All conditions trained on the **same in-distribution** data; perturbations appear only
  at evaluation (true OOD test, not augmentation).
- Same backbone parameter budget across C1–C4.
- Same episode seeds across conditions and levels (paired factorial).
- Manifold reference set = expert demonstrations (fixed, hashed).

### Ablations embedded
- **Ablate iterative+noise:** C3 (Full DDPM) → C1 (Regression) isolates the
  iterative+noise mechanism's contribution to robustness.
- **Ablate distribution fitting:** C3 (Full DDPM) → C2 (MIP) tests whether fitting
  matters for robustness or just iterative+noise (per H4, MIP ≈ DDPM).
- **GCP variant:** C3 (DDPM) vs C4 (Flow Matching) checks robustness generalizes across
  GCP families.

### Negative / positive controls
- **Regression (C1):** negative control; predicted worst OOD robustness.
- **Level 0 (ID):** positive control; all conditions should match EXP-001 success.

---

## 7. Data

### Sample size
- 100 episodes per (condition, level, seed) × 5 seeds × 4 conditions × 5 levels
  = **10,000 episodes** total.

### Perturbation levels

| Level | Perturbation | Description |
|-------|-------------|-------------|
| 0 | None (ID) | In-distribution test (baseline) |
| 1 | Object color | Change object texture/color |
| 2 | Object position | Shift object ±5 cm from training distribution |
| 3 | Camera viewpoint | Rotate camera ±10° |
| 4 | Combined | Levels 1+2+3 applied together |

### Seeds
`[0, 1, 2, 42, 123]` — fixed across conditions and levels. Episode initial states from
`data/eval_seeds_exp003.json` (committed, hashed).

### Power analysis (pre-registered)
- Two-way ANOVA (condition × level) on per-seed success rates. Interaction test at
  α=0.05. With 4×5 cells × 5 seeds, MDE for the interaction ≈ 5 pp at power 0.80.
- Pairwise condition comparisons at each level: Wilcoxon with Bonferroni over
  4 conditions → 6 pairs × 5 levels = 30 tests (α ≈ 0.0017).

### Data storage
```
experiments/EXP-003-ood-robustness/outputs/
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
    └── figures/
```

---

## 8. Analysis Plan

### Primary analysis
1. **Two-way ANOVA:** condition (4) × perturbation level (5) on per-seed success rate.
   Report F-statistics, p-values, η² for main effects and interaction.
2. **Interaction test (H4 gate):** A significant interaction (p < 0.05) is required to
   support H4 (conditions degrade differently). Non-significant interaction → H4 not
   supported.
3. **Post-hoc pairwise:** At each perturbation level, Wilcoxon signed-rank on per-seed
   success across the 6 condition pairs, Bonferroni-corrected over 30 tests.
4. **Degradation slope:** Per (condition, seed), fit `success = a + b·level`; report
   mean slope ± std. Apply the ordering test from §1.2.
5. **Manifold adherence:** Per (condition, level, seed) mean k-NN distance; apply the
   ordering test from §1.1 at levels ≥ 2.

### Secondary analysis
- Plot success rate vs perturbation level per condition (degradation curves).
- Plot manifold adherence vs perturbation level per condition.
- Correlate manifold adherence with success (across levels) per condition.

### Reporting
- ANOVA table, all p-values, η², slopes ± std, manifold scores.
- Raw per-episode JSONL committed.
- Figures from committed results only.
- Analysis script: `experiments/EXP-003-ood-robustness/analyze.py` (committed).

### Software
- `scipy.stats.f_oneway` / `statsmodels.anova` for ANOVA.
- `scipy.stats.wilcoxon`, `statsmodels` for Bonferroni.
- `sklearn.neighbors.NearestNeighbors` for k-NN manifold score.

---

## 9. Stopping Rules

1. **Manifold-score sanity stop:** If expert actions do not score ≈ 0 on the manifold
   metric at level 0, STOP. The metric is mis-scaled; fix before any analysis. Do not
   proceed with a broken metric.
2. **Perturbation-integrity stop:** If visual inspection shows a perturbation level is
   not applied (or applied inconsistently), stop that level, fix the renderer, re-run.
3. **Level-0 consistency stop:** If a condition's level-0 success deviates > 5 pp from
   its EXP-001 value, stop that condition; re-check training/eval pipeline, re-train if
   needed, log amendment.
4. **Training failure stop:** NaN/divergence → re-run once with fresh init; if it fails
   again, exclude and report.
5. **Resource stop:** If wall-clock exceeds 2× budget estimate, stop, log partial
   results, re-plan. No silent extension.
6. **No peeking:** No hypothesis testing until all 5 seeds × 100 episodes complete for a
   given (condition, level). Loss/manifold curves may be monitored for debugging only.

---

## 10. Pre-registration Checklist

| # | Item | Status | Evidence / Location |
|---|------|--------|---------------------|
| 1 | Hypothesis stated (H4) | ✅ Done | §1 |
| 2 | Quantitative predictions with thresholds | ✅ Done | §1 (ordering, slopes, ANOVA) |
| 3 | Falsification conditions explicit | ✅ Done | §1 |
| 4 | Primary outcome defined | ✅ Done | §3 (success per level) |
| 5 | Conditions table complete (4 conditions) | ✅ Done | §5 |
| 6 | Perturbation levels defined (0–4) | ✅ Done | §7 |
| 7 | Controls & ablations specified | ✅ Done | §6 |
| 8 | Sample size + power analysis | ✅ Done | §7 (10,000 episodes; MDE ~5 pp) |
| 9 | Seeds fixed & documented | ✅ Done | §7 ([0,1,2,42,123]) |
| 10 | Analysis plan (ANOVA + post-hoc) | ✅ Done | §8 |
| 11 | Stopping rules | ✅ Done | §9 |
| 12 | Decision rules pre-specified | ✅ Done | §1, §8.2 |
| 13 | Data & artifact storage plan | ✅ Done | §7 |
| 14 | Environment lock referenced | ✅ Done | `../environment.lock` |
| 15 | Cross-experiment consistency (EXP-001) | ✅ Done | §9 rule 3 |
| 16 | Code commit recorded at run time | ⏳ Pending execution | `env_info.json` |
| 17 | No look-ahead at outcomes before full run | ✅ Committed | §9 rule 6 |

**Pre-registration status:** COMPLETE. No changes to hypotheses, conditions, perturbation
levels, sample size, or analysis plan after the first evaluation episode. Deviations
logged in `outputs/protocol_amendments.md`.

---

## 11. Execute / Verify / Refine Log

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Design | ✅ Complete | 2026-08-11 | This document |
| Pre-register | ✅ Complete | 2026-08-11 | §10 checklist done |
| Execute | ⏳ Not started | — | Run via `../run_all.sh` (exp 3) |
| Verify | ⏳ Not started | — | Perturbation-integrity + manifold-sanity audit |
| Refine | ⏳ Not started | — | Triggered on metric/perturbation/consistency failure |

---

## 12. Links

- Comparison plan: `../../docs/comparison_plan.md` (§4.4)
- Research questions: `../../docs/research_questions.md` (H4)
- Methodology: `../../docs/methodology.md` (§OOD Perturbation Levels)
- Depends on: `../EXP-001-mechanism-ablation/protocol.md` (level-0 consistency)
- Environment lock: `../environment.lock`
- Runner: `../run_all.sh`
