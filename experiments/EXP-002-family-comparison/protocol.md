# EXP-002 — Three-Family Comparison

**Status:** Pre-registered
**Skill:** experiment-protocol (design → controls → pre-register → execute → verify → refine)
**Hypotheses under test:** H5 (MPC niche), H6 (VLA niche), H7 (hybrid), H8 (Pareto, partial)
**Owner:** Gyanateet
**Created:** 2026-08-11

---

## 1. Hypothesis & Prediction

### Hypothesis
**Head-to-head comparison** of the three controller families — Classical MPC, VLA, and
Diffusion/GCP (plus hybrids) — across benchmarks chosen to span each family's strength
zone. This experiment addresses:

- **H5 (MPC niche):** Classical MPC dominates on constraint-heavy / real-time tasks
  (cluttered reaching) and fails on vision/language tasks.
- **H6 (VLA niche):** VLA dominates on language/visual-generalization tasks and
  struggles on latency / precision.
- **H7 (hybrid):** Hybrid approaches (Diffusion Warm-Start MPC) outperform pure
  approaches on tasks needing *both* constraint satisfaction and learned priors.
- **H8 (partial):** Establishes the latency–performance points that EXP-004 turns into a
  Pareto frontier.

### Prediction (pre-registered, quantitative)
1. **MPC dominance zone:** On 2D Reaching (cluttered), `success(Collision-Free MPC) >
   success(DDPM Policy)` and `> success(SmallVLA)` by ≥ 5 pp, with significant paired
   test.
2. **VLA dominance zone:** On PushT (image+language variant), `success(SmallVLA) >
   success(DDPM Policy)` by ≥ 5 pp (semantic generalization), and SmallVLA latency >
   5× MPC latency.
3. **Diffusion middle:** On 2D Reaching (clear), `success(DDPM Policy) ≥ success(MIP)`
   and both within 3 pp of Nonlinear MPC (no family dominates a "fair" task).
4. **Hybrid win:** On 2D Reaching (cluttered), `success(Diffusion Warm-Start MPC) >
   max(success(Collision-Free MPC), success(DDPM Policy))` by ≥ 3 pp.
5. **Latency ordering (predicted):** Linear MPC < Nonlinear MPC < Collision-Free MPC <
   MIP < DDPM < Flow Matching < Diffusion Warm-Start MPC < SmallVLA.

### Falsification conditions
- H5 falsified if Collision-Free MPC is NOT the top performer on cluttered reaching by a
  significant margin, OR if it does NOT win on the clear-reaching latency dimension.
- H6 falsified if SmallVLA does NOT lead on the language-conditioned PushT variant by a
  significant margin.
- H7 falsified if Diffusion Warm-Start MPC does not beat *both* of its pure components
  (Collision-Free MPC and DDPM Policy) on cluttered reaching.

---

## 2. Design

| Element | Specification |
|---------|---------------|
| Design type | Within-subject (same evaluation episodes across all 8 conditions), repeated measures |
| Independent variable | Controller (8 levels, see Conditions table) |
| Benchmarks | PushT, 2D Reaching (clear), 2D Reaching (cluttered) |
| Unit of analysis | One evaluation episode |
| Pairing | All conditions evaluated on identical episode seeds per benchmark |
| Blinding | Scripted task-success predicate (no human rating) |
| Randomization | Episode order shuffled per seed; condition order counterbalanced |

### Procedure (execute phase)
1. Prepare/train each controller once per seed (learning-based) or configure once
   (model-based MPC). Save configs + checkpoints to `outputs/<condition>/seed_<s>/`.
2. For each (condition, benchmark, seed), roll out 100 evaluation episodes from the
   fixed seed file (`data/eval_seeds_exp002.json`).
3. Record per-episode: success, path length, collision flag, inference latency, action
   trajectory, config hash, git commit.
4. Aggregate per-(condition, benchmark, seed); do not collapse across seeds before
   testing.

### Verify phase
- Determinism spot-check: re-run 1 seed of Linear MPC and DDPM Policy with
  `cudnn.deterministic=True`; success must match within ±2 pp.
- Sanity audit: 5 random episodes per (condition, benchmark) checked for NaN actions,
  valid episode length, plausible path length.
- Cross-check: MPC controllers must report near-zero constraint violation on clear
  reaching (positive control for the solver).

### Refine phase
- If a family is systematically below the Pure-Regression floor from EXP-001 on a
  benchmark, flag the implementation (likely a bug, not a scientific result), debug,
  and re-run that family only. Log the amendment.

---

## 3. Primary Outcome

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Success rate** (primary) | Fraction of 100 episodes achieving the task goal per (condition, benchmark, seed) | Binary per episode, averaged; mean ± std over 5 seeds |

Task-success predicates:
- **PushT:** final pose within 0.05 m / 0.1 rad of target.
- **2D Reaching (clear):** end-effector reaches within 0.02 m of goal, holds 5 steps,
  no collisions.
- **2D Reaching (cluttered):** same as clear + zero collisions with obstacles.

---

## 4. Secondary Outcomes

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Inference latency** | Wall-clock observation → action | Median + p95 over 1000 timed calls (100 warmup) |
| **Path length** | Summed end-effector travel distance per episode | Per episode, averaged |
| **Collision rate** | Fraction of episodes with ≥1 collision (cluttered only) | Binary per episode |
| **Constraint violation rate** | % timesteps violating safety constraints | Per-step binary, averaged |

---

## 5. Conditions Table

| ID | Condition | Family | Iterative? | Constraints | Language | Notes |
|----|-----------|--------|-----------|-------------|----------|-------|
| C1 | Linear MPC | MPC | No (QP, 1 solve) | Hard (linearized) | No | Low-latency floor |
| C2 | Nonlinear MPC | MPC | No (SQP/iLQR) | Hard | No | Mid-latency MPC |
| C3 | Collision-Free MPC | MPC | No (SDF constraints) | Hard (collision) | No | MPC strength zone |
| C4 | SmallVLA | VLA | No (regression head) | None | Yes | Compute-matched VLA |
| C5 | DDPM Policy | Diffusion | Yes (T=100) | None | No | Standard GCP |
| C6 | Flow Matching Policy | Diffusion | Yes (T steps) | None | No | Rectified-flow GCP |
| C7 | MIP | Diffusion | Yes (2-step + noise) | None | No | From EXP-001 |
| C8 | Diffusion Warm-Start MPC | Hybrid | Yes (diffusion) + MPC | Hard (MPC refine) | No | H7 test |

**Held constant:** observation preprocessing, action space, control frequency (10 Hz for
learning-based; native for MPC but logged), evaluation episodes, training data & budget
(for C4–C7).

---

## 6. Controls & Ablations

### Controls (fairness)
- Same observation preprocessing across all conditions.
- Same action space (end-effector delta, 2D).
- Learning-based methods (C4–C8) share training data and compute-matched training
  budget.
- MPC methods (C1–C3, C8-MPC-part) use the same dynamics model and solver tolerances.
- Fixed episode seeds shared across all conditions (paired design).
- Latency measured under identical protocol (DGX Spark GB10, fixed GPU clock, 100
  warmup, 1000 timed calls).

### Ablations embedded
- **MPC family gradient:** C1 (linear) → C2 (nonlinear) → C3 (collision-free) shows the
  cost/benefit of adding constraint sophistication.
- **Diffusion family gradient:** C5 (DDPM) → C6 (flow) → C7 (MIP) shows the effect of
  generative vs minimal-iterative (links to EXP-001).
- **Hybrid decomposition:** C8 vs C3 and C5 isolates the warm-start contribution.

### Negative / positive controls
- **Linear MPC (C1):** low-latency positive control; should dominate on latency,
  underperform on vision tasks.
- **MIP (C7):** carries forward from EXP-001 as the "minimal diffusion" anchor.

---

## 7. Data

### Sample size
- 100 episodes per condition per seed × 5 seeds × 8 conditions × 3 benchmarks
  = **12,000 episodes** total.

### Seeds
`[0, 1, 2, 42, 123]` — fixed across all conditions and benchmarks. Episode initial
states from `data/eval_seeds_exp002.json` (committed, hashed).

### Power analysis (pre-registered)
- Paired Wilcoxon, α=0.05 with Bonferroni over the 28 pairwise comparisons per
  benchmark (8 conditions → C(8,2)=28). Effective α ≈ 0.0018 per test.
- MDE ~6 pp at power 0.80 given 5 seeds × 100 episodes, ρ=0.3. If observed std > 13 pp,
  flag as underpowered and record (do not silently add seeds).

### Data storage
```
experiments/EXP-002-family-comparison/outputs/
├── <condition>/
│   └── <benchmark>/
│       └── seed_<s>/
│           ├── config.yaml
│           ├── env_info.json
│           ├── checkpoint.pt (learning-based only)
│           ├── episodes.jsonl
│           └── metrics_summary.json
└── analysis/
    ├── comparison_table.csv
    ├── wilcoxon_results.json
    └── figures/
```

---

## 8. Analysis Plan

### Primary analysis
1. **Per-benchmark success-rate table:** mean ± std per condition, per benchmark.
2. **Pairwise comparisons:** Wilcoxon signed-rank test on per-seed success rates for all
   28 condition pairs, per benchmark.
3. **Multiple-comparison correction:** Bonferroni over 28 tests per benchmark
   (α = 0.05/28 ≈ 0.0018).
4. **Effect size:** rank-biserial correlation + 95% CI per comparison.
5. **Niche decision rules:** Apply pre-registered thresholds from §1 (5 pp / 3 pp gaps,
   significance).

### Secondary analysis
- Latency: median + p95 per condition; latency ranking vs prediction (§1.5).
- Path length & collision rate: per condition on cluttered reaching.
- Cross-benchmark profile: radar/spider chart per family (success on each benchmark).
- Pareto scatter: success vs latency points (input to EXP-004).

### Reporting
- Mean ± std, all p-values, effect sizes, CIs.
- Raw per-episode JSONL committed.
- Figures from committed results only.
- Analysis script: `experiments/EXP-002-family-comparison/analyze.py` (committed).

### Software
- `scipy.stats.wilcoxon`, `statsmodels` (Bonferroni, CIs).

---

## 9. Stopping Rules

1. **Implementation-failure stop:** If any condition fails to train or its MPC solver
   fails to converge on > 20% of clear-reaching episodes, stop that condition, log the
   failure, re-run once with adjusted solver tolerances / fresh init. If it fails again,
   exclude and report explicitly.
2. **Resource stop:** If wall-clock exceeds 2× the budget estimate (see README
   timeline), stop, log partial results, re-plan. No silent extension.
3. **No peeking:** No hypothesis testing until all 5 seeds × 100 episodes complete for a
   given (condition, benchmark). Training/loss and solver-convergence curves may be
   monitored for debugging only.
4. **Cross-experiment gate:** EXP-002 depends on EXP-001's MIP (C7) and replication
   passing. If EXP-001's replication gate fails, defer C7 results and run the remaining
   7 conditions; mark C7 as pending.
5. **Futility (optional):** After 3 seeds, if a predicted dominance (§1) is already
   reversed by > 15 pp on a benchmark, one may stop that comparison early and report;
   otherwise complete all 5 seeds.

---

## 10. Pre-registration Checklist

| # | Item | Status | Evidence / Location |
|---|------|--------|---------------------|
| 1 | Hypotheses stated (H5, H6, H7, H8-partial) | ✅ Done | §1 |
| 2 | Quantitative predictions with thresholds | ✅ Done | §1 (5 pp / 3 pp) |
| 3 | Falsification conditions explicit | ✅ Done | §1 |
| 4 | Primary outcome defined | ✅ Done | §3 (success rate) |
| 5 | Conditions table complete (8 conditions) | ✅ Done | §5 |
| 6 | Controls & ablations specified | ✅ Done | §6 |
| 7 | Sample size + power analysis | ✅ Done | §7 (12,000 episodes; MDE ~6 pp) |
| 8 | Seeds fixed & documented | ✅ Done | §7 ([0,1,2,42,123]) |
| 9 | Analysis plan + test + correction | ✅ Done | §8 (Wilcoxon, Bonferroni α/28) |
| 10 | Stopping rules | ✅ Done | §9 |
| 11 | Decision rules pre-specified | ✅ Done | §1, §8.5 |
| 12 | Data & artifact storage plan | ✅ Done | §7 |
| 13 | Environment lock referenced | ✅ Done | `../environment.lock` |
| 14 | Cross-experiment dependency (EXP-001) | ✅ Done | §9 rule 4 |
| 15 | Code commit recorded at run time | ⏳ Pending execution | `env_info.json` |
| 16 | No look-ahead at outcomes before full run | ✅ Committed | §9 rule 3 |

**Pre-registration status:** COMPLETE. No changes to hypotheses, conditions, sample
size, or analysis plan after the first evaluation episode. Deviations logged in
`outputs/protocol_amendments.md`.

---

## 11. Execute / Verify / Refine Log

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Design | ✅ Complete | 2026-08-11 | This document |
| Pre-register | ✅ Complete | 2026-08-11 | §10 checklist done |
| Execute | ⏳ Not started | — | Run via `../run_all.sh` (exp 2) |
| Verify | ⏳ Not started | — | Determinism + solver-convergence audit |
| Refine | ⏳ Not started | — | Triggered on implementation failure / cross-exp gate |

---

## 12. Links

- Comparison plan: `../../docs/comparison_plan.md` (§2, §4)
- Research questions: `../../docs/research_questions.md` (H5, H6, H7, H8)
- Methodology: `../../docs/methodology.md` (§Benchmarks, §Metrics)
- Depends on: `../EXP-001-mechanism-ablation/protocol.md` (MIP condition)
- Environment lock: `../environment.lock`
- Runner: `../run_all.sh`
