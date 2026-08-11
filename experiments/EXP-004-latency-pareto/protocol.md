# EXP-004 — Latency-Performance Pareto

**Status:** Pre-registered
**Skill:** experiment-protocol (design → controls → pre-register → execute → verify → refine)
**Hypothesis under test:** H8 (latency–performance Pareto frontier)
**Owner:** Gyanateet
**Created:** 2026-08-11

---

## 1. Hypothesis & Prediction

### Hypothesis
**H8:** The three controller families occupy distinct regions of the latency–performance
Pareto frontier, and no single family dominates across the full range. Specifically:

- **Classical MPC** occupies the low-latency / moderate-performance region. It is
  Pareto-optimal when latency is the binding constraint but is dominated once
  generalization or multi-modal tasks matter.
- **VLA** occupies the high-latency / high-generalization region. It is
  Pareto-optimal when semantic / visual generalization is the binding constraint
  but is dominated on latency-sensitive tasks.
- **Diffusion / GCP** occupies the middle of the frontier, trading latency for
  performance by varying the number of iterative steps (T). Increasing T improves
  performance but increases latency monotonically.
- **MIP** (minimal iterative policy, 2-step + noise) is predicted to lie on the
  Pareto frontier — it achieves most of the diffusion benefit at a fraction of the
  latency, making it non-dominated by both full DDPM and single-step regression.
- **Regression** (single-step, no noise) is predicted to be Pareto-dominated by
  Linear MPC (similar latency, lower or equal performance on constraint tasks) or
  by MIP (slightly higher latency, significantly higher performance).

### Prediction (pre-registered, quantitative)
1. **Pareto frontier membership:** At least one configuration from each of the
   three families (MPC, Diffusion, VLA) lies on the Pareto frontier (non-dominated
   in the latency–success-rate plane) on at least one benchmark. A family with
   zero frontier points is considered dominated.
2. **MIP on the frontier:** MIP (2-step + noise) is on the Pareto frontier on
   **both** PushT and 2D Reaching. It is non-dominated: no other condition has
   both lower latency AND higher (or equal) success rate.
3. **Diffusion step sweep monotonicity:** For DDPM, increasing T from 1 → 100
   monotonically increases success rate (diminishing returns) and monotonically
   increases latency. The marginal performance gain per step decreases: the gain
   from T=1→8 exceeds the gain from T=8→100.
4. **Flow Matching efficiency:** Flow Matching with T=4 achieves success rate
   within 3 pp of DDPM with T=32, at ≤ 50% of the latency (fewer steps for
   comparable quality).
5. **Regression dominated:** Pure Regression is Pareto-dominated by MIP on both
   benchmarks (MIP has higher success at latency within 2× of regression).
6. **VLA frontier point:** SmallVLA is on the Pareto frontier on PushT (high
   success, high latency — not dominated by any diffusion config because no
   diffusion config achieves comparable PushT success at any latency).

### Falsification conditions
- H8 falsified if fewer than two families have any point on the Pareto frontier
  (i.e., one family dominates all others across the full latency range).
- Prediction 2 (MIP on frontier) falsified if MIP is dominated on both benchmarks
  (some condition has both lower latency and higher/equal success rate).
- Prediction 4 (Flow Matching efficiency) falsified if Flow Matching T=4 requires
  > 50% of DDPM T=32 latency to get within 3 pp, OR if it cannot get within 3 pp
  at any latency.
- Prediction 6 (VLA frontier on PushT) falsified if some diffusion configuration
  matches or exceeds SmallVLA's PushT success rate at lower latency.

---

## 2. Design

| Element | Specification |
|---------|---------------|
| Design type | Multi-condition observational sweep with Pareto analysis; within-subject (same evaluation episodes across conditions), repeated measures |
| Independent variable | Controller configuration (22 levels, see Conditions table) |
| Benchmarks | PushT, 2D Reaching |
| Unit of analysis | One evaluation episode |
| Pairing | All conditions evaluated on identical episode seeds per benchmark |
| Blinding | Scripted task-success predicate (no human rating); latency measured by automated harness |
| Randomization | Episode order shuffled per seed; condition order counterbalanced |

### Procedure (execute phase)
1. Train each learning-based controller once per seed on the shared training set
   (same data, same budget as EXP-001). For DDPM and Flow Matching, train a single
   model per seed and vary only the inference-time step count (T) — no retraining
   per T value. Save checkpoints + config snapshots to
   `outputs/<condition>/seed_<s>/`.
2. For MPC controllers (Linear MPC, Nonlinear MPC), configure once per seed (no
   training). For Regression and SmallVLA, train once per seed.
3. For each (condition, benchmark, seed), roll out 100 evaluation episodes using
   the fixed episode seed file (`data/eval_seeds_exp004.json`).
4. Separately, measure inference latency: 1000 timed calls (after 100 warmup) per
   (condition, benchmark, seed) under the identical latency protocol (DGX Spark
   GB10, fixed GPU clock, single-stream, no concurrent evaluation).
5. Record per-episode: success (0/1), inference latency (ms), action trajectory,
   config hash, git commit. Record per-timing-call: wall-clock latency (ms).
6. Aggregate to per-(condition, benchmark, seed) means; do NOT collapse across
   seeds before Pareto analysis.

### Verify phase
- **Latency measurement integrity:** Verify that latency is measured on the same
  hardware (DGX Spark GB10), same GPU clock, with no competing processes. Log
  `nvidia-smi` snapshot at measurement time. If GPU clock varies > 5%, re-measure.
- **Step-sweep monotonicity check:** For DDPM, verify that latency increases
  monotonically with T (T=1 < T=2 < ... < T=100). If not, the timing harness is
  faulty; fix and re-measure before analysis.
- **Determinism spot-check:** Re-run 1 seed of DDPM T=100 and Linear MPC with
  `cudnn.deterministic=True`; success must match within ±2 pp.
- **Sanity audit:** 5 random episodes per condition checked for NaN actions,
  valid episode length, plausible latency (latency > 0, no outliers > 10× median).

### Refine phase
- If any latency measurement is inconsistent (non-monotonic in T, or variance
  > 50% of mean), re-measure that condition with a longer warmup (500 calls) and
  more samples (5000 calls). Log the amendment.
- If a condition's success rate deviates > 5 pp from its EXP-001 / EXP-002 value
  (for matched configurations), flag the implementation; re-check, re-train if
  needed, log amendment.

---

## 3. Primary Outcome

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Success rate vs inference latency (Pareto plot)** (primary) | Per-(condition, benchmark, seed): mean success rate (over 100 episodes) plotted against median inference latency (over 1000 timed calls) | Pareto scatter plot per benchmark; frontier computed from per-seed means, with seed-level variability shown as error bars |

Task-success predicates:
- **PushT:** final pose of T-shaped object within 0.05 m (translation) and 0.1 rad
  (rotation) of target.
- **2D Reaching:** end-effector reaches within 0.02 m of goal and holds for 5 steps.

---

## 4. Secondary Outcomes

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Pareto dominance count** | For each condition, the number of other conditions that dominate it (lower latency AND higher success rate) | Computed per benchmark from per-seed means; 0 = on the frontier |
| **Pareto frontier membership** | Binary: is the condition non-dominated (dominance count = 0)? | Per (condition, benchmark), aggregated over seeds (on-frontier if non-dominated in ≥ 3 of 5 seeds) |
| **Marginal performance per step** | Δsuccess / ΔT for consecutive T values in the DDPM and Flow Matching sweeps | Per (benchmark, seed); averaged over seeds |
| **Latency per step** | Median inference latency / T for iterative methods | Per condition; shows per-step compute cost |
| **p95 latency** | 95th percentile inference latency | Per condition; for real-time feasibility assessment |

---

## 5. Conditions Table

| ID | Condition | Family | Parameter | Values | Iterative? | Notes |
|----|-----------|--------|-----------|--------|-----------|-------|
| C1 | Linear MPC | MPC | — (fixed) | 1 config | No (QP, 1 solve) | Low-latency floor |
| C2 | Nonlinear MPC | MPC | — (fixed) | 1 config | No (iLQR) | Moderate-latency MPC |
| C3 | DDPM (T=1) | Diffusion | Steps T | 1 | Yes | Single-step diffusion; ablates iterative compute |
| C4 | DDPM (T=2) | Diffusion | Steps T | 2 | Yes | Minimal iterative diffusion |
| C5 | DDPM (T=4) | Diffusion | Steps T | 4 | Yes | Low-step diffusion |
| C6 | DDPM (T=8) | Diffusion | Steps T | 8 | Yes | Mid-step diffusion |
| C7 | DDPM (T=16) | Diffusion | Steps T | 16 | Yes | Mid-step diffusion |
| C8 | DDPM (T=32) | Diffusion | Steps T | 32 | Yes | High-step diffusion |
| C9 | DDPM (T=64) | Diffusion | Steps T | 64 | Yes | Near-full diffusion |
| C10 | DDPM (T=100) | Diffusion | Steps T | 100 | Yes | Full DDPM; replication anchor from EXP-001 |
| C11 | Flow Matching (T=1) | Diffusion | Steps T | 1 | Yes | Single-step rectified flow |
| C12 | Flow Matching (T=2) | Diffusion | Steps T | 2 | Yes | Low-step rectified flow |
| C13 | Flow Matching (T=4) | Diffusion | Steps T | 4 | Yes | Mid-step rectified flow |
| C14 | Flow Matching (T=8) | Diffusion | Steps T | 8 | Yes | High-step rectified flow |
| C15 | Flow Matching (T=10) | Diffusion | Steps T | 10 | Yes | Full rectified flow |
| C16 | MIP (iter=1) | Diffusion | Iterations | 1 | Yes (1-step + noise) | Minimal iterative — below EXP-001 config |
| C17 | MIP (iter=2) | Diffusion | Iterations | 2 | Yes (2-step + noise) | EXP-001 MIP config; predicted frontier point |
| C18 | MIP (iter=3) | Diffusion | Iterations | 3 | Yes (3-step + noise) | Slightly above EXP-001 MIP |
| C19 | MIP (iter=5) | Diffusion | Iterations | 5 | Yes (5-step + noise) | Higher-iteration MIP |
| C20 | Regression | Diffusion | — (fixed) | 1 config | No (T=1, no noise) | Single-step floor; from EXP-001 |
| C21 | SmallVLA | VLA | — (fixed) | 1 config | No (regression head) | High-latency / high-generalization |

**Total configurations:** 21 conditions × 2 benchmarks = 42 (condition, benchmark) cells.

**Held constant across all conditions:** backbone architecture & parameter count
(compute-matched for learning-based), training data, training budget (epochs /
steps), observation preprocessing, action space, control frequency (10 Hz),
evaluation episodes, latency measurement protocol (hardware, warmup, sample count).

---

## 6. Controls & Ablations

### Controls (fairness)
- Same observation preprocessing (resize 224×224, ImageNet normalize) across all
  learning-based conditions.
- Same action space (end-effector delta, 2D for PushT/Reaching).
- Same control frequency (10 Hz).
- Same training data and same number of gradient steps (compute-matched) for all
  learning-based conditions.
- Same backbone parameter budget (±5%) across C3–C21.
- DDPM and Flow Matching models trained once per seed; only inference-time T
  varies. This isolates the effect of iterative steps from training stochasticity.
- MPC methods (C1, C2) use the same dynamics model and solver tolerances as
  EXP-002.
- Fixed episode seeds shared across all conditions (paired design).
- Latency measured under identical protocol: DGX Spark GB10, fixed GPU clock,
  100 warmup calls, 1000 timed calls, single-stream (no concurrent evaluation).

### Ablations embedded
- **DDPM step sweep (C3–C10):** Varying T from 1 → 100 traces the diffusion
  latency–performance curve. The shape (convex, diminishing returns) is a key
  result.
- **Flow Matching step sweep (C11–C15):** Varying T from 1 → 10 traces the
  rectified-flow curve. Compared to DDPM, flow matching should achieve comparable
  performance in fewer steps (straighter trajectories).
- **MIP iteration sweep (C16–C19):** Varying iterations from 1 → 5 traces the
  minimal-iterative curve. Tests whether 2 iterations (EXP-001 config) is already
  on the frontier or if more iterations help.
- **Family extremes:** C1 (Linear MPC, lowest latency) and C21 (SmallVLA, highest
  latency) anchor the two ends of the frontier.

### Negative / positive controls
- **Regression (C20):** negative control; predicted to be Pareto-dominated.
- **Linear MPC (C1):** low-latency positive control; should be on the frontier at
  the low-latency end.
- **DDPM T=100 (C10):** replication anchor from EXP-001; success rate must match
  EXP-001 within ±5 pp (cross-experiment consistency gate).

---

## 7. Data

### Sample size
- 100 episodes per (condition, benchmark, seed) × 5 seeds × 21 conditions × 2
  benchmarks = **21,000 episodes** total.
- Plus 1000 latency measurements per (condition, benchmark, seed) × 5 seeds × 21
  conditions × 2 benchmarks = **210,000 latency measurements** total.

### Seeds
`[0, 1, 2, 42, 123]` — fixed across all conditions and benchmarks. Episode initial
states drawn from `data/eval_seeds_exp004.json` (committed, hashed).

### Power analysis (pre-registered)
- This is primarily an observational / Pareto analysis, not a pairwise hypothesis
  test. Power considerations:
  - **Latency precision:** With 1000 timed calls, the standard error of the median
    is approximately IQR / (1.25 × √1000) ≈ 0.03 × IQR. Latency differences > 5%
    between conditions are detectable.
  - **Success-rate precision:** With 5 seeds × 100 episodes, the standard error of
    the mean success rate is ≈ std / √5. MDE ~5 pp at power 0.80 (consistent with
    EXP-001/002/003).
  - **Pareto frontier stability:** A condition is declared "on the frontier" if
    non-dominated in ≥ 3 of 5 seeds. With 5 seeds, this requires the condition to
    be non-dominated in the majority of seeds, providing robustness to seed-level
    noise.
- If observed success-rate std > 12 pp, flag as underpowered and record (do not
  silently add seeds).

### Data storage
```
experiments/EXP-004-latency-pareto/outputs/
├── <condition>/
│   └── <benchmark>/
│       └── seed_<s>/
│           ├── config.yaml
│           ├── env_info.json
│           ├── checkpoint.pt (learning-based only)
│           ├── episodes.jsonl          # per-episode success, latency
│           ├── latency_timings.json    # 1000 raw latency measurements
│           └── metrics_summary.json
└── analysis/
    ├── pareto_points.csv              # condition, benchmark, seed, success, latency
    ├── pareto_frontier.csv            # non-dominated points per benchmark
    ├── dominance_counts.csv           # per condition: # of dominators
    ├── marginal_performance.csv       # Δsuccess / ΔT for DDPM and Flow Matching
    ├── step_sweep_curves.csv          # T vs success and T vs latency
    └── figures/
        ├── pareto_pusht.png
        ├── pareto_reaching.png
        ├── ddpm_step_sweep.png
        ├── flow_matching_step_sweep.png
        └── mip_iteration_sweep.png
```

---

## 8. Analysis Plan

### Primary analysis
1. **Pareto scatter plot:** For each benchmark, plot all 21 conditions as points
   (x = median latency, y = mean success rate) with error bars (p95 latency,
   success-rate std over seeds). One plot per benchmark.
2. **Pareto frontier computation:** For each benchmark, compute the set of
   non-dominated conditions (no other condition has both lower latency AND higher
   or equal success rate). Use per-seed means; a condition is "on the frontier" if
   non-dominated in ≥ 3 of 5 seeds.
3. **Pareto dominance counting:** For each condition, count the number of other
   conditions that dominate it (lower latency AND higher success rate). A dominance
   count of 0 = on the frontier. Report as a table per benchmark.
4. **Frontier membership by family:** Report which families (MPC, Diffusion, VLA)
   have at least one condition on the frontier per benchmark. This directly tests
   Prediction 1.
5. **MIP frontier test:** Check whether MIP (iter=2, C17) is on the frontier on
   both benchmarks (Prediction 2).

### Secondary analysis
- **DDPM step sweep curve:** Plot success rate and latency vs T (1, 2, 4, 8, 16,
  32, 64, 100). Fit a saturating function (e.g., `a × (1 - exp(-b × T))`) to
  success rate; report the half-saturation point T*. Compute marginal performance
  per step (Δsuccess / ΔT) and verify diminishing returns (Prediction 3).
- **Flow Matching step sweep curve:** Plot success rate and latency vs T (1, 2, 4,
  8, 10). Compare to DDPM curve: verify Flow Matching T=4 is within 3 pp of DDPM
  T=32 at ≤ 50% latency (Prediction 4).
- **MIP iteration sweep curve:** Plot success rate and latency vs iterations (1,
  2, 3, 5). Check whether iter=2 is already near the plateau.
- **Regression dominance test:** Verify Regression (C20) is dominated by MIP
  (C17) on both benchmarks (Prediction 5).
- **VLA frontier test:** Verify SmallVLA (C21) is on the PushT frontier
  (Prediction 6).
- **Latency per step:** Report median latency / T for each iterative condition;
  compare DDPM, Flow Matching, and MIP per-step costs.

### Reporting
- Pareto plots (one per benchmark) with all conditions labeled.
- Frontier membership table (condition × benchmark, on/off frontier).
- Dominance count table.
- Step-sweep curves (DDPM, Flow Matching, MIP) with fitted saturating functions.
- All raw per-episode and per-timing data committed (JSONL + JSON) so analyses
  are re-runnable.
- Figures generated from committed results, not manual edits.
- Analysis script: `experiments/EXP-004-latency-pareto/analyze.py` (committed).

### Software
- `numpy` / `matplotlib` for Pareto plots and step-sweep curves.
- `scipy.optimize.curve_fit` for saturating function fits.
- Pareto frontier computed via custom non-dominated sort (O(n²) is sufficient for
  n=21 conditions).
- No pairwise hypothesis tests are primary; Wilcoxon may be used exploratorily for
  specific condition pairs (e.g., MIP vs Regression) but is not the primary
  analysis.

---

## 9. Stopping Rules

1. **Cross-experiment consistency gate (hard stop):** If DDPM T=100 (C10) success
   rate deviates > 5 pp from its EXP-001 value on either benchmark, STOP. Do not
   run Pareto analysis. Re-check training pipeline, re-train if needed, log
   amendment. Resume only after consistency passes.
2. **Latency measurement failure stop:** If latency measurements are non-monotonic
   in T for DDPM (i.e., latency(T=1) ≥ latency(T=100)), STOP. The timing harness
   is faulty; fix and re-measure before analysis. Do not proceed with broken
   latency data.
3. **Training failure stop:** If any condition fails to train (loss NaN, diverges,
   or success ≤ 0 on held-out training rollouts), stop that condition, log the
   failure, and re-run once with a fresh seed initialization. If it fails again,
   exclude and report the exclusion explicitly.
4. **Resource stop:** If wall-clock exceeds 2× the pre-registered budget estimate
   (see README timeline), stop, log partial results, and re-plan. Do not silently
   extend.
5. **No peeking:** Interim success rates and latency measurements are NOT examined
   for Pareto analysis until all 5 seeds × 100 episodes AND all 1000 latency
   measurements are complete for a given (condition, benchmark). Training-loss
   curves may be monitored for debugging only.
6. **Futility (optional, pre-registered):** After 3 seeds, if one family dominates
   all others across the full latency range on both benchmarks (i.e., the Pareto
   frontier consists entirely of one family), one may stop early and report H8 as
   falsified; otherwise complete all 5 seeds.

---

## 10. Pre-registration Checklist

| # | Item | Status | Evidence / Location |
|---|------|--------|---------------------|
| 1 | Hypothesis stated (H8) | ✅ Done | §1 |
| 2 | Quantitative predictions with thresholds | ✅ Done | §1 (frontier membership, 3 pp, 50% latency) |
| 3 | Falsification conditions explicit | ✅ Done | §1 |
| 4 | Primary outcome defined | ✅ Done | §3 (Pareto plot: success vs latency) |
| 5 | Conditions table complete (21 conditions) | ✅ Done | §5 |
| 6 | Controls & ablations specified | ✅ Done | §6 |
| 7 | Sample size + power analysis | ✅ Done | §7 (21,000 episodes + 210,000 latency measurements) |
| 8 | Seeds fixed & documented | ✅ Done | §7 ([0,1,2,42,123]) |
| 9 | Analysis plan (Pareto frontier + dominance) | ✅ Done | §8 |
| 10 | Stopping rules | ✅ Done | §9 |
| 11 | Decision rules pre-specified | ✅ Done | §1, §8.4–8.5 |
| 12 | Data & artifact storage plan | ✅ Done | §7 |
| 13 | Environment lock referenced | ✅ Done | `../environment.lock` |
| 14 | Cross-experiment dependency (EXP-001, EXP-002) | ✅ Done | §9 rule 1, §6 (DDPM T=100 consistency gate) |
| 15 | Latency measurement protocol specified | ✅ Done | §2, §6 (DGX Spark, 100 warmup, 1000 calls) |
| 16 | Code commit recorded at run time | ⏳ Pending execution | `env_info.json` |
| 17 | No look-ahead at outcomes before full run | ✅ Committed | §9 rule 5 |

**Pre-registration status:** COMPLETE. No changes to hypotheses, conditions,
parameter sweeps, sample size, or analysis plan are permitted after the first
evaluation episode is run. Any deviation must be logged as a protocol amendment
with date and rationale in `outputs/protocol_amendments.md`.

---

## 11. Execute / Verify / Refine Log

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Design | ✅ Complete | 2026-08-11 | This document |
| Pre-register | ✅ Complete | 2026-08-11 | §10 checklist done |
| Execute | ⏳ Not started | — | Run via `../run_all.sh` (exp 4) |
| Verify | ⏳ Not started | — | Latency-integrity + monotonicity + determinism audit |
| Refine | ⏳ Not started | — | Triggered on consistency-gate or latency-measurement failure |

---

## 12. Links

- Comparison plan: `../../docs/comparison_plan.md` (§4.5)
- Research questions: `../../docs/research_questions.md` (H8)
- Methodology: `../../docs/methodology.md` (§Pareto Analysis, §Latency Protocol)
- Depends on: `../EXP-001-mechanism-ablation/protocol.md` (DDPM T=100 consistency)
- Depends on: `../EXP-002-family-comparison/protocol.md` (MPC, VLA, MIP configs)
- Environment lock: `../environment.lock`
- Runner: `../run_all.sh`
