# EXP-001 — GCP Mechanism Ablation

**Status:** Pre-registered
**Skill:** experiment-protocol (design → controls → pre-register → execute → verify → refine)
**Hypothesis under test:** H3 (mechanism of GCP advantage)
**Owner:** Gyanateet
**Created:** 2026-08-11

---

## 1. Hypothesis & Prediction

### Hypothesis
**H3:** Is the advantage of generative control policies (GCPs) due to *distribution fitting*
(the conventional view, H3a) or to *iterative compute + noise injection* (the Simchowitz
view, H3b)?

- **H3a (conventional):** The generative / distribution-fitting component is essential.
  Removing it (MIP, pure regression) should cause a large drop in success rate and mode
  coverage even when iterative compute and noise are retained.
- **H3b (Simchowitz):** Iterative compute + noise injection is the key mechanism.
  A minimal iterative policy (MIP, 2-step regression + noise) should match the full DDPM
  GCP. Distribution fitting contributes little beyond what iterative compute + noise
  already provide.

### Prediction (pre-registered, quantitative)
1. **Replication target:** On PushT, Full DDPM (T=100) success rate will fall within
   ±5 percentage points of the value reported by Simchowitz et al. for the matched
   configuration. If it falls outside, the replication is flagged before any further
   inference is drawn.
2. **Mechanism test:** `success(MIP) ≥ success(Full DDPM) − 3 pp` on **both** PushT and
   2D Reaching → supports H3b. A gap > 3 pp with `success(Full DDPM) > success(MIP)`
   and a significant paired test → supports H3a.
3. **Component ordering (predicted under H3b):**
   `Full DDPM ≈ DDPM no-noise⁻ ≈ MIP > DDPM single-step > Pure Regression`,
   where the largest single drop comes from removing *iterative compute* (single-step),
   and removing *noise* has a smaller but non-zero effect.
4. **Mode coverage:** Full DDPM > MIP > Pure Regression on PushT (multi-modal task).
   On 2D Reaching (uni-modal), mode coverage differences are predicted to be negligible.

### Falsification conditions
- H3b is falsified if `success(Full DDPM) − success(MIP) > 3 pp` AND the Wilcoxon
  signed-rank test is significant (p < 0.05, Bonferroni-corrected) on **both** benchmarks.
- H3a is falsified if `success(MIP) ≥ success(Full DDPM) − 3 pp` on **both** benchmarks
  with non-significant paired test.
- The replication is considered failed (and the experiment paused for refinement) if the
  Full DDPM PushT success rate deviates > 5 pp from the Simchowitz reported value.

---

## 2. Design

| Element | Specification |
|---------|---------------|
| Design type | Within-subject (same evaluation episodes across conditions), repeated measures |
| Independent variable | Policy variant (5 levels, see Conditions table) |
| Benchmark | PushT, 2D Reaching (state + image obs) |
| Unit of analysis | One evaluation episode |
| Pairing | Conditions evaluated on identical episode seeds (same initial states) |
| Blinding | Episode success judged by scripted task-success predicate (no human rating) |
| Randomization | Episode order shuffled per seed; condition order counterbalanced |

### Procedure (execute phase)
1. Train each policy variant once per seed on the shared training set (same data, same
   budget). Save checkpoints + config snapshots to `outputs/<condition>/seed_<s>/`.
2. For each (condition, seed), roll out 100 evaluation episodes using the fixed episode
   seed file (`data/eval_seeds_exp001.json`).
3. Record per-episode: success (0/1), mode-coverage vector, inference latency (ms),
   action trajectory, config hash, git commit.
4. Aggregate to per-(condition, seed) means; do NOT collapse across seeds before testing.

### Verify phase
- Re-run 1 seed of Full DDPM and Pure Regression with `cudnn.deterministic=True`;
  success rate must match the non-deterministic run within ±2 pp (tolerance for
  non-determinism). If outside, flag and re-run with deterministic settings.
- Spot-check 5 random episodes per condition against logged trajectories for sanity
  (no NaN actions, episode length within task bounds).

### Refine phase
- If replication fails (Prediction 1), do NOT proceed to mechanism inference. Instead:
  (a) re-check hyperparameters against Simchowitz config, (b) re-train Full DDPM,
  (c) re-evaluate. Only after replication passes does mechanism analysis run.

---

## 3. Primary Outcome

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Success rate** (primary) | Fraction of 100 episodes achieving the task goal per (condition, seed) | Binary per episode, averaged; reported as mean ± std over 5 seeds |

Task-success predicates:
- **PushT:** final pose of T-shaped object within 0.05 m (translation) and 0.1 rad
  (rotation) of target.
- **2D Reaching:** end-effector reaches within 0.02 m of goal and holds for 5 steps.

---

## 4. Secondary Outcomes

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Mode coverage** | Fraction of expert action clusters hit by ≥1 policy sample | Cluster expert actions (k-means, k=4 on PushT, k=2 on Reaching); per-episode sample hits |
| **Inference latency** | Wall-clock time observation → action | Median + p95 over 1000 timed calls (after 100 warmup), per condition |
| **Action KL** (exploratory) | KL(policy actions ‖ expert actions) | Kernel density estimate per condition |

---

## 5. Conditions Table

| ID | Condition | Iterative steps | Noise injection | Distribution fitting | Backbone | Notes |
|----|-----------|-----------------|-----------------|----------------------|----------|-------|
| C1 | Full DDPM | T=100 | Yes (DDPM schedule) | Yes (score matching) | Shared UNet/MLP | Baseline GCP; replication anchor |
| C2 | DDPM no-noise | T=100 | No (deterministic) | Yes | Shared UNet/MLP | Ablate noise; keep iterative + fitting |
| C3 | DDPM single-step | T=1 | Yes | Yes | Shared UNet/MLP | Ablate iterative compute; keep fitting + noise |
| C4 | MIP (2-step + noise) | T=2 | Yes | No (regression head) | Shared backbone | Simchowitz minimal iterative policy |
| C5 | Pure Regression | T=1 | No | No (regression head) | Shared backbone | RCP baseline; floor |

**Held constant across all conditions:** backbone architecture & parameter count
(compute-matched), training data, training budget (epochs / steps), observation
preprocessing, action space, control frequency (10 Hz), evaluation episodes.

---

## 6. Controls & Ablations

### Controls (fairness)
- Same observation preprocessing (resize 224×224, ImageNet normalize) across conditions.
- Same action space (end-effector delta, 2D for PushT/Reaching).
- Same control frequency (10 Hz).
- Same training data and same number of gradient steps (compute-matched).
- Same backbone parameter budget (±5%) across C1–C5.
- Fixed episode seeds shared across conditions (paired design).

### Ablations (the experiment *is* an ablation)
- **Ablate noise:** C1 → C2 (hold iterative + fitting, remove noise).
- **Ablate iterative compute:** C1 → C3 (hold fitting + noise, set T=1).
- **Ablate distribution fitting:** C1 → C4 (replace score matching with regression head,
  keep 2-step + noise).
- **Ablate everything:** C1 → C5 (regression, single-step, no noise).

### Negative controls
- **Pure Regression (C5)** serves as the floor; any condition not significantly above C5
  is treated as "no effective mechanism."

### Positive control
- **Full DDPM (C1)** must replicate Simchowitz's reported PushT success rate (±5 pp)
  before mechanism claims are made.

---

## 7. Data

### Sample size
- 100 episodes per condition per seed × 5 seeds × 5 conditions × 2 benchmarks
  = **5,000 episodes** total.

### Seeds
`[0, 1, 2, 42, 123]` — fixed across all conditions and benchmarks. Episode initial states
drawn from `data/eval_seeds_exp001.json` (committed, hashed).

### Power analysis (pre-registered)
- Minimal detectable effect (paired, Wilcoxon): ~5 pp difference in success rate at
  α=0.05 (Bonferroni-corrected for 10 pairwise comparisons), power ≈ 0.80, given
  5 seeds × 100 episodes and an assumed within-seed correlation ρ=0.3.
- If observed std > 12 pp, the design is underpowered; record and flag for refinement
  (add seeds, do not silently increase).

### Data storage
```
experiments/EXP-001-mechanism-ablation/outputs/
├── <condition>/
│   └── seed_<s>/
│       ├── config.yaml
│       ├── env_info.json
│       ├── checkpoint.pt
│       ├── episodes.jsonl        # per-episode success, mode-coverage, latency
│       └── metrics_summary.json
└── analysis/
    ├── comparison_table.csv
    ├── wilcoxon_results.json
    └── figures/
```

---

## 8. Analysis Plan

### Primary analysis
1. **Replication check (gate):** Compare Full DDPM (C1) PushT success rate to Simchowitz
   reported value. Must be within ±5 pp to proceed.
2. **Pairwise comparisons:** Wilcoxon signed-rank test on per-seed success rates for all
   10 condition pairs (C1–C5), separately per benchmark.
3. **Multiple-comparison correction:** Bonferroni correction over the 10 pairwise tests
   per benchmark (α = 0.05/10 = 0.005 per test).
4. **Effect size:** Report Cohen's d (or rank-biserial correlation for Wilcoxon) and
   95% CI per comparison.
5. **Mechanism decision rule:** Apply the pre-registered thresholds from §1 (3 pp gap,
   significance on both benchmarks).

### Secondary analysis
- Mode coverage: bar chart per condition per benchmark; Wilcoxon on mode-coverage scores.
- Inference latency: report median + p95; plot success-rate vs latency (feeds EXP-004).
- Component-attribution: decompose the C1→C5 gap into (noise) + (iterative) + (fitting)
  contributions via the ablation deltas.

### Reporting
- Mean ± std per condition per benchmark.
- All p-values, effect sizes, CIs.
- Raw per-episode data committed (JSONL) so tests are re-runnable.
- Figures generated from committed results, not manual edits.

### Software
- `scipy.stats.wilcoxon` for paired tests.
- `statsmodels` for Bonferroni correction and effect-size CIs.
- Analysis script: `experiments/EXP-001-mechanism-ablation/analyze.py` (committed).

---

## 9. Stopping Rules

1. **Replication gate (hard stop):** If Full DDPM PushT success rate is > 5 pp from the
   Simchowitz reported value, STOP. Do not run mechanism analysis. Enter refine phase
   (re-check config, re-train, re-evaluate). Resume only after replication passes.
2. **Training failure stop:** If any condition fails to train (loss NaN, diverges, or
   success ≤ 0 on held-out training rollouts), stop that condition, log the failure,
  and re-run once with a fresh seed initialization. If it fails again, exclude and
  report the exclusion explicitly.
3. **Resource stop:** If wall-clock exceeds 2× the pre-registered budget estimate
   (see README timeline), stop, log partial results, and re-plan. Do not silently
   extend.
4. **No peeking:** Interim success rates are NOT examined for hypothesis testing until
   all 5 seeds × 100 episodes are complete for a given (condition, benchmark). Latency
   and training-loss curves may be monitored for debugging only.
5. **Futility (optional, pre-registered):** After 3 seeds, if the C1–C4 gap is already
   > 15 pp in the same direction on both benchmarks (i.e., H3b clearly false), one may
   stop early and report; otherwise complete all 5 seeds.

---

## 10. Pre-registration Checklist

| # | Item | Status | Evidence / Location |
|---|------|--------|---------------------|
| 1 | Hypothesis stated (H3a vs H3b) | ✅ Done | §1 |
| 2 | Quantitative prediction with thresholds | ✅ Done | §1 (3 pp, 5 pp replication) |
| 3 | Falsification conditions explicit | ✅ Done | §1 |
| 4 | Primary outcome defined | ✅ Done | §3 (success rate) |
| 5 | Conditions table complete | ✅ Done | §5 |
| 6 | Controls & ablations specified | ✅ Done | §6 |
| 7 | Sample size + power analysis | ✅ Done | §7 (5,000 episodes; MDE ~5 pp) |
| 8 | Seeds fixed & documented | ✅ Done | §7 ([0,1,2,42,123]) |
| 9 | Analysis plan + test + correction | ✅ Done | §8 (Wilcoxon, Bonferroni α/10) |
| 10 | Stopping rules | ✅ Done | §9 |
| 11 | Decision rule pre-specified | ✅ Done | §1, §8.5 |
| 12 | Data & artifact storage plan | ✅ Done | §7 |
| 13 | Environment lock referenced | ✅ Done | `../environment.lock` |
| 14 | Code commit to be recorded at run time | ⏳ Pending execution | logged in `env_info.json` |
| 15 | No look-ahead at outcomes before full run | ✅ Committed | §9 rule 4 |

**Pre-registration status:** COMPLETE. No changes to hypotheses, conditions, sample size,
or analysis plan are permitted after the first evaluation episode is run. Any deviation
must be logged as a protocol amendment with date and rationale in
`outputs/protocol_amendments.md`.

---

## 11. Execute / Verify / Refine Log

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Design | ✅ Complete | 2026-08-11 | This document |
| Pre-register | ✅ Complete | 2026-08-11 | §10 checklist done |
| Execute | ⏳ Not started | — | Run via `../run_all.sh` (exp 1) |
| Verify | ⏳ Not started | — | Determinism spot-check + sanity audit |
| Refine | ⏳ Not started | — | Triggered only if replication gate fails |

---

## 12. Links

- Comparison plan: `../../docs/comparison_plan.md` (§4.1)
- Research questions: `../../docs/research_questions.md` (H3)
- Methodology: `../../docs/methodology.md` (§Ablation Protocol)
- Environment lock: `../environment.lock`
- Runner: `../run_all.sh`
