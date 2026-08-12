# EXP-005 — World Action Model Baseline

**Status:** Pre-registered  
**Skill:** experiment-protocol (design → controls → pre-register → execute → verify → refine)  
**Hypothesis under test:** `H_WAM` (World Action Models improve long-horizon and contact-rich manipulation over pure VLA)  
**Owner:** Gyanateet  
**Created:** 2026-08-11  

---

## 1. Hypothesis & Prediction

### Hypothesis

**`H_WAM`:** A small world action model (WAM) trained from scratch — which jointly predicts future latent states / observations and produces actions — improves long-horizon and contact-rich manipulation over pure VLA and pure diffusion baselines by leveraging an explicit, roll-out-capable dynamics model.

This experiment addresses:

- Where does WAM fit in the MPC vs VLA vs Diffusion comparison?
- Does an explicit world model help on tasks that require multi-step reasoning, tool use, or contact transitions?
- Does WAM retain comparable performance on short-horizon, single-stage tasks where reactive policies should suffice?

### Prediction (pre-registered, quantitative)

1. **Long-horizon / contact-rich advantage:** On `StackCube` and `Square` (multi-stage, contact-rich), `success(WAM) ≥ success(SmallVLA) + 5 pp` and `success(WAM) ≥ success(MIP) + 3 pp`.
2. **Short-horizon parity:** On `Lift` and `PickCube` (single-stage), WAM is within `±3 pp` of the best performing condition (likely DDPM, MIP, or MPC).
3. **Sub-goal completion:** WAM completes ≥ 10% more sub-goals (grasp, transport, place/insert) than SmallVLA on long-horizon tasks.
4. **Inference latency:** WAM forward pass is < 2× the latency of SmallVLA when using a small latent rollout (≤ 8 steps).
5. **MPC on high-precision short tasks:** Nonlinear MPC outperforms WAM on `Lift` by ≥ 3 pp when the object pose and dynamics model are known.

### Falsification conditions

- `H_WAM` is falsified if WAM does **not** beat SmallVLA by ≥ 3 pp on either long-horizon / contact-rich task AND is not within 3 pp of the best condition on single-stage tasks.
- If WAM is > 3× slower than SmallVLA (latency), the real-time feasibility claim is flagged regardless of success.

---

## 2. Design

| Element | Specification |
|---------|---------------|
| Design type | Within-subject (same evaluation episodes across conditions), repeated measures |
| Independent variable | Controller (5 levels, see Conditions table) |
| Benchmarks | MuJoCo robosuite (`Lift`, `Square`) and ManiSkill3 (`PickCube`, `StackCube`) |
| Unit of analysis | One evaluation episode |
| Pairing | All conditions evaluated on identical episode seeds per benchmark |
| Blinding | Scripted task-success predicate and sub-goal checker (no human rating) |
| Randomization | Episode order shuffled per seed; condition order counterbalanced |

### Procedure (execute phase)

1. Set up real-physics sim bindings and the shared real-robot vision preprocessing pipeline (see [`docs/real_robotics/phase2_roadmap.md`](../../docs/real_robotics/phase2_roadmap.md)).
2. Train each learning-based condition once per seed on the shared training set:
   - Use the same demonstrations for all learning-based methods.
   - Train WAM, SmallVLA, DDPM, and MIP with compute-matched total parameter budgets.
   - Save checkpoints + config snapshots to `outputs/<condition>/seed_<s>/`.
3. Configure the Nonlinear MPC condition once per seed (no learning; uses full Franka dynamics model).
4. For each (condition, benchmark, seed), roll out 100 evaluation episodes using a fixed episode seed file (`data/eval_seeds_exp005.json`).
5. Record per-episode: success, sub-goal completion vector, contact flag, inference latency, action trajectory, config hash, git commit.
6. Aggregate to per-(condition, benchmark, seed) means; do NOT collapse across seeds before testing.

### Verify phase

- **WAM ablation:** Re-run WAM with the future-prediction head disabled (pure policy) on 1 seed. WAM-with-rollout must outperform WAM-no-rollout by ≥ 3 pp on `StackCube` or `Square` to justify the world-model component.
- **Determinism spot-check:** Re-run 1 seed of WAM and DDPM with `cudnn.deterministic=True`; success must match within ±2 pp.
- **Sanity audit:** 5 random episodes per (condition, benchmark) checked for NaN actions, valid episode length, and plausible end-effector trajectories.

### Refine phase

- If WAM underperforms SmallVLA on long-horizon tasks, inspect:
  - Whether the latent rollout length is too short.
  - Whether the training loss properly weights future prediction vs action regression.
  - Whether the observation preprocessing is consistent with SmallVLA.
- Re-train only the WAM condition with amended config; log the amendment.

---

## 3. Primary Outcome

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Success rate** (primary) | Fraction of 100 episodes achieving the task goal per (condition, benchmark, seed) | Binary per episode, averaged; reported as mean ± std over 5 seeds |

Task-success predicates:

- **`Lift`:** object lifted ≥ 5 cm above the table and held for ≥ 5 steps.
- **`Square`:** square peg fully inserted into the hole (translation error < 0.01 m, rotation error < 0.05 rad).
- **`PickCube`:** cube grasped and lifted ≥ 5 cm.
- **`StackCube`:** cube A picked and placed stably on top of cube B (top cube stays for ≥ 5 steps).

---

## 4. Secondary Outcomes

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Sub-goal completion rate** | Fraction of sub-goals (e.g., reach, grasp, transport, place) completed per episode | Scripted sub-goal predicates; reported as mean over episodes |
| **Contact-rich success** | Success rate on tasks with ≥ 10 contact events (`Square`, `StackCube`) | Binary per episode, averaged |
| **Inference latency** | Wall-clock time observation → action | Median + p95 over 1000 timed calls (100 warmup) |
| **Data efficiency** | Success rate vs. number of demonstrations | Curve and AUC for learning-based conditions |
| **Manifold adherence** (exploratory) | Distance of policy actions to expert action manifold under clean RGB | k-NN distance (k=5) to expert demonstrations; lower = better |
| **Constraint violation rate** | % timesteps violating joint limits, torque limits, or collision constraints | Per-step binary, averaged; for MPC and learning-based conditions |

---

## 5. Conditions Table

| ID | Condition | Family | World model? | Constraints | Language | Notes |
|----|-----------|--------|--------------|-------------|----------|-------|
| C1 | **WAM + Action Policy** | WAM | Yes (small, from scratch, latent rollout) | None / soft | Yes (via VLM or task embedding) | Jointly trained dynamics + action policy; ≤ 100M params, compute-matched to SmallVLA |
| C2 | **SmallVLA** | VLA | No | None | Yes | Small VLM + regression/head; compute-matched to WAM |
| C3 | **DDPM Policy** | Diffusion | No | None | No | Standard DDPM policy; T=100 or T=50 depending on latency budget |
| C4 | **MIP** | Diffusion | No | None | No | 2-step regression + noise from `EXP-001` |
| C5 | **Nonlinear MPC** | MPC | Explicit (analytical/identified) | Hard (dynamics, joint/torque limits) | No | Full Franka full-order or operational-space MPC; uses object pose if known |

**Held constant across conditions:**
- Same observation preprocessing (resize, normalize, real-robot vision pipeline).
- Same action space (joint positions or end-effector pose with gripper).
- Same control frequency (10 Hz for learning-based; native for MPC but logged).
- Same training data (demonstrations) for C1–C4.
- Fixed episode seeds shared across conditions (paired design).
- Same success and sub-goal predicates.

---

## 6. Controls & Ablations

### Controls (fairness)

- **Parameter budget:** WAM and SmallVLA use the same total parameter budget (within ±10%). DDPM and MIP use the same backbone size as in `EXP-001`/`EXP-002`.
- **Training data:** C1–C4 trained on the same demonstration set and same number of gradient steps.
- **Vision preprocessing:** All conditions share the same RGB (and optional depth) preprocessing.
- **Episode pairing:** Identical initial states for every condition and seed.
- **Latency protocol:** Measured on the same hardware with 100 warm-up and 1000 timed calls.

### Ablations embedded

- **World model contribution:** C1 (WAM) vs a WAM-with-rollout-disabled variant (internal ablation) isolates the benefit of future prediction.
- **End-to-end vs model-based:** C1/C2/C3/C4 (learned) vs C5 (MPC) isolates the cost/benefit of an explicit dynamics model.
- **Iterative vs single-step:** C3 (DDPM) vs C4 (MIP) links back to `EXP-001` mechanism question.

### Negative / positive controls

- **MIP (C4):** negative control for long-horizon tasks if WAM does not outperform it.
- **Nonlinear MPC (C5):** positive control for short-horizon, high-precision tasks where the model is known.
- **SmallVLA (C2):** positive control for language-conditioned generalization.

---

## 7. Data

### Sample size

- 100 episodes per (condition, benchmark, seed) × 5 seeds × 5 conditions × 4 benchmarks = **10,000 episodes** total.

### Seeds

`[0, 1, 2, 42, 123]` — fixed across all conditions and benchmarks. Episode initial states from `data/eval_seeds_exp005.json` (committed, hashed).

### Power analysis (pre-registered)

- Paired Wilcoxon, α=0.05 with Bonferroni over the 10 pairwise comparisons per benchmark (5 conditions → C(5,2)=10). Effective α ≈ 0.005 per test.
- Minimal detectable effect ≈ 6 pp at power 0.80 given 5 seeds × 100 episodes and assumed within-seed correlation ρ=0.3.
- If observed std > 13 pp, flag as underpowered and record; do not silently add seeds.

### Data storage

```
experiments/EXP-005-world-models/outputs/
├── <condition>/
│   └── <benchmark>/
│       └── seed_<s>/
│           ├── config.yaml
│           ├── env_info.json
│           ├── checkpoint.pt                (learning-based only)
│           ├── episodes.jsonl               # success, sub-goals, contact, latency
│           ├── sample_frames/               # visual sanity check
│           └── metrics_summary.json
└── analysis/
    ├── success_rates.csv
    ├── subgoal_completion.csv
    ├── latency.csv
    ├── manifold_adherence.csv
    └── figures/
```

---

## 8. Analysis Plan

### Primary analysis

1. **Pairwise comparisons:** Wilcoxon signed-rank test for WAM vs each other condition, per benchmark, with Bonferroni correction across the 10 pairs per benchmark.
2. **Long-horizon / contact-rich focus:** Pre-registered contrast on `StackCube` and `Square`.
3. **Short-horizon focus:** Pre-registered contrast on `Lift` and `PickCube`.
4. **Effect size:** Report Cohen's d for each significant pairwise difference.

### Secondary analysis

1. **Sub-goal completion:** Compare sub-goal completion rates with a repeated-measures ANOVA (condition × benchmark).
2. **Latency vs. success:** Plot success vs. median latency per condition; identify Pareto points.
3. **Data efficiency:** Plot success vs. number of demonstrations for C1–C4.
4. **Manifold adherence:** k-NN distance to expert actions; compare WAM vs VLA vs Diffusion.

### Figures

- `exp005_success_by_benchmark.pdf` — bar plot of mean success ± std, grouped by benchmark.
- `exp005_subgoals.pdf` — stacked sub-goal completion for long-horizon tasks.
- `exp005_latency_pareto.pdf` — success vs. median latency.

---

## 9. Links

- Real-robotics master plan: [`../../docs/real_robotics/README.md`](../../docs/real_robotics/README.md)
- Phase 2 roadmap: [`../../docs/real_robotics/phase2_roadmap.md`](../../docs/real_robotics/phase2_roadmap.md)
- Comparison plan: [`../../docs/comparison_plan.md`](../../docs/comparison_plan.md)
- Methodology: [`../../docs/methodology.md`](../../docs/methodology.md)
