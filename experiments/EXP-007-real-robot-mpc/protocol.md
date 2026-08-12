# EXP-007 — Whole-Body Real-Robot MPC

**Status:** Pre-registered  
**Skill:** experiment-protocol (design → controls → pre-register → execute → verify → refine)  
**Hypothesis under test:** `H_RRMPC` (Full-order whole-body MPC with diffusion warm start outperforms centroidal/SRB approximations and learned policies on stability and task success while maintaining real-time rates)  
**Owner:** Gyanateet  
**Created:** 2026-08-11  

---

## 1. Hypothesis & Prediction

### Hypothesis

**`H_RRMPC`:** For real-robot-like whole-body dynamics (humanoid / quadruped / floating-base manipulation), **full-order nonlinear MPC** achieves the highest task success and lowest fall rate by explicitly modeling all contacts, dynamics, and torque limits. A **diffusion warm-start MPC** can retain most of the success of full-order MPC while running at a higher control frequency by using a learned prior to initialize the solver. A **centroidal / single-rigid-body (SRB) MPC** approximation runs fastest but sacrifices contact fidelity. A pure **learned policy** is the fastest but least stable.

This experiment extends the MPC comparison from toy 2D reaching to true whole-body dynamics, where solver speed, model accuracy, and contact stability are all in tension.

### Prediction (pre-registered, quantitative)

1. **Task success:** `success(Full-order MPC) ≥ success(Diffusion Warm-Start MPC) ≥ success(Centroidal/SRB MPC) ≥ success(Learned Policy)`.
2. **Stability:** Fall rate ordering (lower is better): `Full-order < Diffusion Warm-Start ≈ Centroidal/SRB < Learned Policy`.
3. **Control frequency:** `Centroidal/SRB MPC` ≥ 50 Hz, `Diffusion Warm-Start MPC` ≥ 30 Hz, `Full-order MPC` ≥ 20 Hz, `Learned Policy` ≥ 10 Hz (or native rate).
4. **Contact-rich tasks:** On a reaching/manipulation task with floating base, `Full-order MPC` outperforms `Centroidal/SRB MPC` by ≥ 10 pp.
5. **Latency-quality trade-off:** `Diffusion Warm-Start MPC` is within 3 pp of full-order MPC at ≥ 1.5× the control frequency.

### Falsification conditions

- `H_RRMPC` is falsified if `Full-order MPC` is not the top performer on task success.
- `H_RRMPC` is falsified if `Diffusion Warm-Start MPC` cannot run at ≥ 20 Hz.
- `H_RRMPC` is falsified if `Learned Policy` has a lower or equal fall rate than full-order MPC.

---

## 2. Design

| Element | Specification |
|---------|---------------|
| Design type | Within-subject (same evaluation episodes across conditions), repeated measures |
| Independent variable | Controller (4 levels, see Conditions table) |
| Benchmarks | MuJoCo humanoid (`Humanoid-v4` / `HumanoidBench` stand, walk, reach) and MuJoCo Menagerie quadruped (e.g., Go2 / B2 stand, walk, reach). `RoboDojo` humanoid/quadruped tasks if available. |
| Unit of analysis | One evaluation episode |
| Pairing | All conditions evaluated on identical episode seeds per benchmark |
| Blinding | Scripted task-success and fall predicates (no human rating) |
| Randomization | Episode order shuffled per seed; condition order counterbalanced |

### Procedure (execute phase)

1. Set up the whole-body robot model in MuJoCo or RoboDojo:
   - Humanoid: torso height, joint limits, contact geometry, target end-effector or torso pose.
   - Quadruped: base position / velocity, foot contacts, target velocity or position.
2. Configure each condition once per seed:
   - **C1:** Centroidal/SRB MPC — model the robot as a single rigid body with centroidal dynamics and approximate foot/contact forces.
   - **C2:** Full-order nonlinear MPC — use the full floating-base dynamics with all joint states and contact constraints.
   - **C3:** Diffusion warm-start MPC — train a small diffusion/flow policy to generate an initial trajectory; run full-order MPC refinement from that warm start.
   - **C4:** Learned policy — train a diffusion policy or VLA as a visuomotor end-to-end controller on the same task.
3. Save config snapshots + learned-policy checkpoints to `outputs/<condition>/seed_<s>/`.
4. For each (condition, benchmark, seed), roll out 100 evaluation episodes using a fixed episode seed file (`data/eval_seeds_exp007.json`).
5. Record per-timestep: joint torques, joint limits, contact forces, control frequency, solver iterations, fall flag.
6. Record per-episode: success, fall, control frequency, energy, path length, action trajectory, config hash, git commit.
7. Aggregate to per-(condition, benchmark, seed) means.

### Verify phase

- **MPC solver check:** Full-order and diffusion warm-start MPC must satisfy hard constraints (joint/torque limits) with < 1% violation on a stand task; otherwise the solver is misconfigured.
- **Determinism spot-check:** Re-run 1 seed of C1 and C2 with fixed random seeds; success and fall rate must match within ±2 pp.
- **Control-frequency sanity:** Log control loop time for each condition; values must match the predicted ordering.
- **Fall predicate audit:** Visually inspect 5 random episodes per condition to confirm scripted fall detection agrees with human judgment.

### Refine phase

- If full-order MPC is too slow (< 20 Hz), reduce the horizon or use a faster solver; re-run only if the config change is small enough that the comparison remains fair.
- If diffusion warm start does not improve over cold-start full-order MPC, inspect the warm-start quality (action manifold coverage) and the MPC trust-region settings.

---

## 3. Primary Outcome

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Task success rate** (primary) | Fraction of 100 episodes achieving the task goal without falling | Binary per episode, averaged; mean ± std over 5 seeds |
| **Stability (fall rate)** (co-primary) | Fraction of episodes in which the robot falls | Binary per episode, averaged; mean ± std over 5 seeds |
| **Control frequency** (co-primary) | Median controller update rate in Hz | Per-episode mean; reported as median over 100 episodes and 5 seeds |

Task-success predicates:

- **`Stand`:** robot torso height stays above threshold for ≥ 5 s; no fall.
- **`Walk`:** robot base reaches a target position within the time limit; no fall; feet make appropriate contact.
- **`Reach`:** end-effector (humanoid hand / quadruped mounted arm) reaches within 0.05 m of a target and holds for ≥ 5 steps; no fall.

A **fall** is defined as any of:
- Torso/head height below a task-specific threshold.
- Any non-feet body part contacting the ground.
- Any joint limit or torque limit violation persisting > 0.1 s.

---

## 4. Secondary Outcomes

| Outcome | Definition | Measurement |
|---------|------------|-------------|
| **Joint limit violation rate** | % timesteps with joint position outside limits | Per-step binary, averaged |
| **Torque limit violation rate** | % timesteps with joint torque outside limits | Per-step binary, averaged |
| **Energy (torque-dot-velocity)** | Cumulative mechanical work per episode | Sum over timesteps; lower is more efficient |
| **Path length** | Integrated base / end-effector travel distance | Per episode, averaged |
| **Solver iterations** | Number of SQP/interior-point iterations per MPC step | Median over episode |
| **Inference / solve latency** | Wall-clock time for one control step | Median + p95 over 1000 steps |
| **Time to task completion** | Steps to first success criterion | Median over successful episodes |

---

## 5. Conditions Table

| ID | Condition | Family | Model | Warm start? | Constraints | Notes |
|----|-----------|--------|-------|-------------|-------------|-------|
| C1 | **Centroidal / SRB MPC** | MPC | Single rigid body / centroidal dynamics | No | Hard (friction cone, CoM limits) | Fastest; loses contact/joint detail |
| C2 | **Full-order Nonlinear MPC** | MPC | Full floating-base dynamics with all joints and contacts | No | Hard (joints, torques, contacts, friction) | Most accurate; slowest cold start |
| C3 | **Diffusion Warm-Start MPC** | Hybrid | Full floating-base dynamics | Yes (diffusion/flow generates initial trajectory) | Hard (same as C2) | Diffusion prior + MPC refinement |
| C4 | **Learned Policy** | Learned | Implicit in diffusion / VLA policy | N/A | None (soft, from data) | Reactive end-to-end visuomotor or state policy |

**Held constant across conditions:**
- Same robot model and contact parameters (all conditions use the same MuJoCo/RoboDojo model).
- Same cost/reward terms where applicable (target tracking, regularization, energy).
- Same control frequency target when possible; actual achieved frequency is a measured outcome.
- Same evaluation episode seeds (paired design).
- Same action space: joint torques or joint position targets.

---

## 6. Controls & Ablations

### Controls (fairness)

- **Same dynamics model:** C1–C3 use the same analytical/identified dynamics (or the same MuJoCo model), differing only in approximation and warm start.
- **Same cost function:** Where C1 uses a centroidal cost, map it to the same high-level objective (e.g., target CoM height, base velocity, end-effector error).
- **Same solver tolerances:** MPC convergence tolerances are identical for C1–C3 unless required by numerical stability.
- **Same training data for C4:** If C4 is a diffusion policy, train on the same demonstrations used to generate the warm-start prior for C3.
- **Same evaluation seeds and perturbations.**

### Ablations embedded

- **Model complexity gradient:** C1 → C2 shows the cost/benefit of moving from centroidal to full-order dynamics.
- **Warm-start contribution:** C2 vs C3 isolates the value of a learned diffusion prior for MPC initialization.
- **Model vs learned:** C2/C3 vs C4 isolates the value of an explicit dynamics model and hard constraints.
- **Vision vs state (exploratory):** If time permits, run C4 with both state and image observations; record the gap.

### Negative / positive controls

- **C1 (Centroidal/SRB):** positive control for speed; negative control for contact-rich tasks.
- **C2 (Full-order MPC):** positive control for task success and stability.
- **C4 (Learned Policy):** negative control for safety/stability; positive control for latency.

---

## 7. Data

### Sample size

- 100 episodes per (condition, benchmark, seed) × 5 seeds × 4 conditions × 3 benchmarks = **6,000 episodes** total.

### Seeds

`[0, 1, 2, 42, 123]` — fixed across conditions and benchmarks. Episode initial states from `data/eval_seeds_exp007.json` (committed, hashed).

### Power analysis (pre-registered)

- Paired Wilcoxon, α=0.05 with Bonferroni over the 6 pairwise comparisons per benchmark (4 conditions → C(4,2)=6). Effective α ≈ 0.008.
- Minimal detectable effect ≈ 7 pp in success rate at power 0.80 given 5 seeds × 100 episodes and assumed within-seed correlation ρ=0.3.
- If observed success std > 14 pp, flag as underpowered and record.

### Data storage

```
experiments/EXP-007-real-robot-mpc/outputs/
├── <condition>/
│   └── <benchmark>/
│       └── seed_<s>/
│           ├── config.yaml
│           ├── env_info.json
│           ├── checkpoint.pt              (learning-based only)
│           ├── episodes.jsonl             # success, fall, control freq, violations
│           ├── per_step.jsonl             # torque, contact, solver iters
│           ├── sample_videos/             # sanity visual check
│           └── metrics_summary.json
└── analysis/
    ├── success_rate.csv
    ├── fall_rate.csv
    ├── control_frequency.csv
    ├── violations.csv
    ├── solver_iterations.csv
    └── figures/
```

---

## 8. Analysis Plan

### Primary analysis

1. **Success and fall rate:** Report mean ± std over seeds per (condition, benchmark).
2. **Pairwise comparisons:** Wilcoxon signed-rank for each pair of conditions, per benchmark, with Bonferroni correction.
3. **Control frequency:** Report median and p95 control frequency; identify which conditions meet the real-time gate (≥ 20 Hz).
4. **Pareto analysis:** Plot task success vs. control frequency; identify non-dominated conditions.

### Secondary analysis

1. **Constraint violations:** ANOVA on joint/torque violation rates across conditions.
2. **Energy efficiency:** Compare cumulative energy across conditions.
3. **Solver iteration reduction:** For C3, compare solver iterations with and without warm start.
4. **Contact-rich vs locomotion:** Split results by task type and test whether the full-order advantage is larger in contact-rich tasks.

### Figures

- `exp007_success_fall.pdf` — grouped bar plot of success and fall rates by benchmark.
- `exp007_control_freq.pdf` — control frequency distribution per condition.
- `exp007_pareto.pdf` — success vs. control frequency Pareto plot.
- `exp007_violations.pdf` — joint/torque limit violation rates.
- `exp007_solver_iters.pdf` — solver iterations for MPC conditions.

---

## 9. Links

- Real-robotics master plan: [`../../docs/real_robotics/README.md`](../../docs/real_robotics/README.md)
- Phase 2 roadmap: [`../../docs/real_robotics/phase2_roadmap.md`](../../docs/real_robotics/phase2_roadmap.md)
- Comparison plan: [`../../docs/comparison_plan.md`](../../docs/comparison_plan.md)
- Methodology: [`../../docs/methodology.md`](../../docs/methodology.md)
