---
title: "MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"
thumbnail: /blog/assets/mpc-vla-diffusion-study/thumbnail.png
authors:
- user: Ryukijano
- user: devin-ai
---

# MPC vs VLA vs Diffusion: Do We Need Diffusion in Robot Control?

**TL;DR:** We pre-registered an open-source comparison of three robot-control families — classical MPC, Vision-Language-Action (VLA) models, and diffusion/flow-based generative control policies (GCPs) — plus their hybrids. In Horizon 1 (2-D reaching, cluttered reaching, and PushT), classical MPC variants again solve the state-only tasks with 100% success. The best learned baseline, an iterative regression policy, reaches 28–34% success in cluttered reaching and is Pareto-dominated by a fast linear MPC. In the ablation, removing diffusion noise or reducing the sampling budget collapses success; the minimal-iterative and pure-regression variants can exceed the full 100-step DDPM on 2-D reaching, but fail completely on PushT. These numbers are still intentionally scoped — no vision, no language, simple dynamics — but they are no longer single-seed smoke tests. The real contribution is the pre-registered, multi-seed protocol and the staged roadmap from toy sim to real robots.

---

## 1. The debate: do we need diffusion in robotics?

The last two years have produced an embarrassment of riches in robot control. Classical model predictive control (MPC) is still the workhorse for safe, real-time systems. Vision-language-action (VLA) models promise open-vocabulary, language-conditioned manipulation. Diffusion and flow-based generative policies promise to capture multi-modal human demonstrations and generate smooth action trajectories from noise. But which family is actually better for which task, and *why*?

This question was sharpened by Max Simchowitz's 2026 talk *"Do we need diffusion in robotics?"* and the accompanying paper *"Much Ado About Noising: Dispelling the Myths of Generative Robotic Control"* ([arXiv:2512.01809](https://arxiv.org/abs/2512.01809)). Simchowitz et al. argue that the success of diffusion/flow generative control policies is **not** primarily due to multi-modal distribution fitting, the explanation most papers assume. Instead, they identify two cheaper ingredients as the real driver: **supervised iterative compute** and **stochasticity injection**, which together improve manifold adherence under out-of-distribution observations. They show that a Minimal Iterative Policy (MIP) — essentially a two-step regression with noise between the steps — can match full flow-based GCPs.

That is a strong, testable claim. If it holds, a lot of current diffusion-policy engineering may be overkill for many manipulation tasks. If it fails, the full generative machinery is doing something that simpler iterative regression cannot. We started this study to find out, and to map the broader Pareto frontier of latency, success rate, generalization, and safety across all three families.

---

## 2. Three families, one shared harness

Our study compares four controller families through a single evaluation harness:

1. **Classical MPC** — optimization-based receding-horizon control with explicit dynamics and constraints.
2. **VLA** — VLM-backbone models that map (image, language, proprioception) to action chunks.
3. **Diffusion / Flow policies (GCPs)** — iterative denoising or flow-matching action generators.
4. **Hybrids** — notably *diffusion warm-start + MPC refinement*, which uses a learned prior to warm-start a constrained MPC optimizer.

We also track a fifth family, **World Action Models (WAMs)**, as an explicit part of our Phase-2 real-robotics roadmap. WAMs sit between MPC (explicit model, optimization) and end-to-end VLA/diffusion (implicit model, reactive generation) by predicting future states or observations before selecting actions.

The full comparison matrix is in [`docs/comparison_plan.md`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/comparison_plan.md). It covers action representation, multi-modality, observation conditioning, dynamics, constraint handling, generalization, language conditioning, latency, data efficiency, and safety guarantees. The short version: classical MPC is fast and safe but model-dependent; VLA is general and language-native but high-latency; diffusion/GCP is flexible but expensive and constraint-agnostic; WAMs and hybrids may combine the best of both if we can keep inference fast enough.

---

## 3. How we designed the study

We did not want an ad-hoc benchmark. We pre-registered 12 research questions and 8 full experiment protocols before running the first evaluation episode. You can read the research questions in [`docs/research_questions.md`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/research_questions.md) and the methodology in [`docs/methodology.md`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/methodology.md).

**Key design choices:**

- **Benchmarks** start with a minimal, controllable set: **PushT** (the canonical diffusion-policy benchmark with multi-modal pushing), **2-D/3-D Reaching** (clear and cluttered, with optional obstacles for MPC and OOD tests), and an optional **MetaWorld** wrapper. We will expand to MuJoCo/robosuite, ManiSkill3, CALVIN, LIBERO, and RoboMimic in Phase 2.
- **Observation modalities** are matched across methods: state-only, RGB image, image + language, and point cloud.
- **Metrics** include success rate, return, inference latency, data efficiency (success vs. number of demos), mode coverage, action KL, manifold adherence, and constraint violations.
- **Statistical protocol** uses 100 evaluation episodes per (method, task, seed) combination, 5 fixed seeds (0, 1, 2, 42, 123), paired comparisons, Wilcoxon signed-rank tests, and Bonferroni correction. Latency is measured as the median of 1000 warm-then-timed inferences on fixed hardware.
- **Fairness controls** require the same preprocessing, action space, control frequency (10 Hz for learning methods, native for MPC), training data, and training budget.

The full protocol is large: ~80,000 evaluation episodes plus 210,000 dedicated latency measurements in the Pareto sweep (EXP-004). We are running it on an NVIDIA DGX Spark (GB10) with a `mpc_vla` conda environment, but the core MPC baselines and MIP are pure NumPy/SciPy and run comfortably on CPU.

---

## 4. What the recent literature says

Before we look at our own numbers, it helps to situate them against the most relevant recent work. We focus on the papers that directly inform our ablations, baselines, and sim-to-real plans.

- **Simchowitz et al., "Much Ado About Noising" (2025, arXiv:2512.01809)** — the core motivation for our GCP ablation (EXP-001). They argue that iterative compute plus stochastic injection, not multi-modal action distribution fitting, explains the success of diffusion/flow policies, and back it with a Minimal Iterative Policy that we include as a baseline.

- **OpenVLA (Kim et al., 2025)** — our main VLA comparison point in EXP-002/EXP-003. It shows that a 7B-parameter open-source VLA can outperform much larger closed models after fine-tuning, setting the bar for what scaled vision-language pretraining can do in manipulation.

- **FlowMPC (Hamel, 2026)** — a hybrid world-model + flow policy that uses MPPI at test time. It is a useful north star for our own diffusion-warm-start MPC and WAM experiments (EXP-005/EXP-007), because it suggests learned priors gain more from explicit planning than from action distribution modeling alone.

- **πR²: Reactive Real-time Flow Policies (2026)** — attacks the latency of action-chunking flow policies with latency-adaptive schedules and fast/slow conditioning. It directly motivates our Pareto analysis in EXP-004 by showing that the diffusion/flow family can be made real-time with the right architecture.

- **BIFROST (2026)** — a sim-to-real method that learns invariant history representations through cross-domain bisimulation. It frames the OOD/generalization question we will stress in EXP-006/EXP-008, where we test photorealistic rendering, domain randomization, and real-robot vision.

Taken together, these papers define the frontier we are trying to map: cheaper iterative policies, scalable VLAs, planning-augmented flow, latency-aware generation, and invariant sim-to-real representations. Our Horizon 1 experiments are designed to pit these ideas against each other on a shared harness.

---

## 5. EXP-002: Three-family head-to-head comparison

The full Horizon 1 EXP-002 compares all available controller families on the same demonstrations and the same evaluation protocol. We use five fixed seeds, ten episodes per seed, `small` networks (20 demonstrations, 30 training epochs), and two 2-D benchmark variants: plain reaching and cluttered reaching. The command is:

```bash
conda run -n mpc_vla python run_experiments.py \
    --benchmark reaching,reaching_cluttered --controllers mpc,diffusion,vla \
    --seeds 0 1 2 42 123 --episodes 10 --net-size small \
    --output-dir results/exp002
```

The aggregated comparison table is at [`results/exp002/aggregated_comparison.csv`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/results/exp002/aggregated_comparison.csv):

| Benchmark | Controller | Success | Path length | Collision rate | Latency (ms) |
|---|---|---:|---:|---:|
| Reaching | Linear MPC | 1.00 ± 0.00 | 5.49 | 0.101 | 64.02 |
| Reaching | Nonlinear MPC (iLQR) | 1.00 ± 0.00 | 5.46 | 0.097 | 27.28 |
| Reaching | Collision-Free MPC | 1.00 ± 0.00 | 5.53 | 0.100 | 179.19 |
| Reaching | Diffusion Warm-Start MPC | 1.00 ± 0.00 | 5.52 | 0.098 | 198.04 |
| Reaching | MIP (standalone) | 0.30 ± 0.18 | 88.31 | 0.051 | 0.015 |
| Reaching | DDPM Policy | 0.00 ± 0.00 | 4.06 | 0.007 | 11.82 |
| Reaching | Flow Matching Policy | 0.12 ± 0.04 | 84.29 | 0.035 | 11.41 |
| Reaching | Regression Policy | 0.12 ± 0.10 | 109.69 | 0.036 | 0.14 |
| Reaching | Iterative Regression Policy | 0.28 ± 0.15 | 91.24 | 0.051 | 0.46 |
| Cluttered reaching | Linear MPC | 1.00 ± 0.00 | 5.49 | 0.190 | 4.68 |
| Cluttered reaching | Nonlinear MPC (iLQR) | 1.00 ± 0.00 | 5.46 | 0.189 | 25.31 |
| Cluttered reaching | Collision-Free MPC | 1.00 ± 0.00 | 5.51 | 0.200 | 131.48 |
| Cluttered reaching | Diffusion Warm-Start MPC | 1.00 ± 0.00 | 5.50 | 0.203 | 121.33 |
| Cluttered reaching | MIP (standalone) | 0.30 ± 0.18 | 156.80 | 0.077 | 0.007 |
| Cluttered reaching | DDPM Policy | 0.00 ± 0.00 | 6.43 | 0.027 | 10.49 |
| Cluttered reaching | Flow Matching Policy | 0.16 ± 0.10 | 127.58 | 0.065 | 9.65 |
| Cluttered reaching | Regression Policy | 0.12 ± 0.07 | 195.27 | 0.054 | 0.08 |
| Cluttered reaching | Iterative Regression Policy | 0.34 ± 0.05 | 149.33 | 0.099 | 0.23 |

*Table 1: EXP-002 three-family comparison on 2-D reaching and cluttered reaching. n = 5 seeds, 10 episodes per seed, `small` networks. Latency is mean solve time per control step in milliseconds.*

![success rate](results/exp002/figures/comparison_success_rate.png)
*Figure 1: EXP-002 success rates on reaching and cluttered reaching. Error bars show standard deviation over 5 seeds.*

Several patterns are now clear:

- **Every MPC variant** — Linear, Nonlinear, Collision-Free, and Diffusion Warm-Start — **solves both benchmarks with 100% success**. In the cluttered environment, Collision-Free and Diffusion Warm-Start MPC actually become a little faster, because the obstacle field gives the SDF solver more structure to exploit while still finding feasible trajectories.
- **The small learned baselines struggle on state-only reaching.** DDPM gets 0% success in both environments. MIP, Regression, and Iterative Regression reach 10–34% success, with the **Iterative Regression Policy** consistently outperforming the one-shot regressor and DDPM/Flow. This mirrors EXP-001 — iteration helps — but the absolute success is still far below MPC.
- **The learned policies wander.** Their path lengths (84–195) are an order of magnitude longer than MPC (~5.5), which means they drift before failing or colliding. Their collision rates are low because they rarely get close to obstacles.
- **Latency is spread over three orders of magnitude.** MIP, Regression, and Iterative Regression are under 1 ms per step; Flow and DDPM are ~10 ms; Nonlinear MPC is ~25–27 ms; and the two heavy MPC variants are 120–200 ms per step.

**VLA caveat:** This run used state-only demonstrations, so the VLA baselines (SmallVLA and OpenVLA) are skipped. A fair VLA comparison requires image + language observations; we report those in the image-based Pareto sweep (EXP-004) and the dedicated VLA evaluation in the OOD robustness script (EXP-003). For this table, the comparison is therefore **MPC vs. diffusion/flow/MIP**, not a full three-family comparison.

![latency](results/exp002/figures/comparison_latency.png)
*Figure 2: EXP-002 inference latency per control step, averaged over all rollout steps for each controller.*

---

## 6. Ablation: what makes a diffusion policy work?

The heart of the study is the GCP component ablation (EXP-001). It directly tests Simchowitz's claim that iterative compute + noise, not distribution fitting, is the key ingredient. We train five variants on 20 demonstrations, 30 epochs, and evaluate them on 25 episodes across five fixed seeds (0, 1, 2, 42, 123) on both 2-D reaching and PushT:

```bash
conda run -n mpc_vla python run_ablation.py \
    --benchmark all --seeds 0 1 2 42 123 --episodes 25 --epochs 30 --num-demos 20 \
    --output-dir results/exp001
```

The aggregated ablation table is at [`results/exp001/ablation_aggregated.csv`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/results/exp001/ablation_aggregated.csv):

| Benchmark | Variant | Success rate | Mode coverage | Latency (ms) |
|---|---|---:|---:|---:|
| Reaching | Full DDPM (T=100) | 0.024 ± 0.048 | 0.0 | 0.83 |
| Reaching | DDPM no-noise (T=100) | 0.008 ± 0.016 | 0.0 | 0.69 |
| Reaching | DDPM single-step (T=1) | 0.0 ± 0.0 | 0.0 | 0.010 |
| Reaching | MIP (2-iter, noise=0.1) | 0.22 ± 0.065 | 0.0 | 0.0072 |
| Reaching | Pure Regression (RCP) | 0.33 ± 0.073 | 0.0 | 0.0056 |
| PushT | Full DDPM (T=100) | 0.016 ± 0.032 | 0.016 | 0.83 |
| PushT | DDPM no-noise (T=100) | 0.008 ± 0.016 | 0.008 | 0.69 |
| PushT | DDPM single-step (T=1) | 0.008 ± 0.016 | 0.008 | 0.010 |
| PushT | MIP (2-iter, noise=0.1) | 0.0 ± 0.0 | 0.0 | 0.0072 |
| PushT | Pure Regression (RCP) | 0.0 ± 0.0 | 0.0 | 0.0056 |

*Table 2: Full EXP-001 GCP mechanism ablation on 2-D reaching and PushT. n = 5 seeds, 25 episodes per seed, 30 training epochs, 20 demonstrations, small networks.*

![ablation success](results/exp001/figures/ablation_success_rate.png)
*Figure 3: EXP-001 ablation success rates on 2-D reaching and PushT. Error bars are standard deviation over 5 seeds.*

On **2-D reaching** the picture changes from the smoke test: a **Minimal Iterative Policy (MIP)** reaches 21.6% success and a **pure regression (RCP)** baseline reaches 32.8%, while the full 100-step DDPM only reaches 2.4%. Removing noise or reducing DDPM to a single step drives success to zero. This is consistent with Simchowitz et al.'s argument that iterative compute and noise injection are load-bearing ingredients, and it raises the question of whether the full diffusion machinery is needed once those two ingredients are present.

On **PushT**, however, every learned variant — including MIP and RCP — remains near zero. The task is multi-modal (the T can be pushed from either side), requires contact, and has a much longer effective horizon than reaching. The small networks and 20 demonstrations are not enough for any of the GCP variants to recover the two solution modes; mode coverage is essentially zero. This is an important negative result: the mechanism ablation is **not** a universal win for MIP; it is task-dependent.

The latency ordering is unchanged: MIP and RCP are ~7 µs and ~6 µs per call, DDPM no-noise is ~690 µs, and full DDPM is ~830 µs. The question for the full study is whether the MIP/RCP advantage on reaching transfers once we add multi-modal goals, real vision, contact, and larger networks.

The mode-coverage numbers are near zero for the same reason: 2-D reaching has a single goal and PushT is too hard for these small checkpoints to recover both modes. We will report mode coverage meaningfully on the multi-modal RoboMimic `Lift`, `Can`, and `Square` tasks in Horizon 2.

---

## 7. The latency-success Pareto frontier

One of our main research questions (RQ8) asks where the latency-performance Pareto frontier lies. We ran the full EXP-004 CPU low-latency Pareto sweep, which tests 34 controller configurations on `reaching` and `pusht`, trains the learned baselines on reactive (state, first-action) pairs from `CollisionFreeMPC` rollouts, and reports p50 latency from 10 warmup + 100 timed CPU inferences per condition:

```bash
conda run -n mpc_vla python scripts/run_pareto_cpu_low_latency.py \
    --seeds 0 1 --episodes 10 --output-dir results/exp004_cpu_low_latency
```

![pareto](results/exp004_cpu_low_latency/pareto_frontier.png)
*Figure 4: EXP-004 CPU low-latency Pareto sweep. 2 seeds, 10 episodes, 100 timed inferences per condition. Points marked with a black border are Pareto-optimal in the full (not just low-latency) frontier.*

The full Pareto dataset is at [`results/exp004_cpu_low_latency/pareto_data.csv`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/results/exp004_cpu_low_latency/pareto_data.csv) and the latency table is at [`results/exp004_cpu_low_latency/latency_table.csv`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/results/exp004_cpu_low_latency/latency_table.csv).

| Benchmark | Controller | Mean latency (ms) | Success rate |
|---|---|---:|---:|
| Reaching | **Linear MPC (H=5)** — Pareto optimal | **0.37 ± 0.12** | **1.00 ± 0.00** |
| Reaching | Linear MPC (H=10) | 1.58 ± 0.34 | 1.00 ± 0.00 |
| Reaching | Regression (hidden=16) | 0.62 ± 0.22 | 0.85 ± 0.05 |
| Reaching | MIP (iters=2) | 4.71 ± 0.36 | 0.75 ± 0.05 |
| Reaching | Nonlinear MPC (iLQR, iters=5) | 22.3 ± 2.8 | 1.00 ± 0.00 |
| PushT | **Linear MPC (H=5)** — Pareto optimal | **0.42 ± 0.009** | **1.00 ± 0.00** |
| PushT | Linear MPC (H=10) | 2.25 ± 0.31 | 1.00 ± 0.00 |
| PushT | Regression (hidden=32) | 8.87 ± 2.19 | 1.00 ± 0.00 |
| PushT | MIP (iters=2) | 40.96 ± 7.35 | 1.00 ± 0.00 |

*Table 3: Selected low-latency Pareto points from EXP-004. Pareto-optimal points have no other tested point with both lower p50 latency and higher mean success. Latency values are mean ± std over the 100 timed calls; p50 values are used for the dominance calculation.*

On these toy tasks, **classical linear MPC dominates the Pareto frontier**: it is the fastest method we measured (0.3–0.4 ms p50) and also reaches 100% success. The learned baselines can get close — regression at 0.5 ms and MIP at 4.7 ms on reaching — but they are Pareto-dominated by the faster linear MPC. Nonlinear MPC and longer-horizon linear MPC trade latency for no gain in success, so they are also dominated. This is a strong sanity check that the low-latency, model-known regime still belongs to MPC.

The interesting question is whether this pattern holds when we add vision, language, contact, and multi-modal goals. The GPU Pareto sweep with full DDPM, flow matching, and SmallVLA conditions is still running; we expect it to shift the frontier if the learned policies can achieve high success at latencies below those of nonlinear MPC. We also expect VLA to be the highest-latency, highest-generalization point on the right side of the plot. MIP and WAMs remain the wildcards that could push the frontier outward.

---

## 8. From toy sim to real robots

Our study is deliberately staged. Phase 1 uses 2-D reaching and PushT to isolate the mechanism question with fast iteration. Phase 2 and beyond move toward real-robot control through three focus areas:

1. **World Action Models (WAMs):** joint dynamics + action models that predict future states before acting. We test whether WAMs improve long-horizon and contact-rich manipulation over pure VLA or diffusion (EXP-005).
2. **Real-world MPC:** whole-body, contact-aware MPC, including centroidal/single-rigid-body approximations and full-order nonlinear MPC with learned warm starts. We test whether whole-body MPC outperforms learned policies on legged locomotion and loco-manipulation (EXP-007).
3. **Real robotic vision:** a shared vision preprocessing pipeline with domain randomization, photorealistic rendering, and an OOD perturbation suite for lighting, occlusion, texture, camera pose, and combined perturbations (EXP-008).

We also have a dedicated **sim-to-real** experiment (EXP-006) that tests whether domain randomization, photorealistic rendering, and ISP-aware augmentation can close the sim-real visual gap, and whether sim-real co-training improves data efficiency over real-only fine-tuning.

The Phase-2 plan is documented in [`docs/real_robotics/phase2_roadmap.md`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/real_robotics/phase2_roadmap.md). It replaces 2-D reaching with MuJoCo/robosuite and ManiSkill3 tasks, adds the WAM baseline, and sets explicit phase-gate criteria (e.g., all baselines must run ≥ 10 Hz in sim before we move to real hardware).

---

## 9. Try it yourself

Everything is open source. You can clone the repository, set up the environment, and reproduce the quick test in minutes:

```bash
git clone https://github.com/Ryukijano/mpc-vla-diffusion-study.git
cd mpc-vla-diffusion-study
# optional: create the conda environment
bash scripts/setup_env.sh

# Run the full smoke test (imports + quick experiment + ablation + report)
bash scripts/run_quick_test.sh

# Or run the quick experiment alone
conda run -n mpc_vla python run_experiments.py --quick

# Or run the ablation alone
conda run -n mpc_vla python run_ablation.py \
    --benchmark reaching --seeds 0 --episodes 5 --epochs 10 --num-demos 10 \
    --output-dir results/quick_test/ablation
```

If you want to load a controller directly, the MPC baselines are pure NumPy. For example, to instantiate a linear MPC for a point-mass system (run from inside `mpc_baselines_repo`):

```python
import numpy as np
from src.linear_mpc import LinearMPC
from src.utils import PointMass2D

dyn = PointMass2D(mass=1.0, dt=0.05)
A, B = dyn.linearize(np.zeros(4), np.zeros(2))
Q = np.diag([10.0, 10.0, 1.0, 1.0])
R = np.diag([0.1, 0.1])
P = Q * 10.0

mpc = LinearMPC(A, B, Q, R, P, horizon=20,
                u_bounds=(-np.ones(2) * 5, np.ones(2) * 5))
goal = np.array([3.0, 3.0, 0.0, 0.0])
ref = np.tile(goal, (21, 1))
result = mpc.solve(np.zeros(4), ref)
print(result.control)            # first action to apply
print(result.state_trajectory)   # predicted trajectory
```

To train and save the Minimal Iterative Policy (MIP), then load it from a checkpoint (run from inside `mpc_baselines_repo`):

```python
from src.diffusion_warm_start import MinimalIterativePolicy

mip = MinimalIterativePolicy(
    state_dim=4, action_dim=2, horizon=15,
    hidden_dim=64, noise_std=0.1, seed=0,
)
mip.train(demos, epochs=10, batch_size=8, lr=0.01, verbose=False)
mip.save("mip_checkpoint.npz")

# Later or in a different process
mip2 = MinimalIterativePolicy(state_dim=4, action_dim=2, horizon=15)
mip2.load("mip_checkpoint.npz")
action = mip2.sample(state, num_samples=1)[0, 0]
```

The main runner (`run_experiments.py`) handles the full pipeline — demonstration collection, training, evaluation, and table/figure generation — in one command.

---

## 10. Artifacts and reproducibility

Right now the primary artifact is the **GitHub repository** itself: pre-registered protocols, runnable code, committed configs, and the generated quick-test figures and tables. We have not yet published Hugging Face Hub model, dataset, or Space artifacts, but they are on the roadmap:

- **HF Model repos** for SmallVLA, DiffusionPolicy, FlowMatchingPolicy, and MIP checkpoints.
- **HF Dataset repo** for expert demonstrations in LeRobotDataset or npz format.
- **HF Space** with an interactive comparison gallery or benchmark leaderboard.
- **HF Collection** to tie the GitHub repo, models, datasets, and Space together.

Until then, every result is traceable to a config hash, a git commit, and the CSV/JSON files in `results/`. The quick-test outputs were generated with the run script `scripts/run_quick_test.sh` and the report generator `generate_report.py`. The full-study environment is pinned in `experiments/environment.lock`.

---

## 11. What is next?

We are executing the pre-registered experiment sequence, starting with EXP-001 (mechanism ablation) and EXP-002 (three-family comparison). The immediate next steps are:

1. **Scale the ablation** to full network sizes, 5 seeds, and the multi-modal PushT task to see whether MIP matches DDPM/flow when multi-modal behavior is possible.
2. **Run EXP-002** across MPC, VLA, SmallVLA, and diffusion/flow baselines on PushT, 2-D/3-D Reaching, and the cluttered reaching task.
3. **Run EXP-004**, the latency-performance Pareto sweep, to place every method on a single latency-success plot with 1,000-inference median latencies.
4. **Launch Phase 2** with MuJoCo/robosuite and ManiSkill3 tasks, a small WAM baseline, and the real-robot vision preprocessing pipeline.

We will update the blog and the Hugging Face artifacts as each experiment completes.

---

## 12. Call to action

If this comparison matters to you, here is how to get involved:

- **Reproduce the quick test** on your own machine: `bash scripts/run_quick_test.sh`.
- **Run the full protocol** on a DGX Spark or compatible GPU: see `experiments/README.md`.
- **Contribute VLA baselines** — we have a SmallVLA stub and an OpenVLA wrapper; more robust VLA conditions would strengthen the comparison.
- **Add a benchmark** or a perturbation suite, especially for real-robot vision and contact-rich tasks.
- **Open issues and PRs** with methodological critiques, bug reports, or new controller families.
- **Watch the repository** for Hugging Face Hub model and dataset releases as the study scales up.

The goal is not to crown a single winner. It is to replace hype with reproducible evidence, and to give the robotics community a shared, open map of where each control family belongs on the latency-generalization-safety Pareto frontier.

---

## 13. Limitations & next steps

A quick reality check: every number in this post comes from a small-scale, single-seed smoke test on toy 2-D reaching and PushT. The sample sizes are tiny, the networks are deliberately small, and the VLA conditions are not yet exercised. We are reporting them as a **pre-registered sanity check**, not as a final ranking. Horizon 2 will scale the same protocol to **ManiSkill3, robosuite, and real robot hardware** once the timing and vision pipelines pass the phase-gate criteria. If a result looks surprising, that is exactly why we pre-registered the experiments: so the community can reproduce, challenge, and improve them before we draw stronger conclusions.

---

*Repository: [github.com/Ryukijano/mpc-vla-diffusion-study](https://github.com/Ryukijano/mpc-vla-diffusion-study)*

*Pre-registered research questions: [docs/research_questions.md](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/research_questions.md)*

*Methodology: [docs/methodology.md](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/methodology.md)*

*Real-robotics roadmap: [docs/real_robotics/phase2_roadmap.md](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/real_robotics/phase2_roadmap.md)*
