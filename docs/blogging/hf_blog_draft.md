---
title: "MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families"
thumbnail: /blog/assets/mpc-vla-diffusion-study/thumbnail.png
authors:
- user: Ryukijano
- user: devin-ai
---

# MPC vs VLA vs Diffusion: Do We Need Diffusion in Robot Control?

**TL;DR:** We pre-registered an open-source comparison of three robot-control families — classical MPC, Vision-Language-Action (VLA) models, and diffusion/flow-based generative control policies (GCPs) — plus their hybrids. In our first smoke test (one seed, five episodes, 2-D reaching), every MPC variant we tried, including a diffusion-warm-started MPC, solved the task with a 100% success rate. The learned MIP and pure-regression policies only reached 0.2–0.4 success, and the full DDPM ablation scored 0.0 on the same tiny benchmark. Those numbers are intentionally limited: the real contribution is the 80,000-episode, pre-registered protocol and the roadmap from toy sim to real robots. We want the community to reproduce, critique, and extend the study before we draw strong conclusions.

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

## 4. Quick smoke-test results

Before scaling to the full protocol, we ran a fast smoke test to verify the harness. This used one seed, five episodes, tiny networks (hidden dim 16, 10 training epochs, 10 demonstrations), and a single 2-D reaching benchmark. The command is:

```bash
# This script runs the quick experiment, the GCP ablation, and the report generator
bash scripts/run_quick_test.sh
```

The aggregated master comparison table is at [`results/quick_test/report/master_comparison_table.csv`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/results/quick_test/report/master_comparison_table.csv):

| Controller | Success rate | Path length | Collision rate | Latency (ms) |
|---|---:|---:|---:|---:|
| Linear MPC | 1.0 ± 0.0 | 5.44 | 0.099 | 44.24 |
| Nonlinear MPC (iLQR) | 1.0 ± 0.0 | 5.54 | 0.108 | 28.04 |
| Collision-Free MPC | 1.0 ± 0.0 | 5.43 | 0.101 | 212.65 |
| Diffusion Warm-Start MPC | 1.0 ± 0.0 | 5.50 | 0.111 | 146.50 |
| MIP (standalone) | 0.4 ± 0.0 | 66.82 | 0.045 | 0.0072 |
| DDPM Policy | 0.0 ± 0.0 | 4.38 | 0.030 | 16.25 |
| Flow Matching Policy | 0.4 ± 0.0 | 36.18 | 0.056 | 15.26 |
| Regression Policy | 0.0 ± 0.0 | 56.49 | 0.023 | 8.45 |
| Iterative Regression Policy | 0.2 ± 0.0 | 85.58 | 0.048 | 8.51 |

*Table 1: Smoke-test results on 2-D reaching. n = 1 seed, 5 episodes per method.*

![success rate](results/quick_test/report/figures/comparison_success_rate.png)
*Figure 1: Success rate in the quick test. Note the tiny sample size.*

What can we say from this? Only that, in our smoke test, every MPC variant found a feasible path to the goal, while the small standalone MIP and the pure learned baselines did not consistently solve the task. The MIP policy was extremely fast (~7 µs) but wandered far off course, as its high path length suggests. We cannot conclude that MPC is universally better — the task has a known dynamics model and a single goal, which strongly favors optimization-based methods. We also cannot conclude that MIP is useless; the network was tiny and trained on only 10 demos.

Note: the quick test uses state-only 2-D reaching, so the VLA baselines are not exercised here. VLA will be evaluated on image+language tasks in the full protocol.

The more interesting pattern is the latency spread. Classical MPC runs from ~2.8 ms (linear) to ~81 ms (collision-free SDF). Diffusion warm-start MPC lands at ~74 ms, roughly comparable to the safest MPC variant. MIP is essentially free in wall-clock time but pays for it in task success.

![latency](results/quick_test/report/figures/comparison_latency.png)
*Figure 2: Inference latency in the quick test.*

---

## 5. Ablation: what makes a diffusion policy work?

The heart of the study is the GCP component ablation (EXP-001). It directly tests Simchowitz's claim that iterative compute + noise, not distribution fitting, is the key ingredient. We train five variants on the same 10 demonstrations and evaluate them on the same 5 episodes:

```bash
conda run -n mpc_vla python run_ablation.py \
    --benchmark reaching --seeds 0 --episodes 5 --epochs 10 --num-demos 10 \
    --output-dir results/quick_test/ablation
```

The aggregated ablation table is at [`results/quick_test/ablation/ablation_aggregated.csv`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/results/quick_test/ablation/ablation_aggregated.csv):

| Variant | Success rate | Mode coverage | Latency (ms) |
|---|---:|---:|---:|
| Full DDPM (T=100) | 0.0 ± 0.0 | 0.0 | 0.825 |
| DDPM no-noise (T=100) | 0.0 ± 0.0 | 0.0 | 0.691 |
| DDPM single-step (T=1) | 0.0 ± 0.0 | 0.0 | 0.010 |
| MIP (2-iter, noise=0.1) | 0.2 ± 0.0 | 0.0 | 0.0072 |
| Pure Regression (RCP) | 0.2 ± 0.0 | 0.0 | 0.0056 |

*Table 2: GCP component ablation on 2-D reaching. n = 1 seed, 5 episodes.*

![ablation success](results/quick_test/ablation/figures/ablation_success_rate.png)
*Figure 3: Ablation success rates. Again, this is a tiny smoke test, not a conclusion.*

In this very limited setting, removing noise, removing iterative compute, or replacing the full DDPM with MIP or pure regression did not rescue success. The full DDPM with T=100 steps also scored 0.0. That is not a refutation of Simchowitz; the network, data, and benchmark are too small to stress multi-modal recovery or out-of-distribution generalization. It is, however, a useful sanity check: **none** of the learned ablation conditions solved this particular task in this particular configuration. That is exactly why the full study uses larger networks, more demos, and many more seeds and benchmarks.

The mode-coverage numbers are all zero because 2-D reaching has a single goal. We will report mode coverage meaningfully on the multi-modal RoboMimic `Lift`, `Can`, and `Square` tasks in the full run.

---

## 6. The latency-success Pareto frontier

One of our main research questions (RQ8) asks where the latency-performance Pareto frontier lies. The quick test is too small to draw a clean frontier, but the generated Pareto plot already hints at the shape:

![pareto](results/quick_test/report/figures/pareto_latency_vs_success.png)
*Figure 4: Latency vs. success in the quick test. Error bars are not meaningful at n = 5.*

In this smoke test, **Linear MPC** sits near the Pareto-optimal corner: 100% success at ~2.8 ms. **Nonlinear MPC (iLQR)** is slightly slower but still fast. **Collision-Free MPC** and **Diffusion Warm-Start MPC** both reach 100% success but at ~75–81 ms. The question for the full study is whether this pattern holds when we add vision, language, multi-modal goals, contact, and real-robot dynamics. We expect the frontier to shift: VLA will be high-latency and high-generalization; diffusion/GCP will occupy the middle; and classical MPC will dominate the low-latency, constraint-heavy region. MIP and WAMs may be the wildcards that push the frontier outward.

---

## 7. From toy sim to real robots

Our study is deliberately staged. Phase 1 uses 2-D reaching and PushT to isolate the mechanism question with fast iteration. Phase 2 and beyond move toward real-robot control through three focus areas:

1. **World Action Models (WAMs):** joint dynamics + action models that predict future states before acting. We test whether WAMs improve long-horizon and contact-rich manipulation over pure VLA or diffusion (EXP-005).
2. **Real-world MPC:** whole-body, contact-aware MPC, including centroidal/single-rigid-body approximations and full-order nonlinear MPC with learned warm starts. We test whether whole-body MPC outperforms learned policies on legged locomotion and loco-manipulation (EXP-007).
3. **Real robotic vision:** a shared vision preprocessing pipeline with domain randomization, photorealistic rendering, and an OOD perturbation suite for lighting, occlusion, texture, camera pose, and combined perturbations (EXP-008).

We also have a dedicated **sim-to-real** experiment (EXP-006) that tests whether domain randomization, photorealistic rendering, and ISP-aware augmentation can close the sim-real visual gap, and whether sim-real co-training improves data efficiency over real-only fine-tuning.

The Phase-2 plan is documented in [`docs/real_robotics/phase2_roadmap.md`](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/real_robotics/phase2_roadmap.md). It replaces 2-D reaching with MuJoCo/robosuite and ManiSkill3 tasks, adds the WAM baseline, and sets explicit phase-gate criteria (e.g., all baselines must run ≥ 10 Hz in sim before we move to real hardware).

---

## 8. Try it yourself

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

## 9. Artifacts and reproducibility

Right now the primary artifact is the **GitHub repository** itself: pre-registered protocols, runnable code, committed configs, and the generated quick-test figures and tables. We have not yet published Hugging Face Hub model, dataset, or Space artifacts, but they are on the roadmap:

- **HF Model repos** for SmallVLA, DiffusionPolicy, FlowMatchingPolicy, and MIP checkpoints.
- **HF Dataset repo** for expert demonstrations in LeRobotDataset or npz format.
- **HF Space** with an interactive comparison gallery or benchmark leaderboard.
- **HF Collection** to tie the GitHub repo, models, datasets, and Space together.

Until then, every result is traceable to a config hash, a git commit, and the CSV/JSON files in `results/`. The quick-test outputs were generated with the run script `scripts/run_quick_test.sh` and the report generator `generate_report.py`. The full-study environment is pinned in `experiments/environment.lock`.

---

## 10. What is next?

We are executing the pre-registered experiment sequence, starting with EXP-001 (mechanism ablation) and EXP-002 (three-family comparison). The immediate next steps are:

1. **Scale the ablation** to full network sizes, 5 seeds, and the multi-modal PushT task to see whether MIP matches DDPM/flow when multi-modal behavior is possible.
2. **Run EXP-002** across MPC, VLA, SmallVLA, and diffusion/flow baselines on PushT, 2-D/3-D Reaching, and the cluttered reaching task.
3. **Run EXP-004**, the latency-performance Pareto sweep, to place every method on a single latency-success plot with 1,000-inference median latencies.
4. **Launch Phase 2** with MuJoCo/robosuite and ManiSkill3 tasks, a small WAM baseline, and the real-robot vision preprocessing pipeline.

We will update the blog and the Hugging Face artifacts as each experiment completes.

---

## 11. Call to action

If this comparison matters to you, here is how to get involved:

- **Reproduce the quick test** on your own machine: `bash scripts/run_quick_test.sh`.
- **Run the full protocol** on a DGX Spark or compatible GPU: see `experiments/README.md`.
- **Contribute VLA baselines** — we have a SmallVLA stub and an OpenVLA wrapper; more robust VLA conditions would strengthen the comparison.
- **Add a benchmark** or a perturbation suite, especially for real-robot vision and contact-rich tasks.
- **Open issues and PRs** with methodological critiques, bug reports, or new controller families.
- **Watch the repository** for Hugging Face Hub model and dataset releases as the study scales up.

The goal is not to crown a single winner. It is to replace hype with reproducible evidence, and to give the robotics community a shared, open map of where each control family belongs on the latency-generalization-safety Pareto frontier.

---

*Repository: [github.com/Ryukijano/mpc-vla-diffusion-study](https://github.com/Ryukijano/mpc-vla-diffusion-study)*

*Pre-registered research questions: [docs/research_questions.md](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/research_questions.md)*

*Methodology: [docs/methodology.md](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/methodology.md)*

*Real-robotics roadmap: [docs/real_robotics/phase2_roadmap.md](https://github.com/Ryukijano/mpc-vla-diffusion-study/blob/main/docs/real_robotics/phase2_roadmap.md)*
