# Claim Verification Audit — MPC vs VLA vs Diffusion Study

**Generated:** 2026-08-12  
**Scope:** Review claims that could appear in the Hugging Face blog post against the current repository state.  
**Repository:** `mpc-vla-diffusion-study` (`/home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study/`)

This audit is organized into the five requested categories. For each claim we list the proposed wording, where it appears or could appear, the evidence in the repo, and a verdict. Verdicts are:

- **Supported** — the current repo clearly backs the claim.
- **Overstated** — the claim is present in study documents but the evidence is weaker than the wording implies (e.g., plan vs. result, toy benchmark vs. general finding).
- **Unsupported** — the claim is not backed by the current repo and should not be blogged without additional work.

A final "blog-safe claim list" summarizes claims that can be stated confidently right now.

---

## 1. MPC vs VLA vs Diffusion family comparison

### Claim 1.1 — The study is a systematic comparison of classical MPC, VLA, and diffusion/flow-based generative control policies (GCPs), plus hybrids and WAMs.

- **Claim text:** "We compare four controller families through a single harness: Classical MPC, VLA, Diffusion/Flow policies (GCPs), and Hybrids; World Action Models (WAMs) are tracked for Phase 2."
- **Evidence:**
  - `README.md` lines 19-23 and 1-2 describe the three-family comparison.
  - `docs/comparison_plan.md` lines 1-6 and 19-23 list Classical MPC, VLA, Diffusion MPC/GCP, and hybrid approaches.
  - `docs/research_questions.md` lines 5-6, 49-54, 55-58 add WAMs (RQ9) and hybrid approaches (RQ7).
- **Verdict:** **Supported** as study framing and design.

### Claim 1.2 — The comparison matrix covers 14 design dimensions (action representation, multi-modality, constraints, latency, safety, etc.).

- **Claim text:** "The full comparison matrix in `docs/comparison_plan.md` scores the three families on action representation, multi-modality, observation conditioning, dynamics, constraints, generalization, language, latency, data efficiency, safety, OOD robustness, real-time capability, and training cost."
- **Evidence:**
  - `docs/comparison_plan.md` lines 86-100 contain the 14-row comparison table.
  - `docs/real_robotics/phase2_roadmap.md` lines 58-74 extends this table with a fourth WAM column.
- **Verdict:** **Supported** as a documented design matrix.

### Claim 1.3 — Classical MPC is fast and safe but model-dependent; VLA is general and language-native but high-latency; diffusion/GCP is flexible but expensive and constraint-agnostic.

- **Claim text:** "Classical MPC is fast and safe but model-dependent; VLA is general and language-native but high-latency; diffusion/GCP is flexible but expensive and constraint-agnostic; WAMs and hybrids may combine the best of both."
- **Evidence:**
  - `docs/comparison_plan.md` lines 86-100, especially rows for **Constraint handling**, **Language conditioning**, **Inference latency**, **Safety guarantees**, and **Generalization**.
  - `docs/blogging/hf_blog_draft.md` line 36 makes the same high-level summary.
- **Verdict:** **Supported as framing and background**, but **overstated as a finding** — the comparison table is largely an expectation table, not an empirical result. The quick test only ran a tiny 2-D reaching task and did not include VLA or the full benchmark suite.

### Claim 1.4 — The study's main comparison has already shown that classical MPC dominates on a 2-D reaching task.

- **Claim text:** "In our smoke test, every MPC variant solved the task with 100% success, while the standalone MIP only reached 40% success."
- **Evidence:**
  - `results/quick_test/tables/master_comparison.csv` lines 2-6 and `results/quick_test/metrics/full_results.json` lines 31-104 show 100% success for Linear/Nonlinear/Collision-Free/Diffusion Warm-Start MPC and 0.4 for MIP (standalone).
  - `docs/blogging/hf_blog_draft.md` lines 66-79 repeats this table and interpretation.
- **Verdict:** **Overstated.** The numbers are real quick-test outputs, but the quick test used only one seed, five episodes, tiny networks (`hidden_dim=16`, 10 demos, 10 training epochs), and a single 2-D reaching benchmark. It also excluded VLA entirely. This is a sanity check, not the "main comparison."
- **What needs to be done:** Run the full `EXP-002` protocol (100 episodes × 5 seeds × 8 conditions × 3 benchmarks, 12,000 episodes total) before claiming family-level dominance.

### Claim 1.5 — The study will evaluate all three families on MetaWorld, RoboMimic, CALVIN, LIBERO, DMControl, and a custom cluttered-reaching benchmark.

- **Claim text:** "We select benchmarks that span each family's strength zone: MetaWorld, RoboMimic, CALVIN, LIBERO, DMControl, and real-robot tasks."
- **Evidence:**
  - `docs/comparison_plan.md` lines 119-125 list these benchmarks.
  - `docs/methodology.md` lines 10-16 repeats them.
  - `docs/blogging/hf_blog_draft.md` line 46 also names them.
- **Verdict:** **Supported as the planned benchmark selection.** However, the current repo only implements `reaching` and `pusht` (see `benchmarks/` and `configs/benchmarks/`). The other benchmarks are not integrated yet.

### Claim 1.6 — VLA has native language conditioning and strong generalization via VLM pretraining; diffusion/GCP requires VLM conditioning or text embedding for language.

- **Claim text:** "VLA is language-native because of its VLM backbone; diffusion/GCP can be language-conditioned only via a VLM or text embedding."
- **Evidence:**
  - `docs/comparison_plan.md` lines 94 and 97 state these properties in the comparison table.
  - `docs/research_questions.md` RQ6 (lines 36-41) and H6 state VLA's expected dominance on language-conditioned and zero-shot tasks.
- **Verdict:** **Supported as a design/background claim**, but no VLA has been trained or evaluated in the repo yet. The quick test sets `controller_families = ["mpc", "diffusion"]` (see `run_experiments.py` lines 893-894 and `full_results.json` lines 6-9), and the OpenVLA import in `run_experiments.py` is currently broken (see Category 5).

### Claim 1.7 — The three families occupy distinct regions of the latency–performance Pareto frontier.

- **Claim text:** "Classical MPC occupies the low-latency/moderate-performance region, VLA the high-latency/high-generalization region, and diffusion/GCP the middle, with MIP on the frontier."
- **Evidence:**
  - `docs/research_questions.md` RQ8 / H8 (lines 49-53) makes this prediction.
  - `experiments/EXP-004-latency-pareto/protocol.md` lines 14-29 pre-registered the Pareto analysis.
- **Verdict:** **Unsupported as an empirical result.** No EXP-004 results exist. The quick-test "Pareto" plot (`results/quick_test/report/figures/pareto_latency_vs_success.png`) has only five points on one toy benchmark and is explicitly labeled "error bars are not meaningful at n = 5" in `docs/blogging/hf_blog_draft.md` line 124.
- **What needs to be done:** Run the full 21-condition latency-Pareto sweep on the target benchmarks (100 episodes × 5 seeds × 21 conditions × 2 benchmarks, plus 1,000 latency calls each).

### Claim 1.8 — Diffusion/GCP policies claim multi-modal action distribution capture, but this is disputed by Simchowitz et al.

- **Claim text:** "Diffusion policies are often claimed to capture multi-modal action distributions; Simchowitz et al. argue the advantage actually comes from iterative compute + noise, not distribution fitting."
- **Evidence:**
  - `docs/comparison_plan.md` lines 66-67 and 81-82.
  - `docs/research_questions.md` RQ2 / H2a-H2b (lines 10-16) and RQ3 / H3a-H3b (lines 18-23).
- **Verdict:** **Supported** as the stated scientific debate and the study's central question. It is not yet resolved by the study's own data.

---

## 2. Simchowitz / *Much Ado About Noising* mechanism ablation

### Claim 2.1 — The study is motivated by Simchowitz et al.'s finding that GCP success comes from iterative compute + noise, not multi-modal distribution fitting.

- **Claim text:** "This study is motivated by Max Simchowitz's talk 'Do we need diffusion in robotics?' and the paper *Much Ado About Noising* (arXiv:2512.01809). Simchowitz et al. find that GCPs' success is not due to multi-modal distribution fitting but to supervised iterative compute + stochasticity injection."
- **Evidence:**
  - `README.md` lines 5-18.
  - `docs/comparison_plan.md` lines 81-82.
  - `docs/research_questions.md` lines 18-23 (H3a vs H3b).
- **Verdict:** **Supported** as a motivation/background claim.

### Claim 2.2 — A minimal iterative policy (MIP, 2-step regression + noise) matches full flow-based GCPs.

- **Claim text:** "A MIP (2-step regression with noise) matches full flow GCPs, as claimed by Simchowitz et al."
- **Evidence:**
  - `docs/comparison_plan.md` lines 81-82 and `docs/research_questions.md` H3b (lines 21-22) state this as the hypothesis to test.
  - `configs/diffusion/mip.yaml` lines 1-23 configures MIP as `num_iterations=2, noise_std=0.1`.
  - `results/quick_test/ablation/ablation_results.json` lines 24-87 and `ablation_aggregated.csv` show quick-test success of 0.0 for Full DDPM, 0.2 for MIP, and 0.2 for Pure Regression on reaching — the opposite of "matches" at this scale.
- **Verdict:** **Unsupported as a study result.** The quick test is not designed to validate or refute Simchowitz; the full EXP-001 protocol must be run.
- **What needs to be done:** Run `EXP-001` (5,000 episodes, 5 conditions, 2 benchmarks) and check whether `success(MIP) ≥ success(Full DDPM) - 3 pp` on both PushT and 2-D Reaching, as pre-registered in `experiments/EXP-001-mechanism-ablation/protocol.md` lines 31-37.

### Claim 2.3 — The study has pre-registered a 5,000-episode mechanism ablation (EXP-001) to test the hypothesis.

- **Claim text:** "We pre-registered an 80,000-episode protocol with component-level ablations. EXP-001 alone tests the mechanism of GCPs with 5,000 episodes."
- **Evidence:**
  - `experiments/README.md` lines 17-19 list EXP-001 and its 5,000-episode sample size.
  - `experiments/EXP-001-mechanism-ablation/protocol.md` lines 1-4, 15-23, and 154-156 document the hypothesis, conditions, and sample-size calculation.
- **Verdict:** **Supported** as a pre-registration.

### Claim 2.4 — The quick smoke-test ablation found all five GCP variants at or below 20% success on 2-D reaching.

- **Claim text:** "In the quick GCP ablation, Full DDPM (T=100), DDPM no-noise (T=100), DDPM single-step (T=1), MIP (2-iter + noise), and Pure Regression all failed to solve the toy reaching task, with success rates between 0.0 and 0.2."
- **Evidence:**
  - `results/quick_test/ablation/ablation_results.json` lines 24-87.
  - `results/quick_test/ablation/ablation_aggregated.csv` lines 1-6.
  - `docs/blogging/hf_blog_draft.md` lines 100-113 explicitly cautions that this is not a refutation of Simchowitz.
- **Verdict:** **Supported as a raw smoke-test output**, but only a sanity check. Must be accompanied by the caveat that the data/network/benchmark are too small to test multi-modal recovery or OOD generalization.

### Claim 2.5 — Removing iterative compute is predicted to cause the largest performance drop; removing noise a smaller drop.

- **Claim text:** "Under the Simchowitz view, the largest single drop should come from removing iterative compute (single-step DDPM), while removing noise has a smaller but non-zero effect."
- **Evidence:**
  - `experiments/EXP-001-mechanism-ablation/protocol.md` lines 33-37 pre-registers this component ordering prediction.
- **Verdict:** **Supported as a pre-registered prediction**, not an observed result.

### Claim 2.6 — MIP is expected to lie on the Pareto frontier because it captures most of the diffusion benefit at a fraction of the latency.

- **Claim text:** "MIP (2-step + noise) is predicted to be on the Pareto frontier on both PushT and 2-D Reaching."
- **Evidence:**
  - `experiments/EXP-004-latency-pareto/protocol.md` lines 26-29, 38-40.
  - `docs/research_questions.md` H8 (lines 49-53).
- **Verdict:** **Unsupported as a result.** The Pareto sweep has not been run.
- **What needs to be done:** Run `EXP-004` (21 conditions, 1,000 timed inferences each, 2 benchmarks) and compute Pareto dominance.

### Claim 2.7 — Mode coverage in the quick ablation is zero because 2-D reaching is uni-modal; meaningful mode coverage will be reported on RoboMimic multi-modal tasks.

- **Claim text:** "Mode-coverage numbers are all zero in the quick-test ablation because 2-D reaching has a single goal; we will report mode coverage on multi-modal RoboMimic Lift/Can/Square in the full run."
- **Evidence:**
  - `docs/blogging/hf_blog_draft.md` lines 114-115.
  - `docs/methodology.md` lines 26-34 lists RoboMimic multi-modal tasks.
  - `experiments/EXP-001-mechanism-ablation/protocol.md` lines 38-39 and 103-104 define mode coverage and the prediction.
- **Verdict:** **Supported** as an internally consistent explanation and plan.

---

## 3. DGX Spark results (from quick smoke test)

### Claim 3.1 — The quick smoke test is designed to run in under 2 minutes on an NVIDIA DGX Spark (GB10).

- **Claim text:** "The quick smoke test runs one seed, five episodes, tiny networks, and the 2-D reaching benchmark; it is designed to take < 2 minutes on GB10."
- **Evidence:**
  - `scripts/run_quick_test.sh` lines 3-7.
  - `run_experiments.py` lines 12-13 and 1066-1068 (the `--quick` argument).
  - `configs/system_config.yaml` lines 10-22 targets the DGX Spark / GB10.
- **Verdict:** **Supported as an intended design.** Whether it actually ran on DGX Spark is not independently proven in the repo (see 3.2).

### Claim 3.2 — The quick smoke test results were generated on an NVIDIA DGX Spark (GB10).

- **Claim text:** "The quick test was run on an NVIDIA DGX Spark with GB10 GPU, 128 GB unified memory, and CUDA 12.8."
- **Evidence:**
  - `experiments/environment.lock` lines 21-25 and 32-37 record DGX Spark specs and a git commit `2a656cf`.
  - However, `results/quick_test/` contains no `env_info.json`, `nvidia-smi` log, or commit hash proving the runtime environment.
  - The current repository HEAD is `0867a71`, while `environment.lock` still records `2a656cf` (the initial commit).
- **Verdict:** **Unsupported / needs verification.** The repo *targets* DGX Spark, but the quick-test outputs do not embed the required runtime provenance, and the lock file is out of sync with the current commit.
- **What needs to be done:** Rerun the quick test on DGX Spark, log `env_info.json` (git commit, `nvidia-smi`, package versions, hardware), commit the outputs, and refresh `environment.lock` to the new commit.

### Claim 3.3 — The quick test evaluated MPC and diffusion families on 2-D reaching with one seed and five episodes.

- **Claim text:** "The quick test used `reaching` only, with `controller_families = ["mpc", "diffusion"]`, 1 seed, 5 episodes, tiny networks."
- **Evidence:**
  - `run_experiments.py` lines 885-894 (quick-mode overrides).
  - `results/quick_test/metrics/full_results.json` lines 2-26 (config).
  - `results/quick_test/tables/master_comparison.csv` lines 1-6.
- **Verdict:** **Supported** by the committed results and the quick-mode code path.

### Claim 3.4 — Linear MPC: 100% success / 2.84 ms; Nonlinear MPC: 100% / 27.79 ms; Collision-Free MPC: 100% / 80.93 ms; Diffusion Warm-Start MPC: 100% / 74.19 ms; MIP: 40% / 0.0068 ms.

- **Claim text:** "In the quick test, all four MPC/hybrid variants scored 1.0 success, while standalone MIP scored 0.4; latencies ranged from 2.8 ms (Linear MPC) to 80.9 ms (Collision-Free MPC)."
- **Evidence:**
  - `results/quick_test/tables/master_comparison.csv` lines 2-6.
  - `results/quick_test/report/master_comparison_table.csv` lines 2-6.
  - `results/quick_test/metrics/full_results.json` lines 31-104.
- **Verdict:** **Supported as reported quick-test numbers.** These should be presented with the caveats "one seed, five episodes, tiny networks, toy benchmark."

### Claim 3.5 — The quick test proves that MPC outperforms learned policies and that diffusion/GCP is unnecessary.

- **Claim text:** "The quick test shows MPC outperforms diffusion/MIP, so we may not need diffusion for robot control."
- **Evidence:**
  - `docs/blogging/hf_blog_draft.md` lines 79-83 and 113 explicitly reject this interpretation.
  - `results/quick_test` is a tiny sanity check, not the full study.
- **Verdict:** **Unsupported / dangerous overstatement.** The draft already contains the correct disclaimer.
- **What needs to be done:** Do not blog this; wait for `EXP-001`–`EXP-004` and the real-physics Phase-2 results.

### Claim 3.6 — The quick-test Pareto plot shows Linear MPC near the Pareto-optimal corner and Collision-Free / Diffusion Warm-Start MPC at ~75–81 ms.

- **Claim text:** "The quick Pareto plot already hints that Linear MPC is near the Pareto-optimal corner, while Collision-Free MPC and Diffusion Warm-Start MPC reach 100% success at ~75–81 ms."
- **Evidence:**
  - `results/quick_test/report/figures/pareto_latency_vs_success.png` exists.
  - `docs/blogging/hf_blog_draft.md` lines 121-127 describes it.
- **Verdict:** **Overstated.** The plot exists, but the draft correctly notes that error bars are not meaningful at n = 5 and that the question is whether the pattern holds when adding vision, language, multi-modal goals, contact, and real-robot dynamics.

### Claim 3.7 — The full experiment suite (EXP-001–004) is estimated to take ~16 h on DGX Spark, with the quick suite at ~45 min.

- **Claim text:** "The full four-experiment suite is estimated at ~16 h wall-clock on DGX Spark; the quick suite at ~45 min."
- **Evidence:**
  - `experiments/README.md` lines 119-128 provides these wall-clock estimates.
- **Verdict:** **Supported as an estimate**, not a measured result.

---

## 4. Real-robotics roadmap / WAMs / sim-to-real

### Claim 4.1 — Phase 2 will move from 2-D reaching/PushT to real-physics simulators and add WAM as a fourth family.

- **Claim text:** "Phase 2 replaces the toy 2-D reaching and PushT tasks with real-physics, vision-language, contact-rich benchmarks and adds World Action Models as a fourth family."
- **Evidence:**
  - `docs/real_robotics/README.md` lines 12-34 and 75-81.
  - `docs/real_robotics/phase2_roadmap.md` lines 12-21, 28-34, and 58-75 (WAM column in the updated comparison matrix).
- **Verdict:** **Supported** as a documented roadmap.

### Claim 4.2 — World Action Models (WAMs) explicitly predict future states to inform action generation.

- **Claim text:** "A WAM is an embodied predictive-action model that makes a forecast of the future available to action generation, sitting between classical MPC and end-to-end VLA/diffusion."
- **Evidence:**
  - `docs/real_robotics/world_action_models_briefing.md` lines 11-19.
  - `docs/blogging/hf_blog_draft.md` lines 134-135.
- **Verdict:** **Supported** as a research synthesis/definition.

### Claim 4.3 — Phase 2 has explicit phase-gate criteria.

- **Claim text:** "Phase 2 is gated by: (1) all baselines run ≥ 10 Hz in sim, (2) WAM matches or exceeds SmallVLA within 3 pp on a long-horizon task, and (3) the sim-to-real visual gap is ≤ 15 pp on the easiest task."
- **Evidence:**
  - `docs/real_robotics/README.md` lines 79-81.
  - `docs/real_robotics/phase2_roadmap.md` lines 119-123.
- **Verdict:** **Supported** as documented gating criteria.

### Claim 4.4 — The real-robotics roadmap, WAM baselines, and sim-to-real experiments are currently in planning only.

- **Claim text:** "The Phase-2 real-robotics work is pre-registered but not yet implemented or run."
- **Evidence:**
  - `docs/real_robotics/world_action_models_briefing.md` §5 "Gaps in Current Study" (lines 258-269) states there are no WAM baselines, no joint video-action modeling, no world-model rollouts, and no cross-embodiment transfer.
  - `docs/real_robotics/phase2_roadmap.md` line 1 says **Status: Phase-2 planning**.
  - `git ls-files` shows no `experiments/EXP-005`–`EXP-008/outputs/` directories and no `docs/real_robotics/phase2_report.md`.
- **Verdict:** **Supported** — the roadmap is documented and its unimplemented status is explicit.

### Claim 4.5 — The WAM briefing lists open-source models and Hugging Face / GitHub links (π0, Motus, UnifoLM-WMA-0, DreamZero, Cosmos 3, Octo, etc.).

- **Claim text:** "Open-source WAM/VLA models such as π0, π0.5, Motus, UnifoLM-WMA-0, DreamZero, Cosmos 3, Octo, and VIMA are available with Hugging Face or GitHub links, and some are feasible on DGX Spark."
- **Evidence:**
  - `docs/real_robotics/world_action_models_briefing.md` §4 (lines 228-255) contains a table with model IDs, Hugging Face repo IDs, GitHub URLs, and hardware notes.
  - `docs/real_robotics/tools_and_data_inventory.md` lines 45-61 also lists OpenVLA, Octo, π0, Motus, DreamZero, Cosmos 3 with DGX Spark feasibility notes.
- **Verdict:** **Supported as an inventory.** Caveat: actual download/run feasibility has not been verified in the current environment, and some models require multi-GPU or specific authentication.

### Claim 4.6 — Multi-view fusion and ISP-aware augmentation significantly improve real-robot OOD robustness.

- **Claim text:** "Multi-view fusion + ISP-aware training significantly improves OOD robustness under lighting, occlusion, and distractors, e.g., hammer manipulation improves from 13% single-view to 75% multi-view."
- **Evidence:**
  - `docs/real_robotics/real_robotic_vision_briefing.md` §1.3 (line 30) cites 75% vs 13% on hammer manipulation.
  - `docs/research_questions.md` H12 (line 71) states the same hypothesis.
- **Verdict:** **Supported as a cited literature claim / hypothesis, but not as a study result.** The 75% vs 13% number is from an external paper (Look Closer / 4Diff / 3D-MVP, referenced in the briefing), not from this study.
- **What needs to be done:** If blogging this, attribute it to the cited source, not to the study's own experiments. Run `EXP-008` to produce own numbers.

### Claim 4.7 — The sim-to-real strategy includes domain randomization, photorealistic rendering, ISP-aware augmentation, and sim-real co-training.

- **Claim text:** "Our sim-to-real strategy uses domain randomization, photorealistic rendering, and ISP-aware augmentation, and we will test whether sim-real co-training improves data efficiency over real-only fine-tuning."
- **Evidence:**
  - `docs/real_robotics/real_robotic_vision_briefing.md` §2 (lines 41-72).
  - `docs/real_robotics/phase2_roadmap.md` lines 111 and 196-199.
  - `experiments/EXP-006-sim-to-real/protocol.md` (pre-registered protocol).
- **Verdict:** **Supported as the documented plan and literature synthesis.** No own sim-to-real experiments have been run.

### Claim 4.8 — Sim-real co-training gives +24% OpenVLA and +20% π0.5 over real-only fine-tuning.

- **Claim text:** "RL-Co (sim-real co-training) gives +24% OpenVLA and +20% π0.5 over real-only fine-tuning."
- **Evidence:**
  - `docs/real_robotics/real_robotic_vision_briefing.md` lines 70-73 and 102-105.
- **Verdict:** **Supported as a cited external result (RL-Co, arXiv:2602.12628)**, but not as a result of this study. Must be attributed to the original paper in any blog post.

---

## 5. Open-source reproducibility

### Claim 5.1 — The repository is public on GitHub at `github.com/Ryukijano/mpc-vla-diffusion-study`.

- **Claim text:** "The study repository is open-source at `github.com/Ryukijano/mpc-vla-diffusion-study`."
- **Evidence:**
  - `git remote -v` shows `https://github.com/Ryukijano/mpc-vla-diffusion-study.git`.
  - `docs/blogging/hf_blog_research.md` lines 91, 95 and `docs/blogging/hf_blog_draft.md` line 251 state the repo is public.
- **Verdict:** **Supported by repository metadata, with a small caveat.** The local clone points to that URL and the blog docs assert it is public, but the actual GitHub visibility (public vs. private) is not verifiable from the local filesystem. Confirm in GitHub settings.

### Claim 5.2 — All experiments are pre-registered before running.

- **Claim text:** "We pre-registered 12 research questions and 8 full experiment protocols before running the first evaluation episode."
- **Evidence:**
  - `experiments/README.md` lines 8-12 and the table of `EXP-001`–`EXP-008`.
  - `docs/research_questions.md` RQ1–RQ12 and H2–H12.
  - `experiments/EXP-001` through `EXP-008/protocol.md` files.
- **Verdict:** **Supported** as a pre-registration claim.

### Claim 5.3 — The environment is pinned in `experiments/environment.lock` with hardware, Python/PyTorch/CUDA versions, and a git commit.

- **Claim text:** "The full-study environment is pinned in `experiments/environment.lock`, including DGX Spark hardware, Python 3.11.15, PyTorch 2.12.0.dev+cu128, and a git commit hash."
- **Evidence:**
  - `experiments/environment.lock` lines 1-53 (header, hardware, commit, key packages, full pip freeze).
- **Verdict:** **Supported**, but with the caveat that the recorded commit `2a656cf` does not match the current repo HEAD `0867a71` and the lock has not been regenerated after recent commits.

### Claim 5.4 — Random seeds, evaluation seed files, data manifests, and committed config snapshots are already in place.

- **Claim text:** "Reproducibility is ensured by fixed seeds, committed evaluation seed files in `data/`, data manifests with hashes, and results logged with config hash and git commit."
- **Evidence:**
  - `docs/methodology.md` lines 127-136 lists these items as a checklist (many boxes unchecked in the source).
  - `experiments/README.md` lines 51-58 says evaluation seed files should be in `data/`, but the `data/` directory is currently empty (`ls -la data/` shows only `.` and `..`).
  - `results/quick_test/` contains no `env_info.json`, no config hash, and no per-condition commit hashes.
- **Verdict:** **Unsupported.** The reproducibility checklist is documented but not completed.
- **What needs to be done:** Generate and commit `data/eval_seeds_exp*.json`, create data manifests with hashes, and ensure each run writes `env_info.json` (commit, config hash, package versions, hardware) alongside its outputs.

### Claim 5.5 — One-command setup and smoke-test scripts are provided.

- **Claim text:** "You can clone the repo and run `bash scripts/setup_env.sh` and `bash scripts/run_quick_test.sh` to reproduce the environment and quick test."
- **Evidence:**
  - `scripts/setup_env.sh` lines 1-176.
  - `scripts/run_quick_test.sh` lines 1-144.
  - `docs/blogging/hf_blog_draft.md` lines 148-165 includes the commands.
- **Verdict:** **Supported as documentation and scripts.** Note that the current sandbox does not have the `mpc_vla` conda environment or PyTorch installed, so the scripts have not been verified in this environment.

### Claim 5.6 — The code is runnable end-to-end for all three families via `run_experiments.py` and `run_ablation.py`.

- **Claim text:** "The main runner handles the full four-phase pipeline — demonstration collection, training, evaluation, and table/figure generation — for all controller families."
- **Evidence:**
  - `run_experiments.py` exists and describes the four-phase pipeline (lines 1-20).
  - However, `run_experiments.py` line 107 tries to import `OpenVLAWrapper` from `vla_baselines`, while `vla_baselines/__init__.py` lines 17-27 only exports `OpenVLAInference`.
  - The quick mode explicitly sets `controller_families = ["mpc", "diffusion"]` and `bench_names = ["reaching"]` (lines 893-894), so VLA and Flow Matching are not exercised in the smoke test.
- **Verdict:** **Overstated / partially supported.** The MPC and diffusion paths are exercised, but the VLA import is currently broken and the quick test does not cover all families.
- **What needs to be done:** Fix the VLA import (`OpenVLAWrapper` → `OpenVLAInference` or add an alias), and run a full smoke test that at least imports/initializes all families.

### Claim 5.7 — Every result is traceable to a config hash, git commit, and `env_info.json`.

- **Claim text:** "Every result is traceable to a config hash, a git commit, and the CSV/JSON files in `results/`."
- **Evidence:**
  - `experiments/environment.lock` lines 36-37 and `experiments/EXP-001-mechanism-ablation/protocol.md` lines 175-178 describe the intended `env_info.json`.
  - `docs/blogging/hf_blog_draft.md` line 219 repeats this.
  - However, no `env_info.json` is present in `results/quick_test/` or elsewhere in `results/`.
- **Verdict:** **Unsupported.** The tracing mechanism is described but not demonstrated.
- **What needs to be done:** Update `run_experiments.py` and `run_ablation.py` to write `env_info.json` (or confirm they do and commit it), and rerun the quick test.

### Claim 5.8 — Hugging Face Hub artifacts (SmallVLA, DiffusionPolicy, dataset, Space, Collection) are ready to link.

- **Claim text:** "We will publish HF Model repos, a HF Dataset repo, a HF Space, and a HF Collection."
- **Evidence:**
  - `docs/blogging/hf_artifact_plan.md` line 7 says **Status: planning only — do not upload yet**.
  - The same file's artifact table (lines 30-36) lists all artifacts as **not yet created**.
- **Verdict:** **Unsupported.** Only planning documents exist.
- **What needs to be done:** Create the checkpoints/dataset/Space and upload to Hugging Face Hub, or reword the blog to say they are planned, not ready.

### Claim 5.9 — The top-level repository is open-source and licensed for reuse.

- **Claim text:** "The code is open-source and reusable."
- **Evidence:**
  - `mpc_baselines_repo/LICENSE` is present and `mpc_baselines_repo/README.md` lines 244-246 say it is MIT licensed.
  - A `find_file_by_name LICENSE*` search from the repo root only returned `mpc_baselines_repo/LICENSE`; there is no top-level `LICENSE` file.
- **Verdict:** **Overstated.** The MPC baselines subrepo has a clear MIT license, but the top-level study repository lacks a top-level `LICENSE` file.
- **What needs to be done:** Add a top-level `LICENSE` file (and ideally a `LICENSE` note in `README.md`) before publishing the blog.

---

## Blog-safe claim list

These are claims that are clearly backed by the current repository and can be stated in a blog post without additional experiments or corrections. They are deliberately conservative and avoid empirical conclusions.

1. **Study design and framing** — The study is an open, pre-registered comparison of classical MPC, VLA, and diffusion/flow-based generative control policies, with a Phase-2 plan to add World Action Models (WAMs). (`README.md`, `docs/comparison_plan.md`, `docs/research_questions.md`, `docs/real_robotics/README.md`)

2. **Pre-registered protocols** — Twelve research questions and eight pre-registered experiment protocols (`EXP-001`–`EXP-008`) document the planned comparisons, ablations, OOD tests, Pareto sweep, WAM baseline, sim-to-real, real-robot MPC, and real-vision experiments. (`experiments/README.md`, `experiments/EXP-*/protocol.md`, `docs/research_questions.md`)

3. **Codebase inventory** — The repo contains toy 2-D `reaching` and `pusht` benchmark environments, plus MPC baselines (Linear, Nonlinear, Collision-Free, Diffusion Warm-Start) and diffusion baselines (DDPM, Flow Matching, MIP, Iterative/Regression) implemented in Python. (`benchmarks/`, `mpc_baselines_repo/`, `diffusion_baselines/`)

4. **Quick smoke-test outputs** — A quick smoke test on 2-D reaching (1 seed, 5 episodes, tiny networks) has been run and the raw CSV/JSON/PNG outputs are committed in `results/quick_test/`. (`results/quick_test/tables/master_comparison.csv`, `results/quick_test/ablation/ablation_results.json`, `results/quick_test/report/figures/`)

5. **Quick-test numbers with caveats** — In the quick test, all four MPC/hybrid variants reported 100% success, standalone MIP reported 40%, and the full DDPM ablation reported 0% success. These are raw smoke-test numbers, not conclusions, and the blog must state the tiny sample size and toy-task limitations. (`results/quick_test/report/master_comparison_table.csv`, `results/quick_test/ablation/ablation_aggregated.csv`)

6. **Quick-test scope** — The quick smoke test currently evaluates only `mpc` and `diffusion` families on the `reaching` benchmark; it does not include VLA, flow matching, or the multi-modal/PushT benchmarks. (`run_experiments.py` quick mode, `results/quick_test/metrics/full_results.json` config)

7. **Pre-registered sample sizes** — `EXP-001` is pre-registered at 5,000 episodes (5 conditions × 2 benchmarks × 5 seeds × 100 episodes), `EXP-002` at 12,000 episodes (8 conditions × 3 benchmarks), and `EXP-004` at 21,000 latency-measurement points. (`experiments/README.md` experiment overview table, protocol files)

8. **Real-robotics roadmap** — Phase 2 is documented with target simulators (MuJoCo/robosuite, ManiSkill3, Isaac Sim, RoboDojo), a WAM baseline, sim-to-real experiments, and explicit phase-gate criteria (≥ 10 Hz, WAM ≤ 3 pp gap, sim-real ≤ 15 pp gap). (`docs/real_robotics/phase2_roadmap.md`, `docs/real_robotics/README.md`, `docs/real_robotics/world_action_models_briefing.md`)

9. **WAM definition and landscape** — The repo defines WAMs as predictive-action models and provides an inventory of open-source WAM/VLA models with Hugging Face/GitHub links and DGX Spark feasibility notes. (`docs/real_robotics/world_action_models_briefing.md` §1, §2, §4; `docs/real_robotics/tools_and_data_inventory.md`)

10. **Reproducibility infrastructure exists but is incomplete** — The repo provides an `experiments/environment.lock`, `scripts/setup_env.sh`, `scripts/run_quick_test.sh`, and `configs/` for methods and benchmarks. However, `data/` is empty, no `env_info.json` is present in the quick-test outputs, and the environment lock is pinned to an older commit. (`experiments/environment.lock`, `scripts/`, `configs/`, `data/`)

11. **Open-source status and licensing nuance** — The repository is associated with `github.com/Ryukijano/mpc-vla-diffusion-study` and the `mpc_baselines_repo` is MIT licensed, but the top-level repo does not include a top-level `LICENSE` file. (`git remote`, `mpc_baselines_repo/LICENSE`, `find_file_by_name` search)

12. **Honest status disclosure** — The position-paper abstract and key finding are explicitly marked `[TBD from experiments]`, and the `hf_artifact_plan.md` states `Status: planning only — do not upload yet`, confirming that the empirical comparison is not yet complete. (`docs/position_paper_outline.md` lines 14-15, `docs/blogging/hf_artifact_plan.md` line 7)

---

## Summary for the parent agent

- **Major strengths to blog about now:** the study design, pre-registration, code availability, the quick smoke-test as a sanity check, and the real-robotics roadmap. The draft blog (`docs/blogging/hf_blog_draft.md`) is already careful about most caveats.
- **Claims that need correction / qualification before blogging:**
  - Any implication that the quick test is the main result or that MPC has been shown to dominate.
  - Any claim that the quick test ran on DGX Spark without a committed `env_info.json`.
  - Any claim that all three families are already runnable end-to-end (VLA import bug).
  - Any claim that evaluation seeds, data manifests, or per-result env hashes are already committed (`data/` is empty).
  - Any claim that HF Hub artifacts are already ready (`hf_artifact_plan.md` says "planning only").
- **Missing items to create before a strong reproducibility/reproduced-results blog:**
  1. Run and commit `env_info.json` from the DGX Spark quick test.
  2. Generate and commit `data/eval_seeds_exp*.json` and data manifests.
  3. Refresh `experiments/environment.lock` to the current HEAD.
  4. Fix `run_experiments.py` VLA import (`OpenVLAWrapper` vs `OpenVLAInference`).
  5. Add a top-level `LICENSE` file.
  6. Run `EXP-001`–`EXP-004` before making family-comparison, Pareto, or Simchowitz-replication claims.
