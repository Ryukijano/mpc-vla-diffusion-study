# Live Experiment Status Tracker

**Study:** MPC vs VLA vs Diffusion — An Open-Source Study of Robot Control Families  
**Repository:** `Ryukijano/mpc-vla-diffusion-study`  
**Compute Platform:** NVIDIA DGX Spark (`spark-1240`), GB10 Grace Blackwell Superchip, CUDA 12.8  
**Status Date:** 2026-08-14  
**Author:** Devin continuation session

---

## Horizon 1 Status (EXP-001..004)

### EXP-001 — GCP Mechanism Ablation

| Field | Value |
|-------|-------|
| **Run command** | `conda run -n mpc_vla python run_ablation.py --benchmark all --seeds 0 1 2 42 123 --episodes 25 --epochs 30 --num-demos 20 --output-dir results/exp001` |
| **Status** | **Completed.** 5 seeds, 25 episodes, 30 training epochs, 20 demonstrations, small networks on `reaching` and `pusht`. |
| **Outputs** | `results/exp001/ablation_aggregated.csv`, `ablation_comparison.csv`, `ablation_results.json`, `figures/*.png` |
| **Key finding** | On `reaching`, pure Regression (RCP) reaches 32.8% success, MIP 21.6%, and full DDPM (T=100) only 2.4%. On `pusht` all learned variants remain near 0%. |
| **Blockers** | None — ready for blog integration. |

### EXP-002 — Three-Family Comparison

| Field | Value |
|-------|-------|
| **Run command** | `conda run -n mpc_vla python run_experiments.py --benchmark reaching,reaching_cluttered --controllers mpc,diffusion,vla --seeds 0 1 2 42 123 --episodes 10 --net-size small --output-dir results/exp002` |
| **Status** | **Completed.** 5 seeds, 10 episodes per seed, `small` networks on `reaching` and `reaching_cluttered`. |
| **Outputs** | `results/exp002/aggregated_comparison.csv`, `master_comparison.csv`, `figures/*.png`, `metrics/full_results.json` |
| **Key finding** | All MPC variants (Linear, Nonlinear, Collision-Free, Diffusion Warm-Start) solve both benchmarks with 100% success. Learned policies (MIP, Regression, Iterative, Flow, DDPM) reach 0–34% success, with Iterative Regression the best learned baseline. VLA skipped because the run used state-only demonstrations. |
| **Blockers** | VLA not exercised on state-only benchmarks; needs image+language PushT or reaching run for a true three-family comparison. |

### EXP-003 — OOD Robustness

| Field | Value |
|-------|-------|
| **Script** | `scripts/run_ood_evaluation.py` (created and verified) |
| **Quick test command** | `conda run -n mpc_vla python scripts/run_ood_evaluation.py --benchmarks reaching --seeds 0 --episodes 5 --output-dir results/exp003_quick` |
| **Status** | **Quick test completed.** Supports L0–L4 perturbations and the 8 EXP-002 controllers. |
| **Outputs** | `results/exp003_quick/ood_aggregated.csv`, `ood_results.json`, `figures/*.png` |
| **Key finding** | MPC-based controllers retain non-zero success under quick OOD perturbations; learned baselines scored 0.00 on the fast run. A full 5-seed, 25-episode OOD sweep is pending. |

### EXP-004 — Latency-Performance Pareto

| Field | Value |
|-------|-------|
| **CPU low-latency sweep** | `conda run -n mpc_vla python scripts/run_pareto_cpu_low_latency.py --seeds 0 1 --episodes 10 --output-dir results/exp004_cpu_low_latency` |
| **Status** | **CPU low-latency sweep completed.** 34 conditions on `reaching` and `pusht`, 10 warmup + 100 timed inferences per condition on CPU. |
| **Outputs** | `results/exp004_cpu_low_latency/pareto_data.csv`, `latency_table.csv`, `pareto_frontier.png`, `metrics_summary.json`, `VERIFICATION.md` |
| **Key finding** | `Linear MPC (H=5)` is the Pareto-optimal point on both benchmarks (100% success at ~0.37 ms on reaching, ~0.42 ms on PushT). Regression and MIP come close but are Pareto-dominated. |
| **GPU image-based sweep** | `scripts/run_pareto_sweep.py` completed a fast `reaching` run: 1 seed, 2 episodes, 5 warmup + 100 timed calls. Outputs in `results/exp004_gpu_quick/`. Full 5-seed, 50-episode sweep pending more compute budget. |

---

## HF Hub Artifacts Status

- **Packaging complete locally; upload still pending.**
- Models: `dist/hf_models/{small_vla,ddpm,flow_matching,mip}/`
- Datasets: `dist/hf_datasets/mpc_expert_demos/`
- Model / dataset cards: `results/hf_artifacts/`
- **No Hub repos have been created or pushed.**

## Blog Status

- Draft: `docs/blogging/hf_blog_draft.md`
- **Sections 5 (EXP-002), 6 (EXP-001), and 7 (EXP-004 CPU Pareto) updated with real Horizon 1 numbers.**
- **Publication decision:** publish only after at least one HF Hub artifact is live and the GPU Pareto sweep is complete (per `docs/blogging/publish_decision_analysis.md`).

## Next Actions

1. Run canonical GPU Pareto sweep (`scripts/run_pareto_sweep.py`) to obtain 1,000-inference medians with DDPM/Flow/SmallVLA on `reaching` and `pusht`.
2. Run a full 5-seed OOD sweep with `scripts/run_ood_evaluation.py` if needed for the blog.
3. Create and push at least one HF Hub model and dataset repo.
4. Update `docs/blogging/hf_blog_draft.md` TL;DR and limitations once the GPU Pareto is available.
5. Run final verification, commit, and push.
