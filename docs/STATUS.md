# Live Experiment Status Tracker

**Study:** MPC vs VLA vs Diffusion — An Open-Source Study of Robot Control Families  
**Repository:** `Ryukijano/mpc-vla-diffusion-study`  
**Compute Platform:** NVIDIA DGX Spark (`spark-1240`), GB10 Grace Blackwell Superchip, CUDA 12.8  
**Status Date:** 2026-08-14  
**Author:** Subagent E (status documentation)

---

## Executive Summary

- **Horizon 1 (EXP-001–004):** Partially started. Only quick smoke-test outputs exist. Full pre-registered 5-seed, 100-episode runs have not been completed.
- **Horizon 2 (EXP-005–010):** Pending. Protocol directories for EXP-005–008 exist; EXP-009 and EXP-010 protocols are not yet on disk.
- **Checkpoints:** Local baseline PushT checkpoints have been trained and verified, but the full per-experiment checkpoint trees are missing.
- **HF Hub Artifacts:** Not uploaded; local packaging is incomplete.
- **Blog:** Draft exists; publication is gated on full Horizon 1 results and Hub artifacts.

---

## 1. Horizon 1 Status (EXP-001..004)

### EXP-001 — GCP Mechanism Ablation

| Field | Value |
|-------|-------|
| **Full command (canonical)** | `conda run -n mpc_vla python run_ablation.py --benchmark all --seeds 0 1 2 42 123 --episodes 100 --output-dir experiments/EXP-001-mechanism-ablation/outputs` |
| **Quick smoke command** | `conda run -n mpc_vla python run_ablation.py --benchmark reaching --seeds 0 --episodes 5 --epochs 10 --num-demos 10 --output-dir results/quick_test/ablation` |
| **Expected outputs** | `experiments/EXP-001-mechanism-ablation/outputs/<condition>/seed_<s>/{config.yaml,checkpoint.pt,episodes.jsonl,metrics_summary.json}`; `experiments/EXP-001-mechanism-ablation/outputs/analysis/{comparison_table.csv,mechanism_ablation.pdf}` |
| **Current status** | **In-progress / partial.** A quick smoke test on 2-D Reaching (1 seed, 5 episodes, tiny networks) produced `results/quick_test/ablation/`. The full 5-condition × 2-benchmark × 5-seed × 100-episode run has not succeeded; `experiments/EXP-001-mechanism-ablation/outputs/` is empty. `results/horizon1_runbook.log` records a failed attempt because `run_ablation.py` rejected the `--benchmarks` plural flag (`run_ablation.py:1017–1020` expects `--benchmark` singular). |
| **Blockers** | • Missing `data/eval_seeds_exp001.json` (and 002/003/004); `data/` is currently empty.<br>• `scripts/run_horizon1.sh:32` uses `--benchmarks reaching,pusht` (plural), which `run_ablation.py` does not accept.<br>• Quick test was `reaching` only; PushT condition not validated in the ablation runner yet.<br>• `run_ablation.py` does not yet write `env_info.json` or config hashes at runtime. |

### EXP-002 — Three-Family Comparison

| Field | Value |
|-------|-------|
| **Full command (canonical)** | `conda run -n mpc_vla python run_experiments.py --benchmark all --controllers all --seeds 0 1 2 42 123 --episodes 100 --output-dir experiments/EXP-002-family-comparison/outputs` |
| **Quick smoke command** | `conda run -n mpc_vla python run_experiments.py --quick` |
| **Expected outputs** | `experiments/EXP-002-family-comparison/outputs/<condition>/<benchmark>/seed_<s>/{config.yaml,checkpoint.pt,episodes.jsonl,metrics_summary.json}`; `experiments/EXP-002-family-comparison/outputs/analysis/{comparison_table.csv,radar_*.png}` |
| **Current status** | **In-progress / partial.** Quick smoke test completed for `reaching` only (1 seed, 5 episodes, `mpc` + `diffusion` only; VLA excluded). Results are in `results/quick_test/`. Full 8-condition × 3-benchmark × 5-seed × 100-episode run not executed; `experiments/EXP-002-family-comparison/outputs/` is empty. |
| **Blockers** | • Missing `data/eval_seeds_exp002.json`.<br>• `scripts/run_horizon1.sh:43` uses `--benchmarks reaching,reaching_cluttered,pusht` (plural) but `run_experiments.py:1084` expects `--benchmark` singular.<br>• `scripts/run_horizon1.sh:47` passes `--horizon 16`, which `run_experiments.py` does not support.<br>• VLA was not included in the quick smoke; full three-family end-to-end on PushT has not been validated, although a local `results/checkpoints/small_vla_pusht.pt` exists. |

### EXP-003 — OOD Robustness

| Field | Value |
|-------|-------|
| **Full command (canonical)** | `conda run -n mpc_vla python run_experiments.py --benchmark pusht --controllers diffusion --seeds 0 1 2 42 123 --episodes 100 --output-dir experiments/EXP-003-ood-robustness/outputs` |
| **Expected outputs** | `experiments/EXP-003-ood-robustness/outputs/<condition>/seed_<s>/{config.yaml,checkpoint.pt,episodes.jsonl,metrics_summary.json}`; `experiments/EXP-003-ood-robustness/outputs/analysis/{anova_results.json,degradation_slopes.csv,manifold_adherence_curves.pdf}` |
| **Current status** | **Pending.** No outputs produced; `experiments/EXP-003-ood-robustness/outputs/` is empty. The protocol (`experiments/EXP-003-ood-robustness/protocol.md`) is pre-registered but the perturbation renderer has not been exercised in a full run. |
| **Blockers** | • Missing `data/eval_seeds_exp003.json`.<br>• Depends on EXP-001 trained checkpoints; only the baseline `results/checkpoints/` exist, not the full per-condition suite.<br>• `run_experiments.py` does not currently expose perturbation-level flags; may need a dedicated runner or protocol-specific harness. |

### EXP-004 — Latency-Performance Pareto

| Field | Value |
|-------|-------|
| **Full command (README canonical)** | `conda run -n mpc_vla python run_experiments.py --benchmark all --controllers all --seeds 0 1 2 42 123 --episodes 100 --output-dir experiments/EXP-004-latency-pareto/outputs` |
| **Dedicated Pareto sweep command** | `conda run -n mpc_vla python scripts/run_pareto_sweep.py --benchmark all --seeds 0 1 2 42 123 --episodes 100 --n-warmup 100 --n-timed 1000 --device cuda --output-dir experiments/EXP-004-latency-pareto/outputs` |
| **Expected outputs** | `experiments/EXP-004-latency-pareto/outputs/<condition>/<benchmark>/seed_<s>/{config.yaml,checkpoint.pt,episodes.jsonl,latency_timings.json,metrics_summary.json}`; `experiments/EXP-004-latency-pareto/outputs/analysis/{pareto_frontier.csv,pareto_pusht.pdf,pareto_reaching.pdf,step_saturation_curves.pdf}` |
| **Current status** | **Pending.** No full Pareto sweep outputs. Only quick smoke figures exist in `results/quick_test/figures/pareto_latency_vs_success.png` (generated by `run_experiments.py --quick`). `scripts/run_pareto_sweep.py:971–980` is available but has not been run. |
| **Blockers** | • Missing `data/eval_seeds_exp004.json`.<br>• Need to decide canonical runner: `run_experiments.py` (per `experiments/README.md:98–101`) or `scripts/run_pareto_sweep.py` (per `scripts/run_horizon1.sh:52–58`).<br>• 210,000 timed latency measurements require locked GPU clocks and dedicated DGX Spark time.<br>• EXP-001 DDPM T=100 checkpoint consistency gate must pass before Pareto step sweep. |

---

## 2. Horizon 2 Status (EXP-005..010)

| Experiment | Status | Notes |
|------------|--------|-------|
| EXP-005 — World Action Models | **Pending** | Protocol directory exists (`experiments/EXP-005-world-models/`). |
| EXP-006 — Sim-to-Real Transfer | **Pending** | Protocol directory exists (`experiments/EXP-006-sim-to-real/`). |
| EXP-007 — Whole-Body Real-Robot MPC | **Pending** | Protocol directory exists (`experiments/EXP-007-real-robot-mpc/`). |
| EXP-008 — Real-Robotic Vision Stress Test | **Pending** | Protocol directory exists (`experiments/EXP-008-real-vision/`). |
| EXP-009 — Contact-Rich Multi-Stage Manipulation | **Pending** | Protocol directory does **not** exist on disk yet. |
| EXP-010 — Multi-Modal Mode Recovery | **Pending** | Protocol directory does **not** exist on disk yet. |

All Horizon 2 experiments are gated on completion of Horizon 1 and on setting up the ManiSkill3/robosuite/MuJoCo sim stack.

---

## 3. Checkpoints Status

- **Local baseline checkpoints: partially ready.** `results/checkpoints/` contains four verified PushT checkpoints (`small_vla_pusht.pt`, `ddpm_pusht.pt`, `flow_matching_pusht.pt`, `mip_pusht.npz`) and `release_manifest.json` (`results/checkpoints/release_manifest.json:18–74`). The `train_and_export.log` (`results/checkpoints/train_and_export.log:1–176`) reports all four as **PASS**.
- **Per-experiment checkpoint trees: pending.** `experiments/EXP-00{1..4}/outputs/` directories are empty.
- **HF packaging: incomplete.** `dist/hf_models/` only contains `small_vla/`; `ddpm/`, `flow_matching/`, and `mip/` packages are missing. `results/hf_artifacts/` only has a stub SmallVLA model-card README and an empty dataset-card directory.

---

## 4. Hugging Face Hub Artifacts Status

- **Status: Pending / not uploaded.**
- Local packaging scripts exist:
  - `scripts/package_hf_models.py` — prepares model repos but does **not** call `push_to_hub` or `create_repo`.
  - `scripts/package_hf_dataset.py` — prepares a dataset package but does **not** push.
- No HF Hub repos have been created (`Ryukijano/smallvla-mpc-vla-diffusion-quick`, `Ryukijano/diffusion-policy-mpc-vla-diffusion-quick`, etc.) and no Gradio Spaces are deployed.
- `docs/blogging/publish_decision_analysis.md:16–19` explicitly identifies the absence of Hub artifacts as a blocker to publication.

---

## 5. Blog Status

- **Draft exists; publication waiting on full Horizon 1 results.**
- Draft: `docs/blogging/hf_blog_draft.md`
- Publish decision analysis: `docs/blogging/publish_decision_analysis.md` (dated 2026-08-13) recommends waiting 2–4 weeks until EXP-001/002/004 are complete, at least one HF Hub model and dataset are live, and the VLA baseline is validated end-to-end.
- `docs/blogging/claim_audit.md` flags that quick-test numbers must not be presented as study conclusions.

---

## 6. Current `results/` Directory Inventory

```
results/
├── env_info.json                          # DGX Spark environment provenance
├── horizon1_runbook.log                   # Failed Horizon 1 attempt log
├── checkpoints/                           # 4 verified PushT baseline checkpoints
│   ├── ddpm_pusht.pt
│   ├── flow_matching_pusht.pt
│   ├── mip_pusht.npz
│   ├── small_vla_pusht.pt
│   ├── release_manifest.json
│   └── train_and_export.log
├── quick_test/                            # Quick smoke-test outputs (1 seed, 5 eps, reaching only, no VLA)
│   ├── ablation/
│   │   ├── ablation_aggregated.csv
│   │   ├── ablation_comparison.csv
│   │   ├── ablation_results.json
│   │   └── figures/*.png
│   ├── report/
│   │   ├── master_comparison_table.csv
│   │   └── figures/*.png
│   ├── tables/aggregated_comparison.csv
│   ├── tables/master_comparison.csv
│   ├── metrics/metrics_summary.json
│   ├── metrics/full_results.json
│   ├── aggregated_comparison.csv
│   ├── master_comparison.csv
│   ├── metrics_summary.json
│   └── figures/*.png
├── hf_artifacts/                          # Stub Hub cards (incomplete)
│   ├── model_cards/smallvla-mpc-vla-diffusion-quick/README.md
│   └── dataset_cards/mpc-expert-demos-quick-test/ (empty)
└── logs/                                  # Empty
```

**No `results/exp{001,002,003,004}/` directories exist.**

---

## 7. Next Actions and Blockers

### Immediate blockers
1. **Missing evaluation seed files.** `data/` is empty; `data/eval_seeds_exp{001..004}.json` are required for paired, pre-registered evaluation.
2. **CLI/script bugs in `scripts/run_horizon1.sh`.**
   - Line 33: `run_ablation.py --benchmarks` (plural) must be `--benchmark` (singular).
   - Line 43: `run_experiments.py --benchmarks` (plural) must be `--benchmark` (singular).
   - Line 47: `run_experiments.py` does not accept `--horizon`; remove or add the argument.
3. **Canonical runner ambiguity for EXP-004.** Decide whether to use `run_experiments.py` (per `experiments/README.md`) or `scripts/run_pareto_sweep.py` (per `scripts/run_horizon1.sh`).

### Next actions
1. Generate/restore `data/eval_seeds_exp{001..004}.json`.
2. Fix `scripts/run_horizon1.sh` CLI flags and re-run the full Horizon 1 suite.
3. Run full EXP-001 first; it is a dependency for EXP-002/003/004.
4. Validate SmallVLA end-to-end on PushT with `run_experiments.py` before EXP-002.
5. Complete HF model packaging (`dist/hf_models/{ddpm,flow_matching,mip}`) and create/upload Hub repos.
6. Create HF dataset repo (`Ryukijano/mpc-expert-demos-quick-test`) and Gradio Spaces.
7. Create EXP-009 and EXP-010 protocol directories.
8. Update `docs/blogging/claim_audit.md` and `hf_blog_draft.md` once real results are in.

---

*This file is a living document. Update it after each major run, checkpoint export, or Hub upload.*
