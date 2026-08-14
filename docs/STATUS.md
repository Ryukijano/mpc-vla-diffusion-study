# Live Experiment Status Tracker

**Study:** MPC vs VLA vs Diffusion — An Open-Source Study of Robot Control Families  
**Repository:** `Ryukijano/mpc-vla-diffusion-study`  
**Compute Platform:** NVIDIA DGX Spark (`spark-1240`), GB10 Grace Blackwell Superchip, CUDA 12.8  
**Status Date:** 2026-08-14  
**Author:** Subagent E (status documentation)

---

## Executive Summary

- **Horizon 1 (EXP-001–004):** In progress.
  - **EXP-001** has a completed medium run (5 seeds, 25 episodes, PushT + Reaching, 30 epochs) under `results/exp001/`, but the full pre-registered 100-episode run is still pending.
  - **EXP-002**, **EXP-003**, and the canonical **EXP-004** GPU sweep have not been run.
  - A separate **CPU-only EXP-004 low-latency smoke test** was completed (`results/exp004_cpu_low_latency_smoke/`) with 1 seed, 2 episodes, 17 conditions on 2-D Reaching.
- **Horizon 2 (EXP-005–010):** Pending. Protocol directories for EXP-005–008 exist; EXP-009 and EXP-010 protocols are not yet on disk.
- **Checkpoints:** Baseline PushT checkpoints trained and verified in `results/checkpoints/`. Full per-condition checkpoint trees are not yet produced.
- **HF Hub Artifacts:** Local packaging is now substantially complete in `dist/hf_models/` and `dist/hf_datasets/` (plus `results/hf_artifacts/` cards), but no Hub repos have been created or uploaded.
- **Blog:** Draft exists; publication is gated on full Horizon 1 results and actual Hub uploads.

---

## 1. Horizon 1 Status (EXP-001..004)

### EXP-001 — GCP Mechanism Ablation

| Field | Value |
|-------|-------|
| **Full command (canonical)** | `conda run -n mpc_vla python run_ablation.py --benchmark all --seeds 0 1 2 42 123 --episodes 100 --output-dir experiments/EXP-001-mechanism-ablation/outputs` |
| **Most recently run command** | `conda run -n mpc_vla python run_ablation.py --benchmark all --seeds 0 1 2 42 123 --episodes 25 --epochs 30 --num-demos 50 --output-dir results/exp001` (see `scripts/run_horizon1.sh:32–38`) |
| **Quick smoke command** | `conda run -n mpc_vla python run_ablation.py --benchmark reaching --seeds 0 --episodes 5 --epochs 10 --num-demos 10 --output-dir results/quick_test/ablation` |
| **Expected outputs** | `experiments/EXP-001-mechanism-ablation/outputs/<condition>/seed_<s>/{config.yaml,checkpoint.pt,episodes.jsonl,metrics_summary.json}`; `experiments/EXP-001-mechanism-ablation/outputs/analysis/{comparison_table.csv,mechanism_ablation.pdf}` |
| **Current status** | **In-progress.** The medium 25-episode × 5-seed run on PushT + Reaching is complete and saved under `results/exp001/` (`ablation_results.json`, `ablation_aggregated.csv`, `ablation_comparison.csv`, 3 PNGs). The full pre-registered 100-episode × 5-seed run has not been executed, and `experiments/EXP-001-mechanism-ablation/outputs/` is empty. `results/horizon1_runbook.log` records an earlier failed attempt due to a now-fixed CLI mismatch (`--benchmarks` plural). |
| **Blockers** | • Full 100-episode sample size not yet reached (currently 25 episodes per seed).<br>• The medium run was saved to `results/exp001/` rather than the canonical `experiments/EXP-001-mechanism-ablation/outputs/`; should be reconciled when the full run is performed.<br>• `run_ablation.py` does not yet write `env_info.json` or config hashes at runtime per `docs/blogging/bugbot_review.md:219`. |

### EXP-002 — Three-Family Comparison

| Field | Value |
|-------|-------|
| **Full command (canonical)** | `conda run -n mpc_vla python run_experiments.py --benchmark all --controllers all --seeds 0 1 2 42 123 --episodes 100 --output-dir experiments/EXP-002-family-comparison/outputs` |
| **Planned command (`scripts/run_horizon1.sh:42–47`)** | `conda run -n mpc_vla python run_experiments.py --benchmark reaching,reaching_cluttered,pusht --controllers mpc,diffusion,vla --seeds 0 1 2 42 123 --episodes 25 --output-dir results/exp002` |
| **Quick smoke command** | `conda run -n mpc_vla python run_experiments.py --quick` |
| **Expected outputs** | `experiments/EXP-002-family-comparison/outputs/<condition>/<benchmark>/seed_<s>/{config.yaml,checkpoint.pt,episodes.jsonl,metrics_summary.json}`; `experiments/EXP-002-family-comparison/outputs/analysis/{comparison_table.csv,radar_*.png}` |
| **Current status** | **Pending / not started.** Only the quick smoke test on 2-D Reaching exists (`results/quick_test/`, 1 seed, 5 episodes, `mpc` + `diffusion` only; VLA excluded). No `results/exp002/` or `experiments/EXP-002-family-comparison/outputs/` directory. |
| **Blockers** | • Depends on EXP-001 MIP condition + replication gate (medium run available, full run pending).<br>• VLA has not been included in any integrated run yet; `results/checkpoints/small_vla_pusht.pt` exists but `run_experiments.py` VLA path has not been exercised on PushT.<br>• `run_experiments.py` quick mode defaults to `reaching` only and excludes VLA by design (`run_experiments.py:937–938`). |

### EXP-003 — OOD Robustness

| Field | Value |
|-------|-------|
| **Full command (canonical)** | `conda run -n mpc_vla python run_experiments.py --benchmark pusht --controllers diffusion --seeds 0 1 2 42 123 --episodes 100 --output-dir experiments/EXP-003-ood-robustness/outputs` |
| **Expected outputs** | `experiments/EXP-003-ood-robustness/outputs/<condition>/seed_<s>/{config.yaml,checkpoint.pt,episodes.jsonl,metrics_summary.json}`; `experiments/EXP-003-ood-robustness/outputs/analysis/{anova_results.json,degradation_slopes.csv,manifold_adherence_curves.pdf}` |
| **Current status** | **Pending.** No outputs produced; `experiments/EXP-003-ood-robustness/outputs/` is empty. |
| **Blockers** | • Depends on EXP-001 trained checkpoints being available; only the baseline `results/checkpoints/` and the medium `results/exp001/` ablation summary exist, not a full per-condition checkpoint tree.<br>• `run_experiments.py` does not currently expose perturbation-level flags; may need a dedicated runner or protocol-specific harness.<br>• `data/eval_seeds_exp003.json` is not present (no `data/eval_seeds_*.json` files exist). |

### EXP-004 — Latency-Performance Pareto

| Field | Value |
|-------|-------|
| **Full command (README canonical)** | `conda run -n mpc_vla python run_experiments.py --benchmark all --controllers all --seeds 0 1 2 42 123 --episodes 100 --output-dir experiments/EXP-004-latency-pareto/outputs` |
| **Planned GPU sweep (`scripts/run_horizon1.sh:51–57`)** | `conda run -n mpc_vla python scripts/run_pareto_sweep.py --benchmark all --seeds 0 1 2 42 123 --episodes 25 --n-warmup 100 --n-timed 1000 --device cuda --output-dir results/exp004` |
| **CPU-only smoke that ran** | `conda run -n mpc_vla python scripts/run_pareto_cpu_low_latency.py --seeds 0 1 --episodes 10 --output-dir results/exp004_cpu_low_latency_smoke` (`scripts/run_pareto_cpu_low_latency.py:20`) |
| **Expected outputs** | `experiments/EXP-004-latency-pareto/outputs/<condition>/<benchmark>/seed_<s>/{config.yaml,checkpoint.pt,episodes.jsonl,latency_timings.json,metrics_summary.json}`; `experiments/EXP-004-latency-pareto/outputs/analysis/{pareto_frontier.csv,pareto_pusht.pdf,pareto_reaching.pdf,step_saturation_curves.pdf}` |
| **Current status** | **In-progress (smoke only).** A CPU-only low-latency smoke test completed in `results/exp004_cpu_low_latency_smoke/` with 1 seed, 2 episodes, 17 conditions on 2-D Reaching (`latency_table.csv`, `pareto_data.csv`, `pareto_frontier.png`, `metrics_summary.json`). This is **not** the canonical GPU Pareto sweep. No `results/exp004/` or `experiments/EXP-004-latency-pareto/outputs/` directory. |
| **Blockers** | • Full 21-condition GPU Pareto sweep not yet run.<br>• 210,000 timed latency measurements require fixed GPU clocks and dedicated DGX Spark time.<br>• Canonical runner ambiguity: `experiments/README.md:98–101` points to `run_experiments.py`, while `scripts/run_horizon1.sh:51–57` points to `scripts/run_pareto_sweep.py`.<br>• EXP-001 DDPM T=100 checkpoint consistency gate must pass before the Pareto step sweep. |

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

- **Baseline PushT checkpoints: ready and verified.** `results/checkpoints/` contains four checkpoints (`small_vla_pusht.pt`, `ddpm_pusht.pt`, `flow_matching_pusht.pt`, `mip_pusht.npz`) and `release_manifest.json` (`results/checkpoints/release_manifest.json:18–74`). The `train_and_export.log` reports all four as **PASS**.
- **Per-condition experiment checkpoint trees: pending.** `experiments/EXP-00{1..4}/outputs/` are empty. The medium EXP-001 run produced summary CSVs/JSON but not per-condition `config.yaml` / `checkpoint.pt` trees in the canonical location.
- **HF-ready packages: ready locally.** `dist/hf_models/` now contains packaged SmallVLA, DDPM, Flow Matching, and MIP model directories (each with `.pt`/`.npz`, `config.yaml`, `README.md`, and `example_inference.py`). `dist/hf_models/test-model/` is a dry-run package that can be removed before upload.

---

## 4. Hugging Face Hub Artifacts Status

- **Status: Packaging complete locally; upload pending.**
- Model cards and packages:
  - `dist/hf_models/{small_vla,ddpm,flow_matching,mip}/`
  - `results/hf_artifacts/model_cards/{smallvla,ddpm,flow-matching,mip}-mpc-vla-diffusion-quick/README.md`
  - `results/hf_artifacts/UPLOAD_CHECKLIST.md` with placeholder `hf upload` commands
- Dataset:
  - `dist/hf_datasets/mpc_expert_demos/` with `README.md`, `.parquet`, and `.npz` files
  - `results/hf_artifacts/dataset_cards/mpc-expert-demos-quick-test/README.md`
- **No Hub repos have been created or pushed.** No `push_to_hub`, `create_repo`, or `huggingface-cli upload` calls were found in the packaging scripts (`scripts/package_hf_models.py`, `scripts/package_hf_dataset.py`). `results/hf_artifacts/UPLOAD_CHECKLIST.md:95` explicitly says "Do not upload yet."

---

## 5. Blog Status

- **Draft exists; publication waiting on full Horizon 1 results and Hub uploads.**
- Draft: `docs/blogging/hf_blog_draft.md`
- Publish decision analysis: `docs/blogging/publish_decision_analysis.md` (dated 2026-08-13) recommends waiting until EXP-001/002/004 are complete, at least one HF Hub model and dataset are live, and the VLA baseline is validated end-to-end.
- `docs/blogging/claim_audit.md` flags that quick-test and smoke-test numbers must not be presented as study conclusions.

---

## 6. Current `results/` Directory Inventory

```
results/
├── env_info.json                          # DGX Spark environment provenance
├── horizon1_runbook.log                   # Earlier failed Horizon 1 attempt log (now fixed)
├── checkpoints/                           # 4 verified PushT baseline checkpoints
│   ├── ddpm_pusht.pt
│   ├── flow_matching_pusht.pt
│   ├── mip_pusht.npz
│   ├── small_vla_pusht.pt
│   ├── release_manifest.json
│   └── train_and_export.log
├── exp001/                                # Completed medium EXP-001 run (5 seeds, 25 eps, PushT + Reaching)
│   ├── ablation_aggregated.csv
│   ├── ablation_comparison.csv
│   ├── ablation_results.json
│   └── figures/*.png
├── exp004_cpu_low_latency_smoke/          # CPU-only EXP-004 smoke (1 seed, 2 eps, reaching, 17 conds)
│   ├── latency_table.csv
│   ├── metrics_summary.json
│   ├── pareto_data.csv
│   └── pareto_frontier.png
├── quick_test/                            # Original quick smoke-test outputs (1 seed, 5 eps, reaching, no VLA)
│   ├── ablation/*.csv, *.json, figures/*.png
│   ├── report/*.csv, figures/*.png
│   ├── tables/*.csv
│   ├── metrics/*.json
│   └── figures/*.png
├── hf_artifacts/                          # HF Hub card templates + upload checklist
│   ├── UPLOAD_CHECKLIST.md
│   ├── model_cards/*/
│   └── dataset_cards/*/
└── hf_datasets/                           # Packaged expert demonstrations
    └── mpc_expert_demos/
        ├── README.md
        ├── mpc_expert_demos_state.parquet
        ├── mpc_expert_demos_state.npz
        └── mpc_expert_demos_images.npz
```

**Notable missing items:**
- `experiments/EXP-00{1..4}/outputs/` are all empty.
- No `results/exp002/` or `results/exp003/` directory.
- No `data/eval_seeds_exp{001..004}.json` seed files.

---

## 7. Next Actions and Blockers

### Immediate blockers
1. **Full 100-episode EXP-001 run.** The current 25-episode run is a good smoke test but below the pre-registered sample size.
2. **No `data/eval_seeds_exp{001..004}.json`.** These fixed paired-evaluation seed files are required by the protocols and are currently absent.
3. **EXP-002 not started.** It is gated on EXP-001 and requires VLA to be exercised end-to-end on PushT.
4. **Canonical EXP-004 GPU sweep not started.** The CPU smoke is only a low-latency placeholder.

### Next actions
1. Decide whether the 25-episode `results/exp001/` run is sufficient for an interim blog or if a full 100-episode re-run is required.
2. Run full EXP-001 (100 episodes × 5 seeds) to the canonical `experiments/EXP-001-mechanism-ablation/outputs/`.
3. Generate/restore `data/eval_seeds_exp{001..004}.json`.
4. Run EXP-002 three-family comparison (including SmallVLA on PushT).
5. Run canonical EXP-004 GPU Pareto sweep (`scripts/run_pareto_sweep.py` or `run_experiments.py`).
6. Upload HF Hub model and dataset repos (using `huggingface-cli upload` or `hf upload`) once final artifacts are ready.
7. Remove the dry-run `dist/hf_models/test-model/` package before upload.
8. Create EXP-009 and EXP-010 protocol directories.
9. Update `docs/blogging/claim_audit.md` and `hf_blog_draft.md` with real, full-sample results.

---

*This file is a living document. Update it after each major run, checkpoint export, or Hub upload.*
