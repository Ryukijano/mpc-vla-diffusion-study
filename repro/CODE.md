# Code Manifest — Reproducibility Bundle

This file documents the source code needed to reproduce the Horizon 1 experiments (EXP-001, EXP-002, EXP-004).

---

## 1. Repository root layout

```
/home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study/
├── run_experiments.py              # Main runner for EXP-002 (three-family comparison)
├── run_ablation.py                 # Main runner for EXP-001 (GCP component ablation)
├── generate_report.py              # Generates tables, plots, and PDF/MD reports
├── scripts/
│   ├── run_quick_test.sh           # Smoke test (< 2 minutes on GB10)
│   ├── run_pareto_sweep.py         # Main runner for EXP-004 (Pareto frontier)
│   ├── setup_env.sh                # Standalone environment setup (optional)
│   ├── collect_env_info.py         # Hardware/runtime provenance capture
│   ├── package_hf_dataset.py       # Hugging Face dataset packaging helper
│   └── package_hf_models.py        # Hugging Face model packaging helper
├── mpc_baselines_repo/
│   └── src/                        # Linear, Nonlinear, CollisionFree MPC, utilities
├── diffusion_baselines/            # DDPM, Flow Matching, MIP, Regression policies
├── vla_baselines/                  # SmallVLA and OpenVLA wrappers
├── benchmarks/                     # PushT, Reaching, Evaluation harness
├── src/                            # Shared source (currently sparse, reserved for future modules)
├── configs/
│   └── system_config.yaml          # Experiment defaults, network sizes, benchmark specs
├── experiments/
│   └── environment.lock            # Pinned pip freeze for mpc_vla env
└── repro/                          # This reproducibility bundle
```

---

## 2. Key run scripts

| Experiment | Command (from repo root, with `mpc_vla` active) |
|------------|-------------------------------------------------|
| Smoke test | `bash scripts/run_quick_test.sh` |
| EXP-001    | `python run_ablation.py --benchmark all --seeds 0 1 2 --episodes 50` |
| EXP-002    | `python run_experiments.py --benchmark all --controllers all --seeds 0 1 2 42 123 --episodes 100` |
| EXP-004    | `python scripts/run_pareto_sweep.py --benchmark all --seeds 0 1 2 --episodes 50` |

The bundled `repro/scripts/02_run_experiments.sh` automates the exact sequence above.

---

## 3. Module import path

All runners manage `sys.path` so the following import roots are available:

- `mpc_baselines_repo`
- `mpc_baselines_repo/src`
- `diffusion_baselines`
- `vla_baselines`
- `benchmarks`
- repository root (for `run_experiments` fallbacks)

`02_run_experiments.sh` also exports `PYTHONPATH` to the same roots before launching each runner.

---

## 4. Environment and build

- **Conda environment:** `mpc_vla`
- **Python version:** 3.11 (3.12 is also supported by `scripts/setup_env.sh`; the lock file was generated with 3.11.15)
- **PyTorch:** `2.12.0.dev20260408+cu128` from the PyTorch nightly cu128 index
- **CUDA:** 12.8
- **Pinned dependencies:** `repro/ENVIRONMENT.lock` (symlink to `experiments/environment.lock`)

Run `bash repro/scripts/00_setup.sh` to verify or recreate the environment.

---

## 5. Provenance and git

The bundle is tied to the repository state at creation. The `repro/scripts/02_run_experiments.sh` runner does **not** require a git checkout, but it preserves the exact commit hash in each runner's output log via `scripts/collect_env_info.py`.

To check the current commit from the shell:

```bash
git -C /home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study rev-parse HEAD
```

---

## 6. License

All code in the repository is released under the **MIT License**. See `LICENSE` in the repository root.
