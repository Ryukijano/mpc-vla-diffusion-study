# Data Manifest — Reproducibility Bundle

This bundle contains **no external, proprietary, or manually downloaded datasets**. All demonstration data are generated algorithmically by the repository's own MPC experts.

---

## 1. Data origin

Demonstrations are collected on-the-fly by:

- `run_experiments.py` (Phase 1 of each run)
- `run_ablation.py` (inside `collect_demonstrations()`)
- `scripts/run_pareto_sweep.py` (inside its own `collect_demonstrations()`)

The expert is a **Collision-Free MPC** controller (iLQR with SDF obstacle constraints) implemented in `mpc_baselines_repo/src/collision_free_mpc/`. It produces state-action rollouts for the learning-based baselines. For VLA experiments, the same rollouts are rendered into 96x96 top-down images with a language instruction ("reach green target while avoiding obstacles").

---

## 2. How to obtain the data

### Standalone data generation

```bash
bash repro/scripts/01_download_data.sh
```

This runs `run_experiments.py --quick` and produces:

- Quick demonstration rollouts under `results/quick_test/`
- The quick-test comparison table `results/quick_test/tables/aggregated_comparison.csv`

### Full-reproduction data

The full `02_run_experiments.sh` runner re-collects demonstrations inside each experiment automatically:

- EXP-001: 30 demos per seed/benchmark (default)
- EXP-002: up to 100 demos per seed/benchmark (network-size dependent)
- EXP-004: 30 demos per seed/benchmark

No separate download step is required for full reproduction; `01_download_data.sh` is provided mainly for smoke testing and data inspection.

---

## 3. Data formats

### State-action demonstrations

- **Format:** Python list of `(state, action_sequence)` tuples.
- **State dimensions:** 4D for `Reaching` and `Reaching (Cluttered)`, 4D for `PushT`.
- **Action dimensions:** 2D for all planar tasks.
- **Horizon:** default 15 control steps.

### VLA image-action pairs

- **Image:** 96x96x3 `uint8` top-down render.
- **Instruction:** Text string (e.g. "reach green target while avoiding obstacles").
- **Action:** Same 2D action sequence as above.

### Stored outputs

Runners save per-seed JSON results and aggregate CSV tables, for example:

- `results/quick_test/tables/aggregated_comparison.csv`
- `experiments/EXP-001-mechanism-ablation/outputs/ablation_aggregated.csv`
- `results/EXP-002/aggregated_comparison.csv` (when `02_run_experiments.sh` is used)
- `results/EXP-004/metrics_summary.json`

---

## 4. Data size and retention

Because the data are regenerated for each run, the bundle does not ship large binary files. Typical sizes on disk after a full Horizon 1 run are:

| Artifact | Approximate size |
|----------|------------------|
| Quick-test data + tables | < 5 MB |
| EXP-001 ablation outputs | 20-50 MB |
| EXP-002 full outputs | 100-300 MB |
| EXP-004 Pareto outputs + plots | 50-100 MB |
| Model checkpoints | 5-50 MB each |

---

## 5. Licenses

All generated data are released under the **MIT License** and are free to use, redistribute, and modify. See `LICENSE` in the repository root.
