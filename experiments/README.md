# Experiments — MPC vs VLA vs Diffusion Study

This directory contains the pre-registered experiment protocols for the comparative
study of three controller families — Classical MPC, Vision-Language-Action (VLA)
models, and Diffusion / Generative Control Policies (GCPs) — plus their hybrids.

All protocols follow the skill sequence: **design → controls → pre-register →
execute → verify → refine**. Each protocol is pre-registered before execution;
no changes to hypotheses, conditions, sample size, or analysis plan are permitted
after the first evaluation episode is run.

---

## Experiment Overview

| ID | Title | Hypothesis | Primary Outcome | Episodes | Status |
|----|-------|------------|-----------------|----------|--------|
| [EXP-001](EXP-001-mechanism-ablation/protocol.md) | GCP Mechanism Ablation | H3 (mechanism of GCP advantage: distribution fitting vs iterative compute + noise) | Success rate (5 conditions × 2 benchmarks) | 5,000 | Pre-registered |
| [EXP-002](EXP-002-family-comparison/protocol.md) | Three-Family Comparison | H5 (MPC niche), H6 (VLA niche), H7 (hybrid), H8 (Pareto, partial) | Success rate (8 conditions × 3 benchmarks) | 12,000 | Pre-registered |
| [EXP-003](EXP-003-ood-robustness/protocol.md) | OOD Robustness | H4 (noise + iterative compute → OOD robustness) | Success rate per perturbation level (4 conditions × 5 levels) | 10,000 | Pre-registered |
| [EXP-004](EXP-004-latency-pareto/protocol.md) | Latency-Performance Pareto | H8 (latency–performance Pareto frontier) | Success rate vs inference latency (Pareto plot, 21 conditions × 2 benchmarks) | 21,000 | Pre-registered |

**Total episodes across all experiments:** 48,000 (+ 210,000 latency measurements in EXP-004).

---

## How to Run

### Prerequisites

1. **Conda environment:** The `mpc_vla` conda environment must exist with all
   dependencies installed. See `environment.lock` for the full pinned package
   list. To recreate:

   ```bash
   conda create -n mpc_vla python=3.11
   conda activate mpc_vla
   pip install torch==2.12.0.dev20260408+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
   pip install -r experiments/environment.lock
   ```

2. **Hardware:** NVIDIA DGX Spark with GB10 GPU (sm_121), 128 GB unified memory,
   CUDA 12.8. Latency measurements (EXP-004) require this specific hardware for
   reproducibility.

3. **Data:** Evaluation seed files must be present in `data/`:
   - `data/eval_seeds_exp001.json`
   - `data/eval_seeds_exp002.json`
   - `data/eval_seeds_exp003.json`
   - `data/eval_seeds_exp004.json`

   These files contain fixed episode initial states (committed and hashed) to
   ensure paired comparisons across conditions.

4. **Git:** The study repository must be initialized. The current commit hash is
   recorded in `environment.lock` and logged at run time in each experiment's
   `env_info.json`.

### Running all experiments

```bash
# Full run (all 4 experiments, 5 seeds, 100 episodes each)
./experiments/run_all.sh

# Quick smoke test (fewer seeds, fewer episodes, smaller networks)
./experiments/run_all.sh --quick
```

### Running a single experiment

Each experiment can be run individually via the main runner:

```bash
# EXP-001: Mechanism ablation
conda run -n mpc_vla python run_experiments.py \
    --benchmark all --controllers diffusion \
    --seeds 0 1 2 42 123 --episodes 100 \
    --output-dir experiments/EXP-001-mechanism-ablation/outputs

# EXP-002: Three-family comparison
conda run -n mpc_vla python run_experiments.py \
    --benchmark all --controllers all \
    --seeds 0 1 2 42 123 --episodes 100 \
    --output-dir experiments/EXP-002-family-comparison/outputs

# EXP-003: OOD robustness
conda run -n mpc_vla python run_experiments.py \
    --benchmark pusht --controllers diffusion \
    --seeds 0 1 2 42 123 --episodes 100 \
    --output-dir experiments/EXP-003-ood-robustness/outputs

# EXP-004: Latency-Pareto sweep
conda run -n mpc_vla python run_experiments.py \
    --benchmark all --controllers all \
    --seeds 0 1 2 42 123 --episodes 100 \
    --output-dir experiments/EXP-004-latency-pareto/outputs
```

### Running analysis

After an experiment completes, run its analysis script (if available):

```bash
conda run -n mpc_vla python experiments/EXP-001-mechanism-ablation/analyze.py
conda run -n mpc_vla python experiments/EXP-002-family-comparison/analyze.py
conda run -n mpc_vla python experiments/EXP-003-ood-robustness/analyze.py
conda run -n mpc_vla python experiments/EXP-004-latency-pareto/analyze.py
```

---

## Expected Timeline

| Experiment | Est. Wall-Clock (full) | Est. Wall-Clock (quick) | Bottleneck |
|------------|----------------------|------------------------|------------|
| EXP-001 | ~3 h | ~10 min | Training 5 conditions × 5 seeds |
| EXP-002 | ~6 h | ~15 min | Training 8 conditions × 5 seeds + MPC solve time |
| EXP-003 | ~2 h | ~5 min | Evaluation only (reuses EXP-001 checkpoints) |
| EXP-004 | ~5 h | ~15 min | 21 conditions × latency measurements (1000 calls each) |
| **Total** | **~16 h** | **~45 min** | |

Times are estimates for the DGX Spark (GB10). Actual times depend on network
size, benchmark complexity, and whether checkpoints from earlier experiments can
be reused.

---

## Experiment Dependencies

```
EXP-001 (mechanism ablation)
  │
  ├──→ EXP-002 (family comparison)     [depends on EXP-001 MIP condition + replication gate]
  │
  ├──→ EXP-003 (OOD robustness)        [depends on EXP-001 level-0 consistency]
  │
  └──→ EXP-004 (latency Pareto)        [depends on EXP-001 DDPM T=100 consistency gate]
```

EXP-001 must run first. EXP-002, EXP-003, and EXP-004 can run in parallel after
EXP-001 passes its replication gate, but `run_all.sh` runs them sequentially for
simplicity and reproducibility.

---

## Directory Structure

```
experiments/
├── README.md                              ← this file
├── environment.lock                       ← pinned environment manifest
├── run_all.sh                             ← master runner script
├── EXP-001-mechanism-ablation/
│   ├── protocol.md                        ← pre-registered protocol
│   └── outputs/                           ← results (generated at run time)
├── EXP-002-family-comparison/
│   ├── protocol.md
│   └── outputs/
├── EXP-003-ood-robustness/
│   ├── protocol.md
│   └── outputs/
└── EXP-004-latency-pareto/
    ├── protocol.md
    └── outputs/
```

---

## Links

- Study root: `../README.md`
- Comparison plan: `../docs/comparison_plan.md`
- Research questions: `../docs/research_questions.md`
- Methodology: `../docs/methodology.md`
- Main runner: `../run_experiments.py`
- Environment lock: `environment.lock`
