# EXP-004-CPU Low-Latency Pareto Sweep Verification

## Command run

```bash
cd /home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study
conda run -n mpc_vla python scripts/run_pareto_cpu_low_latency.py \
    --seeds 0 1 \
    --episodes 10 \
    --output-dir results/exp004_cpu_low_latency
```

## Output files

| File | Path | Status |
|------|------|--------|
| Pareto data CSV | `results/exp004_cpu_low_latency/pareto_data.csv` | ✓ exists |
| Latency table CSV | `results/exp004_cpu_low_latency/latency_table.csv` | ✓ exists |
| Pareto frontier figure | `results/exp004_cpu_low_latency/pareto_frontier.png` | ✓ exists |
| Metrics summary | `results/exp004_cpu_low_latency/metrics_summary.json` | ✓ exists |

## Verification checks

- `pareto_data.csv` contains **34 rows** (17 conditions × 2 benchmarks).
- Benchmarks present: `reaching` and `pusht`.
- Total unique (benchmark, condition) pairs: **34**.
- All controller families present: Linear MPC, Nonlinear MPC, MIP, Regression, Iterative Regression.
- Latency measured with `n_warmup=10`, `n_timed=100` using `time.perf_counter_ns()` on CPU only.
- Pareto dominance computed on the full set of conditions; a second Pareto pass was computed for the low-latency subset (`latency_p50_ms < 100 ms`) and stored in `is_pareto_optimal_low_latency`.
- The plot `pareto_frontier.png` shows only the low-latency region (< 100 ms) for both benchmarks.

## Conditions swept

- **Linear MPC**: horizon ∈ {5, 10, 15, 20, 30}
- **Nonlinear MPC (iLQR)**: iterations ∈ {1, 3, 5, 10, 15}
- **MIP (Iterative Regression Policy)**: iterations ∈ {1, 2, 3, 5}
- **Regression Policy**: hidden_dim ∈ {16, 32, 64}
- **Iterative Regression Policy**: iterations ∈ {1, 2, 3, 5} (shared MIP model, different subset lengths)

Total conditions per benchmark: **17**.

## Low-latency (< 100 ms) Pareto-optimal controllers

A controller is Pareto-optimal in the low-latency region if no other tested controller has both a lower p50 latency and a higher mean success rate.

### Reaching

| Controller | Family | p50 latency (ms) | Success rate |
|------------|--------|------------------|--------------|
| Linear MPC (H=5) | Linear MPC | 0.33 | 100.0 % |

### PushT

| Controller | Family | p50 latency (ms) | Success rate |
|------------|--------|------------------|--------------|
| Linear MPC (H=5) | Linear MPC | 0.42 | 100.0 % |

## Notable observations

- `Linear MPC (H=5)` is the fastest controller and also achieves perfect success on both benchmarks, so it dominates all other low-latency conditions in this CPU-only sweep.
- `Regression (hidden=16)` reaches 85–90 % success at ~0.5–8 ms (reaching and PushT respectively) but is Pareto-dominated by `Linear MPC (H=5)`.
- `Nonlinear MPC` variants achieve 100 % success but at higher latency (3–70 ms).
- `MIP` latency grows with iterations as expected; `MIP (iters=1)` is the fastest learned baseline but its success is low on reaching (5 %), while it reaches 100 % on PushT at ~19 ms.

## Reproducibility

- Conda environment: `mpc_vla`
- Device: CPU only (`torch` limited to 4 threads; `CUDA_VISIBLE_DEVICES` not used)
- Random seeds: 0 and 1
- Evaluation: 10 episodes per controller per seed (20 total rollouts per condition)
- Expert demonstrations: 300 reactive (state, first-action) pairs collected with `CollisionFreeMPC` per benchmark
- Code: `scripts/run_pareto_cpu_low_latency.py`

Run verification script:

```bash
conda run -n mpc_vla python scripts/verify_results.py
```

## Status

**PASS**: all required files generated, CSV verified, Pareto front plotted, and low-latency Pareto-optimal controllers identified.
