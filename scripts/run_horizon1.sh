#!/usr/bin/env bash
# Horizon 1 experiment runbook for DGX Spark (GB10)
# Runs EXP-001..004 sequentially to avoid GPU contention.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-conda}"
RUN="${CONDA} run -n mpc_vla python"

cd "${ROOT}"

echo "================================================================"
echo "  Horizon 1: EXP-001, EXP-002, EXP-004 on DGX Spark"
echo "================================================================"

# 0. Environment provenance
${RUN} scripts/collect_env_info.py --output results/env_info.json

# 1. Train and export checkpoints (skip if already present)
if [[ ! -f "results/checkpoints/small_vla_pusht.pt" || ! -f "results/checkpoints/ddpm_pusht.pt" ]]; then
  echo "[1/4] Training and exporting model checkpoints..."
  ${RUN} scripts/train_and_export_checkpoints.py \
      --num-demos 50 --horizon 8 --image-size 96 \
      --vla-epochs 20 --ddpm-epochs 40 --flow-epochs 40 --mip-epochs 40 \
      --output-dir results/checkpoints
else
  echo "[1/4] Checkpoints already exist, skipping training."
fi

# 2. EXP-001: Mechanism ablation (PushT + Reaching)
echo "[2/4] Running EXP-001: GCP Mechanism Ablation..."
${RUN} run_ablation.py \
  --benchmark all \
  --seeds 0 1 2 42 123 \
  --episodes 25 \
  --epochs 30 \
  --num-demos 50 \
  --output-dir results/exp001

# 3. EXP-002: Three-family comparison (reaching only for first full run)
echo "[3/4] Running EXP-002: Three-Family Comparison..."
${RUN} run_experiments.py \
  --benchmark reaching,reaching_cluttered \
  --controllers mpc,diffusion,vla \
  --seeds 0 1 2 42 123 \
  --episodes 10 \
  --output-dir results/exp002

# 4. EXP-004: Latency Pareto sweep (reaching focus)
echo "[4/4] Running EXP-004: Latency-Performance Pareto Sweep..."
${RUN} scripts/run_pareto_sweep.py \
  --benchmark reaching \
  --seeds 0 1 2 42 123 \
  --episodes 10 \
  --n-warmup 100 --n-timed 1000 \
  --device cuda \
  --output-dir results/exp004

echo "================================================================"
echo "  Horizon 1 complete. Results in results/exp{001,002,004}/"
echo "================================================================"
