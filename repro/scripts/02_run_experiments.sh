#!/bin/bash
# =============================================================================
# 02_run_experiments.sh — One-command Horizon 1 reproduction
# =============================================================================
# Runs the four core steps for the Horizon 1 artifact release:
#   1. Smoke test              (scripts/run_quick_test.sh)
#   2. EXP-001 GCP ablation    (run_ablation.py)
#   3. EXP-002 three-family comparison (run_experiments.py)
#   4. EXP-004 Pareto sweep    (scripts/run_pareto_sweep.py)
#
# Expected runtime: ~15-17 GPU-hours on NVIDIA DGX Spark (GB10).
#
# Usage:
#   bash repro/scripts/02_run_experiments.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$STUDY_ROOT"

ENV_NAME="mpc_vla"

echo "================================================================"
echo "  Repro Bundle — Horizon 1 Reproduction (EXP-001..004)"
echo "  Study root: $STUDY_ROOT"
echo "  Started:    $(date)"
echo "================================================================"

# --- Check / activate conda environment -------------------------------------
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Please install conda and run 00_setup.sh."
    exit 1
fi

eval "$(conda shell.bash hook)"
if conda env list | grep -q "^${ENV_NAME} "; then
    conda activate "$ENV_NAME"
    echo "  [OK] Activated conda env: $CONDA_DEFAULT_ENV"
else
    echo "[ERROR] conda env '$ENV_NAME' not found. Run repro/scripts/00_setup.sh first."
    exit 1
fi

# --- Set PYTHONPATH ---------------------------------------------------------
export PYTHONPATH="$STUDY_ROOT:$STUDY_ROOT/mpc_baselines_repo:$STUDY_ROOT/mpc_baselines_repo/src:$STUDY_ROOT/diffusion_baselines:$STUDY_ROOT/benchmarks:$STUDY_ROOT/vla_baselines:$PYTHONPATH"

echo ""
echo "  Python:     $(python --version 2>&1)"
echo "  PyTorch:    $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  CUDA avail: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo ""

# --- Step 1: Smoke test -----------------------------------------------------
echo "================================================================"
echo "  STEP 1 / 4: Smoke test (scripts/run_quick_test.sh)"
echo "  Target: < 2 minutes"
echo "================================================================"
bash "$STUDY_ROOT/scripts/run_quick_test.sh"
echo "  [OK] Smoke test passed."

# --- Step 2: EXP-001 GCP component ablation ---------------------------------
echo ""
echo "================================================================"
echo "  STEP 2 / 4: EXP-001 — GCP Component Ablation"
echo "  Benchmarks: all (pusht, reaching)"
echo "  Seeds:      0 1 2"
echo "  Episodes:   50"
echo "  Expected:   ~3 hours"
echo "================================================================"
python "$STUDY_ROOT/run_ablation.py" \
    --benchmark all \
    --seeds 0 1 2 \
    --episodes 50 \
    --epochs 50 \
    --num-demos 30 \
    --output-dir "$STUDY_ROOT/experiments/EXP-001-mechanism-ablation/outputs"
echo "  [OK] EXP-001 complete."

# --- Step 3: EXP-002 three-family comparison --------------------------------
echo ""
echo "================================================================"
echo "  STEP 3 / 4: EXP-002 — Three-Family Head-to-Head"
echo "  Benchmarks: all"
echo "  Controllers: all"
echo "  Seeds:      0 1 2 42 123"
echo "  Episodes:   100"
echo "  Network:    medium"
echo "  Expected:   ~6 hours"
echo "================================================================"
python "$STUDY_ROOT/run_experiments.py" \
    --benchmark all \
    --controllers all \
    --seeds 0 1 2 42 123 \
    --episodes 100 \
    --net-size medium \
    --output-dir "$STUDY_ROOT/results/EXP-002"
echo "  [OK] EXP-002 complete."

# --- Step 4: EXP-004 Pareto sweep -------------------------------------------
echo ""
echo "================================================================"
echo "  STEP 4 / 4: EXP-004 — Latency-Performance Pareto Sweep"
echo "  Benchmarks: all"
echo "  Seeds:      0 1 2"
echo "  Episodes:   50"
echo "  Expected:   ~5 hours"
echo "================================================================"
python "$STUDY_ROOT/scripts/run_pareto_sweep.py" \
    --benchmark all \
    --seeds 0 1 2 \
    --episodes 50 \
    --output-dir "$STUDY_ROOT/results/EXP-004"
echo "  [OK] EXP-004 complete."

# --- Summary ----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Horizon 1 reproduction complete."
echo "  Finished:   $(date)"
echo "================================================================"
echo "  Outputs:"
echo "    - Smoke test:  $STUDY_ROOT/results/quick_test/"
echo "    - EXP-001:     $STUDY_ROOT/experiments/EXP-001-mechanism-ablation/outputs/"
echo "    - EXP-002:     $STUDY_ROOT/results/EXP-002/"
echo "    - EXP-004:     $STUDY_ROOT/results/EXP-004/"
echo ""
echo "  To generate a report:"
echo "    python generate_report.py --results-dir $STUDY_ROOT/results --output-dir $STUDY_ROOT/results/report"
echo "================================================================"
