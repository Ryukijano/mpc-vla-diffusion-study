#!/bin/bash
# =============================================================================
# Full experiment runner for NVIDIA DGX Spark (GB10 Grace Blackwell)
# =============================================================================
# Runs the complete MPC vs VLA vs Diffusion comparison:
#   - All controllers (MPC, VLA, Diffusion)
#   - All benchmarks (PushT, Reaching, Reaching Cluttered)
#   - 5 seeds, 100 episodes
#   - Saves results to results/ with timestamp
#   - Runs EXP-001 ablation
#   - Generates final report
#
# Usage:
#   bash scripts/run_on_spark.sh
#   # or
#   chmod +x scripts/run_on_spark.sh && ./scripts/run_on_spark.sh
# =============================================================================
set -e

echo "================================================================"
echo "  Full Experiment Runner -- DGX Spark (GB10)"
echo "  MPC vs VLA vs Diffusion Comparison Study"
echo "================================================================"

# --- Determine study root ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$STUDY_ROOT"

# --- Activate conda environment ---------------------------------------------
echo ""
echo "[setup] Activating mpc_vla environment..."

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^mpc_vla "; then
        conda activate mpc_vla
        echo "  [OK] Activated: $CONDA_DEFAULT_ENV"
    else
        echo "  [ERROR] mpc_vla conda environment not found."
        echo "  Run scripts/setup_env.sh first to create the environment."
        exit 1
    fi
else
    echo "  [ERROR] conda not found. Please install conda and run scripts/setup_env.sh."
    exit 1
fi

# --- Set PYTHONPATH ---------------------------------------------------------
export PYTHONPATH="$STUDY_ROOT:$STUDY_ROOT/mpc_baselines_repo:$STUDY_ROOT/mpc_baselines_repo/src:$STUDY_ROOT/diffusion_baselines:$STUDY_ROOT/benchmarks:$STUDY_ROOT/vla_baselines:$PYTHONPATH"

echo "  Python:     $(python --version 2>&1)"
echo "  PyTorch:    $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "  CUDA avail: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  Study root: $STUDY_ROOT"

# --- Timestamped output directory -------------------------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="$STUDY_ROOT/results/spark_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "  Output dir: $RESULTS_DIR"

# --- Log file ---------------------------------------------------------------
LOG_FILE="$RESULTS_DIR/experiment.log"
exec > >(tee "$LOG_FILE") 2>&1

echo ""
echo "================================================================"
echo "  Experiment started at: $(date)"
echo "  Log file: $LOG_FILE"
echo "================================================================"

# --- Phase 1: Main comparison experiment ------------------------------------
echo ""
echo "================================================================"
echo "  PHASE 1: Main Comparison Experiment"
echo "  All controllers, all benchmarks, 5 seeds, 100 episodes"
echo "================================================================"

python "$STUDY_ROOT/run_experiments.py" \
    --benchmark all \
    --controllers all \
    --seeds 0 1 2 42 123 \
    --episodes 100 \
    --net-size medium \
    --output-dir "$RESULTS_DIR"

echo ""
echo "  [OK] Main comparison complete."

# --- Phase 2: EXP-001 Ablation ----------------------------------------------
echo ""
echo "================================================================"
echo "  PHASE 2: EXP-001 GCP Component Ablation"
echo "  PushT + Reaching, 3 seeds, 50 episodes"
echo "================================================================"

python "$STUDY_ROOT/run_ablation.py" \
    --benchmark all \
    --seeds 0 1 2 \
    --episodes 50 \
    --epochs 50 \
    --num-demos 30 \
    --output-dir "$RESULTS_DIR/experiments/EXP-001-mechanism-ablation/outputs"

echo ""
echo "  [OK] Ablation complete."

# --- Phase 3: Generate report -----------------------------------------------
echo ""
echo "================================================================"
echo "  PHASE 3: Generate Report"
echo "================================================================"

python "$STUDY_ROOT/generate_report.py" \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$RESULTS_DIR/report" \
    --format both

echo ""
echo "  [OK] Report generated."

# --- Summary ----------------------------------------------------------------
END_TIME=$(date)
echo ""
echo "================================================================"
echo "  Full Experiment Complete!"
echo "================================================================"
echo "  Started:  $(head -1 "$LOG_FILE" | grep -o 'at:.*' || echo 'see log')"
echo "  Finished: $END_TIME"
echo "  Results:  $RESULTS_DIR"
echo "  Report:   $RESULTS_DIR/report/"
echo "  Log:      $LOG_FILE"
echo ""
echo "  Key outputs:"
echo "    - Master table:     $RESULTS_DIR/tables/aggregated_comparison.csv"
echo "    - Ablation table:   $RESULTS_DIR/experiments/EXP-001-mechanism-ablation/outputs/ablation_aggregated.csv"
echo "    - Comparison charts: $RESULTS_DIR/report/figures/"
echo "    - Pareto plot:      $RESULTS_DIR/report/figures/pareto_latency_vs_success.png"
echo "================================================================"
