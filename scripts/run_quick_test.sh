#!/bin/bash
# =============================================================================
# Quick smoke test for the MPC vs VLA vs Diffusion study
# =============================================================================
# Runs a tiny comparison (1 seed, 5 episodes, small networks) to verify
# that all components are working. Takes < 2 minutes on GB10.
#
# Usage:
#   bash scripts/run_quick_test.sh
#   # or
#   chmod +x scripts/run_quick_test.sh && ./scripts/run_quick_test.sh
# =============================================================================
set -e

echo "================================================================"
echo "  Quick Smoke Test -- MPC vs VLA vs Diffusion"
echo "  Target: < 2 minutes on GB10"
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
        echo "  [WARNING] mpc_vla env not found. Using current Python."
        echo "  Run scripts/setup_env.sh first to create the environment."
    fi
else
    echo "  [WARNING] conda not found. Using system Python."
fi

# --- Set PYTHONPATH ---------------------------------------------------------
export PYTHONPATH="$STUDY_ROOT:$STUDY_ROOT/mpc_baselines_repo:$STUDY_ROOT/mpc_baselines_repo/src:$STUDY_ROOT/diffusion_baselines:$STUDY_ROOT/benchmarks:$STUDY_ROOT/vla_baselines:$PYTHONPATH"

echo "  Python: $(python --version 2>&1)"
echo "  PYTHONPATH set."

# --- Track pass/fail --------------------------------------------------------
PASS_COUNT=0
FAIL_COUNT=0
RESULTS_DIR="$STUDY_ROOT/results/quick_test"
mkdir -p "$RESULTS_DIR"

run_test() {
    local test_name="$1"
    local cmd="$2"
    echo ""
    echo "----------------------------------------------------------------"
    echo "  TEST: $test_name"
    echo "----------------------------------------------------------------"
    if eval "$cmd"; then
        echo "  [PASS] $test_name"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  [FAIL] $test_name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# --- Test 1: MPC baselines import -------------------------------------------
run_test "MPC baselines import" \
    "python -c \"
import sys
sys.path.insert(0, '$STUDY_ROOT/mpc_baselines_repo')
sys.path.insert(0, '$STUDY_ROOT/mpc_baselines_repo/src')
from src.linear_mpc import LinearMPC
from src.nonlinear_mpc import NonlinearMPC
from src.collision_free_mpc import CollisionFreeMPC
from src.diffusion_warm_start import SimpleDiffusionPolicy, MinimalIterativePolicy
print('  MPC baselines: OK')
\""

# --- Test 2: Diffusion baselines import -------------------------------------
run_test "Diffusion baselines import" \
    "python -c \"
import sys
sys.path.insert(0, '$STUDY_ROOT/diffusion_baselines')
from diffusion_baselines.ddpm_policy import DiffusionPolicy
from diffusion_baselines.flow_matching_policy import FlowMatchingPolicy
print('  Diffusion baselines: OK')
\""

# --- Test 3: Benchmarks import ----------------------------------------------
run_test "Benchmarks import" \
    "python -c \"
import sys
sys.path.insert(0, '$STUDY_ROOT')
from benchmarks import ReachingEnv, PushTEnv, DemonstrationCollector, Evaluator
print('  Benchmarks: OK')
\""

# --- Test 4: VLA baselines import -------------------------------------------
run_test "VLA baselines import" \
    "python -c \"
import sys
sys.path.insert(0, '$STUDY_ROOT')
from vla_baselines import SmallVLA
print('  VLA baselines: OK (SmallVLA)')
\""

# --- Test 5: Quick experiment run (MPC + diffusion, reaching) ---------------
run_test "Quick experiment (MPC+diffusion, reaching, 1 seed, 5 episodes)" \
    "python '$STUDY_ROOT/run_experiments.py' --quick --output-dir '$RESULTS_DIR'"

# --- Test 6: Quick ablation run (reaching, 1 seed, 5 episodes) --------------
run_test "Quick ablation (reaching, 1 seed, 5 episodes)" \
    "python '$STUDY_ROOT/run_ablation.py' --benchmark reaching --seeds 0 --episodes 5 --epochs 10 --num-demos 10 --output-dir '$RESULTS_DIR/ablation'"

# --- Test 7: Report generation ----------------------------------------------
run_test "Report generation" \
    "python '$STUDY_ROOT/generate_report.py' --results-dir '$RESULTS_DIR' --output-dir '$RESULTS_DIR/report'"

# --- Summary ----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Quick Test Summary"
echo "================================================================"
echo "  Passed: $PASS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo "  Total:  $((PASS_COUNT + FAIL_COUNT))"

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo ""
    echo "  [ALL PASS] All components are working correctly!"
    echo "  Results saved to: $RESULTS_DIR"
    echo "================================================================"
    exit 0
else
    echo ""
    echo "  [SOME FAILED] $FAIL_COUNT test(s) failed. Check output above."
    echo "  Results saved to: $RESULTS_DIR"
    echo "================================================================"
    exit 1
fi
