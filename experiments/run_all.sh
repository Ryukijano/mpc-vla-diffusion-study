#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Master experiment runner for MPC vs VLA vs Diffusion study
# =============================================================================
# Runs all four pre-registered experiments (EXP-001 through EXP-004) in order,
# saving outputs to experiments/<exp-id>/outputs/, and generates a summary
# table at the end.
#
# Usage:
#   ./experiments/run_all.sh              # full run (5 seeds, 100 episodes)
#   ./experiments/run_all.sh --quick      # quick smoke test (1 seed, 5 episodes)
#
# Prerequisites:
#   - conda env "mpc_vla" with all dependencies (see experiments/environment.lock)
#   - Evaluation seed files in data/ (eval_seeds_exp00{1,2,3,4}.json)
#   - NVIDIA DGX Spark with GB10 GPU (sm_121), CUDA 12.8
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="mpc_vla"
SEEDS_DEFAULT="0 1 2 42 123"
EPISODES_DEFAULT=100

# Quick-mode overrides
SEEDS_QUICK="0"
EPISODES_QUICK=5

# Experiment IDs in execution order
EXPERIMENTS=(
    "EXP-001-mechanism-ablation"
    "EXP-002-family-comparison"
    "EXP-003-ood-robustness"
    "EXP-004-latency-pareto"
)

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
QUICK_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--quick]"
            echo ""
            echo "Options:"
            echo "  --quick    Quick smoke test (1 seed, 5 episodes, tiny networks)"
            echo "  -h, --help Show this help message"
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            echo "Usage: $0 [--quick]" >&2
            exit 1
            ;;
    esac
done

# Select parameters based on mode
if [[ "${QUICK_MODE}" == "true" ]]; then
    SEEDS=${SEEDS_QUICK}
    EPISODES=${EPISODES_QUICK}
    EXTRA_ARGS="--quick"
    echo "=========================================="
    echo "  QUICK MODE (smoke test)"
    echo "  Seeds: ${SEEDS} | Episodes: ${EPISODES}"
    echo "=========================================="
else
    SEEDS=${SEEDS_DEFAULT}
    EPISODES=${EPISODES_DEFAULT}
    EXTRA_ARGS=""
    echo "=========================================="
    echo "  FULL MODE (pre-registered)"
    echo "  Seeds: ${SEEDS} | Episodes: ${EPISODES}"
    echo "=========================================="
fi

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
echo ""
echo "[setup] Study root: ${STUDY_ROOT}"
echo "[setup] Script dir:  ${SCRIPT_DIR}"
echo "[setup] Conda env:   ${CONDA_ENV}"

# Activate conda environment
# Try conda activate first; fall back to conda run
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "[setup] Conda environment activated: ${CONDA_ENV}"
else
    echo "[WARNING] Could not find conda base. Will use 'conda run -n ${CONDA_ENV}' per command."
    CONDA_RUN="conda run -n ${CONDA_ENV}"
fi

# Set PYTHONPATH for all module directories
MODULE_DIRS=(
    "${STUDY_ROOT}/mpc_baselines_repo"
    "${STUDY_ROOT}/vla_baselines"
    "${STUDY_ROOT}/diffusion_baselines"
    "${STUDY_ROOT}/benchmarks"
    "${STUDY_ROOT}/src"
    "${STUDY_ROOT}/mpc_baselines_repo/src"
    "${STUDY_ROOT}"
)
PYTHONPATH_EXPORT=""
for dir in "${MODULE_DIRS[@]}"; do
    if [[ -d "${dir}" ]]; then
        PYTHONPATH_EXPORT="${dir}:${PYTHONPATH_EXPORT}"
    fi
done
export PYTHONPATH="${PYTHONPATH_EXPORT}${PYTHONPATH:-}"
echo "[setup] PYTHONPATH set (${#MODULE_DIRS[@]} module dirs)"

# Record git commit
GIT_COMMIT="$(cd "${STUDY_ROOT}" && git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "[setup] Git commit: ${GIT_COMMIT}"

# Record start time
START_TIME=$(date +%s)
echo "[setup] Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# -----------------------------------------------------------------------------
# Results tracking
# -----------------------------------------------------------------------------
declare -A EXP_STATUS
declare -A EXP_DURATION

# -----------------------------------------------------------------------------
# Run experiments
# -----------------------------------------------------------------------------
for EXP_ID in "${EXPERIMENTS[@]}"; do
    EXP_DIR="${SCRIPT_DIR}/${EXP_ID}"
    OUTPUT_DIR="${EXP_DIR}/outputs"

    echo "=================================================================="
    echo "  Running ${EXP_ID}"
    echo "=================================================================="
    echo "  Output: ${OUTPUT_DIR}"
    echo ""

    # Create output directory
    mkdir -p "${OUTPUT_DIR}"

    EXP_START=$(date +%s)

    # Build the command
    # The main runner (run_experiments.py) handles all controller/benchmark logic.
    # We pass the output directory and seed/episode overrides.
    CMD="python ${STUDY_ROOT}/run_experiments.py \
        --output-dir ${OUTPUT_DIR} \
        --seeds ${SEEDS} \
        --episodes ${EPISODES} \
        ${EXTRA_ARGS}"

    echo "[exec] ${CMD}"
    echo ""

    # Run the experiment; capture output to a log file
    LOG_FILE="${OUTPUT_DIR}/run.log"
    if ${CMD} 2>&1 | tee "${LOG_FILE}"; then
        EXP_STATUS[${EXP_ID}]="PASS"
        echo ""
        echo "[result] ${EXP_ID}: PASS"
    else
        EXP_STATUS[${EXP_ID}]="FAIL"
        echo ""
        echo "[result] ${EXP_ID}: FAIL (see ${LOG_FILE})"
        echo "[WARNING] ${EXP_ID} failed. Continuing to next experiment."
    fi

    EXP_END=$(date +%s)
    EXP_DURATION[${EXP_ID}]=$(( EXP_END - EXP_START ))
    echo "[timing] ${EXP_ID} took ${EXP_DURATION[${EXP_ID}]}s"
    echo ""
done

# -----------------------------------------------------------------------------
# Generate summary table
# -----------------------------------------------------------------------------
END_TIME=$(date +%s)
TOTAL_DURATION=$(( END_TIME - START_TIME ))

SUMMARY_FILE="${SCRIPT_DIR}/summary.txt"
{
    echo "================================================================"
    echo "  Experiment Summary"
    echo "================================================================"
    echo "  Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Git commit: ${GIT_COMMIT}"
    echo "  Mode: $(if [[ "${QUICK_MODE}" == "true" ]]; then echo "quick"; else echo "full"; fi)"
    echo "  Seeds: ${SEEDS} | Episodes: ${EPISODES}"
    echo "  Total wall-clock: ${TOTAL_DURATION}s ($(( TOTAL_DURATION / 60 ))m $(( TOTAL_DURATION % 60 ))s)"
    echo ""
    printf "  %-35s %-8s %10s\n" "Experiment" "Status" "Duration"
    printf "  %-35s %-8s %10s\n" "-----------------------------------" "--------" "----------"
    for EXP_ID in "${EXPERIMENTS[@]}"; do
        printf "  %-35s %-8s %9ss\n" \
            "${EXP_ID}" \
            "${EXP_STATUS[${EXP_ID}]:-N/A}" \
            "${EXP_DURATION[${EXP_ID}]:-0}"
    done
    echo ""
    echo "  Output directories:"
    for EXP_ID in "${EXPERIMENTS[@]}"; do
        echo "    experiments/${EXP_ID}/outputs/"
    done
    echo ""
    echo "  Full logs:"
    for EXP_ID in "${EXPERIMENTS[@]}"; do
        echo "    experiments/${EXP_ID}/outputs/run.log"
    done
    echo "================================================================"
} | tee "${SUMMARY_FILE}"

echo ""
echo "[done] Summary written to ${SUMMARY_FILE}"
echo "[done] Total time: ${TOTAL_DURATION}s ($(( TOTAL_DURATION / 60 ))m $(( TOTAL_DURATION % 60 ))s)"

# Exit with non-zero if any experiment failed
ALL_PASS=true
for EXP_ID in "${EXPERIMENTS[@]}"; do
    if [[ "${EXP_STATUS[${EXP_ID}]}" != "PASS" ]]; then
        ALL_PASS=false
    fi
done

if [[ "${ALL_PASS}" == "true" ]]; then
    echo "[done] All experiments passed."
    exit 0
else
    echo "[done] One or more experiments failed. See summary above."
    exit 1
fi
