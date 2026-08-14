#!/bin/bash
# =============================================================================
# 01_download_data.sh — Generate demonstration data
# =============================================================================
# This bundle does not require any external download. Expert demonstrations
# are generated on-the-fly by the study's Collision-Free MPC expert.
#
# This script runs a quick experiment (run_experiments.py --quick) which
# collects demonstrations, trains a tiny set of baselines, and saves the
# outputs. The generated data are placed under data/quick_demo/.
#
# Usage:
#   bash repro/scripts/01_download_data.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$STUDY_ROOT"

ENV_NAME="mpc_vla"
DATA_DIR="$STUDY_ROOT/data"

echo "================================================================"
echo "  Repro Bundle — Download / Generate Data"
echo "  Data dir: $DATA_DIR"
echo "================================================================"

# --- Activate conda environment ---------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found."
    exit 1
fi

eval "$(conda shell.bash hook)"
if conda env list | grep -q "^${ENV_NAME} "; then
    conda activate "$ENV_NAME"
    echo "  [OK] Activated: $CONDA_DEFAULT_ENV"
else
    echo "[ERROR] conda env '$ENV_NAME' not found. Run repro/scripts/00_setup.sh first."
    exit 1
fi

# --- Set PYTHONPATH ---------------------------------------------------------
export PYTHONPATH="$STUDY_ROOT:$STUDY_ROOT/mpc_baselines_repo:$STUDY_ROOT/mpc_baselines_repo/src:$STUDY_ROOT/diffusion_baselines:$STUDY_ROOT/benchmarks:$STUDY_ROOT/vla_baselines:$PYTHONPATH"

# --- Generate quick demonstration data --------------------------------------
mkdir -p "$DATA_DIR"
QUICK_OUT="$DATA_DIR/quick_demo"

echo ""
echo "[1/2] Generating expert demonstrations via run_experiments.py --quick ..."
echo "  Output: $QUICK_OUT"
python "$STUDY_ROOT/run_experiments.py" \
    --quick \
    --output-dir "$QUICK_OUT"

echo ""
echo "[2/2] Recording data manifest ..."
MANIFEST="$DATA_DIR/manifest.json"
python - << PYEOF
import json
import os
from datetime import datetime, timezone

manifest = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "source": "Collision-Free MPC expert (on-the-fly)",
    "script": "run_experiments.py --quick",
    "output_dir": "$QUICK_OUT",
    "note": "Demonstrations and quick metrics are generated algorithmically by the repository. No external dataset is required.",
}

with open("$MANIFEST", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"  Manifest saved to: $MANIFEST")
PYEOF

echo ""
echo "================================================================"
echo "  Data generation complete."
echo "================================================================"
echo "  Quick demo outputs: $QUICK_OUT"
echo "  Manifest:           $MANIFEST"
echo ""
echo "  Next step: bash repro/scripts/02_run_experiments.sh"
echo "================================================================"
