#!/bin/bash
# =============================================================================
# 00_setup.sh — Reproducibility bundle environment setup
# =============================================================================
# 1. Verifies conda is available.
# 2. Verifies (or creates) the `mpc_vla` conda environment.
# 3. Installs the pinned dependencies from experiments/environment.lock.
# 4. Verifies CUDA / PyTorch and a minimal set of key imports.
#
# Usage:
#   bash repro/scripts/00_setup.sh
# =============================================================================
set -e

# --- Determine study root ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$STUDY_ROOT"

ENV_NAME="mpc_vla"
PYTHON_VERSION="3.11"
LOCK_FILE="$STUDY_ROOT/experiments/environment.lock"

echo "================================================================"
echo "  Repro Bundle — Environment Setup"
echo "  Study root: $STUDY_ROOT"
echo "  Env name:   $ENV_NAME"
echo "================================================================"

# --- Check conda ------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

eval "$(conda shell.bash hook)"

# --- Check / create conda environment ---------------------------------------
echo ""
echo "[1/4] Checking conda environment '$ENV_NAME' ..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  [OK] Environment '$ENV_NAME' already exists."
else
    echo "  [INFO] Environment '$ENV_NAME' not found. Creating with Python $PYTHON_VERSION ..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    echo "  [OK] Environment created."
fi

# --- Install PyTorch nightly for cu128 if not already present ---------------
echo ""
echo "[2/4] Checking PyTorch / CUDA wheels ..."
if ! conda run -n "$ENV_NAME" --no-capture-output python -c "import torch" &> /dev/null; then
    echo "  [INFO] PyTorch not found in '$ENV_NAME'. Installing PyTorch nightly (cu128) ..."
    conda run -n "$ENV_NAME" --no-capture-output pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128
    echo "  [OK] PyTorch nightly installed."
else
    echo "  [OK] PyTorch is already installed."
fi

# --- Install pinned dependencies from environment lock ----------------------
echo ""
echo "[3/4] Installing pinned dependencies from:"
echo "  $LOCK_FILE"
conda run -n "$ENV_NAME" --no-capture-output pip install -r "$LOCK_FILE"
echo "  [OK] Pinned dependencies installed / already satisfied."

# --- Verify CUDA / PyTorch and key imports ----------------------------------
echo ""
echo "[4/4] Verifying CUDA / PyTorch and key packages ..."
conda run -n "$ENV_NAME" --no-capture-output python - << 'PYEOF'
import sys
import torch

print(f"  Python:        {sys.version.split()[0]}")
print(f"  PyTorch:       {torch.__version__}")
print(f"  CUDA avail:    {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA version:  {torch.version.cuda}")
    print(f"  GPU count:     {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}:        {props.name}")
    # Quick tensor test on GPU
    x = torch.randn(100, 100, device="cuda")
    y = x @ x
    print(f"  GPU tensor op: OK (shape={tuple(y.shape)})")
else:
    print("  [WARNING] CUDA not available — CPU-only mode.")

# Check other key packages
for pkg in ["numpy", "scipy", "matplotlib", "yaml", "einops", "transformers", "accelerate", "wandb"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {pkg}: {ver}")
    except ImportError:
        print(f"  [ERROR] {pkg} not installed!")
        sys.exit(1)

print("\n  [OK] Environment verification complete.")
PYEOF

# --- Summary ----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Environment setup complete."
echo "================================================================"
echo "  Conda env:  $ENV_NAME"
echo "  Python:     $(conda run -n "$ENV_NAME" python --version 2>&1)"
echo "  PyTorch:    $(conda run -n "$ENV_NAME" python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  CUDA avail: $(conda run -n "$ENV_NAME" python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo ""
echo "  Next step: bash repro/scripts/01_download_data.sh"
echo "  Or run all:  bash repro/scripts/02_run_experiments.sh"
echo "================================================================"
