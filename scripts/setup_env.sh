#!/bin/bash
# =============================================================================
# Environment setup for MPC vs VLA vs Diffusion study
# =============================================================================
# Target: NVIDIA DGX Spark (Grace Blackwell GB10, aarch64)
# Creates/verifies the `mpc_vla` conda environment with PyTorch nightly
# for CUDA 12.8 and all study dependencies.
#
# Usage:
#   bash scripts/setup_env.sh
#   # or
#   chmod +x scripts/setup_env.sh && ./scripts/setup_env.sh
# =============================================================================
set -e

echo "================================================================"
echo "  MPC vs VLA vs Diffusion -- Environment Setup"
echo "  Target: NVIDIA DGX Spark (GB10 Grace Blackwell)"
echo "================================================================"

# --- Determine study root ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$STUDY_ROOT"

echo "  Study root: $STUDY_ROOT"

# --- Check conda ------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Please install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

ENV_NAME="mpc_vla"
PYTHON_VERSION="3.12"

# --- Create or verify conda environment -------------------------------------
echo ""
echo "[1/5] Checking conda environment: $ENV_NAME"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  [OK] Environment '$ENV_NAME' already exists."
else
    echo "  Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    echo "  [OK] Environment created."
fi

conda activate "$ENV_NAME"
echo "  Active env: $CONDA_DEFAULT_ENV"
echo "  Python:     $(python --version)"

# --- Install PyTorch nightly for cu128 (aarch64) ----------------------------
echo ""
echo "[2/5] Installing PyTorch nightly (cu128)..."

# Detect architecture
ARCH=$(uname -m)
echo "  Architecture: $ARCH"

if [[ "$ARCH" == "aarch64" ]]; then
    # GB10 / Grace Blackwell is aarch64
    echo "  Installing aarch64 PyTorch nightly with cu128..."
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128
else
    # Fallback for x86_64 dev machines
    echo "  Installing x86_64 PyTorch nightly with cu128..."
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128
fi

echo "  [OK] PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"

# --- Install other dependencies ---------------------------------------------
echo ""
echo "[3/5] Installing study dependencies..."

pip install \
    "numpy>=2.0" \
    "scipy>=1.13" \
    "matplotlib>=3.8" \
    "pyyaml>=6.0" \
    "pytest>=8.0" \
    "transformers>=4.40" \
    "accelerate>=0.30" \
    "pillow>=10.0" \
    "einops>=0.7" \
    "wandb>=0.16"

echo "  [OK] Dependencies installed."

# --- Set up PYTHONPATH ------------------------------------------------------
echo ""
echo "[4/5] Setting up PYTHONPATH..."

# Create a conda activation script to set PYTHONPATH automatically
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat > "$ACTIVATE_DIR/env_vars.sh" << EOF
#!/bin/bash
export PYTHONPATH="$STUDY_ROOT:$STUDY_ROOT/mpc_baselines_repo:$STUDY_ROOT/mpc_baselines_repo/src:$STUDY_ROOT/diffusion_baselines:$STUDY_ROOT/benchmarks:$STUDY_ROOT/vla_baselines:\$PYTHONPATH"
EOF
chmod +x "$ACTIVATE_DIR/env_vars.sh"

# Also set for this session
export PYTHONPATH="$STUDY_ROOT:$STUDY_ROOT/mpc_baselines_repo:$STUDY_ROOT/mpc_baselines_repo/src:$STUDY_ROOT/diffusion_baselines:$STUDY_ROOT/benchmarks:$STUDY_ROOT/vla_baselines:$PYTHONPATH"

echo "  PYTHONPATH set to include study modules."
echo "  Activation script: $ACTIVATE_DIR/env_vars.sh"

# --- Verify PyTorch + CUDA on GB10 ------------------------------------------
echo ""
echo "[5/5] Verifying PyTorch + CUDA..."

python - << 'PYEOF'
import sys
print(f"  Python: {sys.version}")

try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version:    {torch.version.cuda}")
        print(f"  GPU count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Total memory:  {props.total_mem / 1e9:.1f} GB")
            print(f"    Compute caps:  {props.major}.{props.minor}")
        # Quick tensor test on GPU
        x = torch.randn(100, 100, device="cuda")
        y = x @ x
        print(f"  GPU tensor test: OK (shape={tuple(y.shape)})")
    else:
        print("  [WARNING] CUDA not available -- CPU-only mode")
except ImportError:
    print("  [ERROR] PyTorch not importable!")
    sys.exit(1)

# Check other key packages
for pkg in ["numpy", "scipy", "matplotlib", "yaml"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {pkg}: {ver}")
    except ImportError:
        print(f"  [WARNING] {pkg} not installed")

print("\n  [OK] Verification complete.")
PYEOF

# --- Print environment info -------------------------------------------------
echo ""
echo "================================================================"
echo "  Environment Setup Complete!"
echo "================================================================"
echo "  Conda env:     $ENV_NAME"
echo "  Python:        $(python --version 2>&1)"
echo "  PyTorch:       $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "  CUDA avail:    $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo ""
echo "  To activate:   conda activate $ENV_NAME"
echo "  Study root:    $STUDY_ROOT"
echo ""
echo "  Next steps:"
echo "    1. Run quick test:    ./scripts/run_quick_test.sh"
echo "    2. Run full exp:      ./scripts/run_on_spark.sh"
echo "    3. Run ablation:      conda run -n $ENV_NAME python run_ablation.py"
echo "    4. Generate report:   conda run -n $ENV_NAME python generate_report.py"
echo "================================================================"
