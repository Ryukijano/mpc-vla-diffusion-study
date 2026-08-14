#!/usr/bin/env bash
# verify_hf_artifacts.sh
#
# Local sanity check that all Hugging Face Hub packaging artifacts exist and
# the demo Space app is syntactically valid.
#
# Exit codes:
#   0  all checks passed
#   1  one or more checks failed

set -euo pipefail

STUDY_ROOT="/home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study"
CHECKPOINT_DIR="${STUDY_ROOT}/results/checkpoints"
DATASET_DIR="${STUDY_ROOT}/results/hf_datasets"
DEMO_APP="${STUDY_ROOT}/demo_space/app.py"

ERRORS=0

echo "=== HF Artifact Verification ==="
echo "Study root: ${STUDY_ROOT}"

# 1. Check results/checkpoints for the four expected trained checkpoints.
echo "[1/3] Checking checkpoints in ${CHECKPOINT_DIR} ..."
EXPECTED_CKPTS=(
  "small_vla_pusht.pt"
  "ddpm_pusht.pt"
  "flow_matching_pusht.pt"
  "mip_pusht.npz"
)
for ckpt in "${EXPECTED_CKPTS[@]}"; do
  path="${CHECKPOINT_DIR}/${ckpt}"
  if [[ -f "${path}" ]]; then
    echo "  OK: ${ckpt} ($(stat -c%s "${path}" | numfmt --to=iec))"
  else
    echo "  MISSING: ${ckpt}"
    ERRORS=$((ERRORS + 1))
  fi
done

# 2. Check dataset package output exists.
echo "[2/3] Checking dataset package in ${DATASET_DIR} ..."
if [[ -d "${DATASET_DIR}/mpc_expert_demos" ]]; then
  if [[ -f "${DATASET_DIR}/mpc_expert_demos/README.md" ]]; then
    echo "  OK: dataset README exists"
  else
    echo "  MISSING: dataset README.md"
    ERRORS=$((ERRORS + 1))
  fi
  if [[ -f "${DATASET_DIR}/mpc_expert_demos/mpc_expert_demos_state.npz" ]]; then
    echo "  OK: state .npz exists"
  else
    echo "  MISSING: state .npz"
    ERRORS=$((ERRORS + 1))
  fi
  if [[ -f "${DATASET_DIR}/mpc_expert_demos/mpc_expert_demos_state.parquet" ]]; then
    echo "  OK: state .parquet exists"
  else
    echo "  MISSING: state .parquet"
    ERRORS=$((ERRORS + 1))
  fi
else
  echo "  MISSING: ${DATASET_DIR}/mpc_expert_demos"
  ERRORS=$((ERRORS + 1))
fi

# 3. Check demo_space/app.py exists and compiles.
echo "[3/3] Checking demo_space app ..."
if [[ -f "${DEMO_APP}" ]]; then
  echo "  OK: ${DEMO_APP} exists"
  if python -m py_compile "${DEMO_APP}"; then
    echo "  OK: ${DEMO_APP} compiles"
  else
    echo "  ERROR: ${DEMO_APP} failed py_compile"
    ERRORS=$((ERRORS + 1))
  fi
else
  echo "  MISSING: ${DEMO_APP}"
  ERRORS=$((ERRORS + 1))
fi

echo "=================================="
if [[ ${ERRORS} -eq 0 ]]; then
  echo "All checks passed."
  exit 0
else
  echo "${ERRORS} check(s) failed."
  exit 1
fi
