# Hugging Face Hub Upload Checklist

This document documents the one-step upload workflow for the **MPC vs VLA vs Diffusion** study artifacts.

All uploads are handled by `scripts/upload_hf_artifacts.py`, which:

- Requires the `HF_TOKEN` environment variable for real uploads (exits with a clear message if not set).
- Supports `--dry-run` to verify paths and file counts before any network call.
- Generates `results/hf_artifacts/upload_report.json` with a per-artifact status.

## Prerequisites

1. Ensure the `mpc_vla` conda environment has `huggingface_hub`:
   ```bash
   conda run -n mpc_vla pip install -U huggingface_hub
   ```

2. Create a Hugging Face **write** token at:
   <https://huggingface.co/settings/tokens>

3. Export the token in your shell **before running the upload script**:
   ```bash
   export HF_TOKEN=hf_...
   ```

   > **Security note:** Never commit the token, never pass it as a command-line argument, and never log it. The helper only reads it from the `HF_TOKEN` environment variable.

## Step 1 — Dry-run to verify all artifacts

```bash
cd /home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study
conda run -n mpc_vla python scripts/upload_hf_artifacts.py --dry-run
```

Expected output: a report for **4 model checkpoints**, **1 dataset**, and **1 Gradio Space**.

## Step 2 — Upload all artifacts

Run the same command without `--dry-run`:

```bash
cd /home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study
conda run -n mpc_vla python scripts/upload_hf_artifacts.py
```

The script will create (or update) the following Hub repositories:

### Models (`repo_type: model`)

| # | Artifact | Hub repo ID | Local checkpoint | Model card |
|---|----------|-------------|------------------|------------|
| 1 | SmallVLA quick checkpoint | `Ryukijano/smallvla-mpc-vla-diffusion-quick` | `results/checkpoints/small_vla_pusht.pt` | `results/hf_artifacts/model_cards/smallvla-mpc-vla-diffusion-quick/README.md` |
| 2 | DDPM Diffusion Policy quick checkpoint | `Ryukijano/ddpm-mpc-vla-diffusion-quick` | `results/checkpoints/ddpm_pusht.pt` | `results/hf_artifacts/model_cards/ddpm-mpc-vla-diffusion-quick/README.md` |
| 3 | Flow Matching Policy quick checkpoint | `Ryukijano/flow-matching-mpc-vla-diffusion-quick` | `results/checkpoints/flow_matching_pusht.pt` | `results/hf_artifacts/model_cards/flow-matching-mpc-vla-diffusion-quick/README.md` |
| 4 | Minimal Iterative Policy (MIP) quick checkpoint | `Ryukijano/mip-mpc-vla-diffusion-quick` | `results/checkpoints/mip_pusht.npz` | `results/hf_artifacts/model_cards/mip-mpc-vla-diffusion-quick/README.md` |

Each model repo receives:

- `README.md` — the model card.
- The checkpoint file (`.pt` or `.npz`).
- `config.json` — extracted automatically from the checkpoint.
- `metadata.json` — provenance from `results/checkpoints/release_manifest.json`.

### Dataset (`repo_type: dataset`)

| # | Artifact | Hub repo ID | Local source | Dataset card |
|---|----------|-------------|--------------|--------------|
| 5 | MPC expert demonstrations quick test | `Ryukijano/mpc-expert-demos-quick-test` | `data/` (preferred) or `dist/hf_datasets/mpc_expert_demos` | `results/hf_artifacts/dataset_cards/mpc-expert-demos-quick-test/README.md` |

Dataset repo receives:

- `README.md` — the dataset card.
- `mpc_expert_demos_state.parquet`
- `mpc_expert_demos_state.npz`
- `mpc_expert_demos_images.npz`
- `manifest.json` (if present in the dataset source directory)

You can override the dataset source directory with:

```bash
conda run -n mpc_vla python scripts/upload_hf_artifacts.py --dataset-dir /path/to/dataset/source
```

### Gradio Space (`repo_type: space`)

| # | Artifact | Hub repo ID | Local folder |
|---|----------|-------------|--------------|
| 6 | MPC vs VLA vs Diffusion Arena | `Ryukijano/mpc-vla-diffusion-arena` | `demo_space/` |

The Space is created with `sdk: gradio` and the entire `demo_space/` folder is uploaded (ignoring `__pycache__`, `*.pyc`, `.git`, `.pytest_cache`, `*.ipynb`, and `upload_report.json`).

## Step 3 — Inspect the upload report

After any run (dry-run or real upload), review:

```bash
cat results/hf_artifacts/upload_report.json
```

The report contains one entry per artifact with `status`, `repo_id`, `files`, `missing_files`, and `hub_url` (when available).

## Optional: Custom report path

```bash
conda run -n mpc_vla python scripts/upload_hf_artifacts.py --report /path/to/report.json
```

## Notes

- The script will not upload without `HF_TOKEN`; it exits before any network call.
- Large `.pt`/`.npz` files are automatically tracked by Git LFS by the `huggingface_hub` upload helpers.
- If a repository already exists on the Hub, `create_repo(..., exist_ok=True)` reuses it and the script uploads a new commit.
- Keep `HF_TOKEN` out of logs, notebooks, and version control.
