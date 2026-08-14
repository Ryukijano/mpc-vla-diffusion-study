# Hugging Face Hub Upload Checklist

This file lists the placeholder `hf upload` commands for releasing each artifact once the actual checkpoints and dataset files are available.

## Prerequisites

- [ ] Install the Hugging Face CLI / `hf` tool:
  ```bash
  pip install "huggingface_hub[cli]"
  # or
  pip install -U hf-transfer
  ```
- [ ] Log in with a write token for the `Ryukijano` account:
  ```bash
  huggingface-cli login
  # or
  hf login
  ```
- [ ] Ensure each `<path>` below is replaced with the actual local directory/file containing the artifact.

## Models

### 1. SmallVLA quick checkpoint

```bash
hf upload Ryukijano/smallvla-mpc-vla-diffusion-quick <path> --repo-type model
```

**Typical upload contents:**
- `small_vla_pusht.pt`
- `config.yaml` (mirroring `configs/vla/small_vla.yaml`)
- `README.md` (this model card)
- `example_inference.py` (optional)

### 2. DDPM Diffusion Policy quick checkpoint

```bash
hf upload Ryukijano/ddpm-mpc-vla-diffusion-quick <path> --repo-type model
```

**Typical upload contents:**
- `ddpm_pusht.pt`
- `config.yaml` / `config.json`
- `README.md` (this model card)
- `example_inference.py` (optional)

### 3. Flow Matching Policy quick checkpoint

```bash
hf upload Ryukijano/flow-matching-mpc-vla-diffusion-quick <path> --repo-type model
```

**Typical upload contents:**
- `flow_matching_pusht.pt`
- `config.yaml` / `config.json`
- `README.md` (this model card)
- `example_inference.py` (optional)

### 4. Minimal Iterative Policy (MIP) quick checkpoint

```bash
hf upload Ryukijano/mip-mpc-vla-diffusion-quick <path> --repo-type model
```

**Typical upload contents:**
- `mip_pusht.npz`
- `config.yaml` / `config.json`
- `README.md` (this model card)
- `example_inference.py` (optional)

## Datasets

### 5. MPC expert demonstrations — quick test set

```bash
hf upload Ryukijano/mpc-expert-demos-quick-test <path> --repo-type dataset
```

**Typical upload contents:**
- `mpc_expert_demos_state.parquet`
- `mpc_expert_demos_state.npz`
- `mpc_expert_demos_images.npz`
- `README.md` (this dataset card)
- `convert_to_lerobot.py` (optional)

## Notes

- The `<path>` placeholder should be a local directory (e.g., `results/checkpoints/` or `dist/hf_datasets/mpc_expert_demos/`) that contains the files listed above.
- Files larger than 10 MB (`.pt` and image `.npz` files) should be tracked with Git LFS. The `hf` / `huggingface-cli` upload tool handles LFS automatically for `.pt`, `.bin`, `.safetensors`, and similar extensions.
- Alternatively, you can use the `huggingface-cli` commands:
  ```bash
  huggingface-cli upload Ryukijano/<repo_id> <local_path> . --repo-type model
  huggingface-cli upload Ryukijano/<repo_id> <local_path> . --repo-type dataset
  ```
- **Do not upload yet** — these are local templates and the commands are intentionally placeholders until final artifacts are produced and verified.
