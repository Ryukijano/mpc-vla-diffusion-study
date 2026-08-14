# Hugging Face Hub Artifact Packaging Plan

**Study:** `mpc-vla-diffusion-study`  
**GitHub:** `github.com/Ryukijano/mpc-vla-diffusion-study`  
**Date:** 12 Aug 2026  
**Status:** planning only — **do not upload yet**

This document describes the set of Hugging Face Hub artifacts (models, datasets, Spaces, Collection) that can be prepared and linked from the study's HF blog *before* the full experimental matrix is complete.  It is based on the current repository state, the existing `docs/blogging/hf_blog_research.md` landscape analysis, and a set of quick size checks run on the DGX Spark / `mpc_vla` conda environment.

---

## 1. TL;DR

Even before the final large-scale experiments, the study can publish:

1. A **SmallVLA** PyTorch checkpoint trained on quick-test image demonstrations.
2. A **tiny DDPM DiffusionPolicy** PyTorch checkpoint trained on quick-test state demonstrations.
3. A **MPC expert demonstration dataset** as `.npz` (and optionally LeRobot/Parquet).
4. A **HF Space** that renders the generated comparison plots.
5. A second **HF Space** that lets a visitor upload a benchmark config and run a one-click controller comparison.
6. A **HF Collection** tying the GitHub repo, models, dataset and Spaces together.

If an arXiv paper is written later, these artifacts can be linked to a HF paper page.  Until then the blog post itself serves as the canonical public write-up, anchored by the GitHub repo and the Hub Collection.

---

## 2. Artifacts we can publish now

| Artifact | Hub type | Suggested repo ID | Format | Measured / estimated size | Status |
|---|---|---|---|---|---|
| SmallVLA quick checkpoint | Model | `Ryukijano/smallvla-mpc-vla-diffusion-quick` | PyTorch `.pt` | ~340 MB | not yet created |
| Tiny DDPM policy quick checkpoint | Model | `Ryukijano/diffusion-policy-mpc-vla-diffusion-quick` | PyTorch `.pt` | ~5.6 MB | not yet created |
| MPC expert demos (quick test) | Dataset | `Ryukijano/mpc-expert-demos-quick-test` | `.npz` (LeRobot optional) | 3 KB – 35 MB (state vs. 96×96 images) | not yet created |
| Quick-test comparison plots | Space | `Ryukijano/mpc-vla-diffusion-plot-gallery` | Gradio static gallery | <1 MB code + ~540 KB PNGs | can be built from existing files |
| Interactive controller arena | Space | `Ryukijano/mpc-vla-diffusion-arena` | Gradio + backend runner | <1 MB code | not yet created |
| Project collection | Collection | `Ryukijano/mpc-vla-diffusion-study` | Collection | n/a | not yet created |

---

## 3. Artifact packaging details

### 3.1 SmallVLA quick checkpoint

**What to upload**
- One `small_vla_quick.pt` model state file.
- One `config.yaml` mirroring `configs/vla/small_vla.yaml` (small preset: `img_size=64`, `hidden_dim=32`, `num_layers=2`, `text_backend=bow`).
- `README.md` model card.
- Optional `example_inference.py` snippet.

**Format**
- PyTorch `torch.save()` dict containing `model_state_dict`, `model_config`, `training_config`, and `history`.
- The file can be saved through `vla_baselines/vla_trainer.py::VLATrainer.save_checkpoint()` (line 501-532).

**Estimated size**
- Quick state-dict: ~327 MB on disk.
- Full trainer checkpoint (state + optimiser + metadata): ~343 MB.
- The vision encoder is a full `SmallViT` (ViT-Base, 12 layers, 768 dim, ~86 M params), so the quick preset does not meaningfully shrink the checkpoint; the parameter count is dominated by the vision backbone.

> Measured on GB10 (`mpc_vla`): `vla_small_state.pt` = 342,811,799 bytes; `vla_small_full.pt` = 342,811,701 bytes. Parameter count = 85,688,094.

**Steps to create**
1. Collect image demonstrations using `DemonstrationCollector(image_mode=True)` on `benchmarks/reaching_env.py` (or `pusht_env.py`) with the `CollisionFreeMPC` expert (see `benchmarks/demonstration_collector.py`, line 157-207).
2. Convert the `.npz` output into a list of dicts `{"image": (H,W,3) uint8, "instruction": env.get_language_instruction(), "action": (horizon, action_dim)}` consumable by `SmallVLA.train()` in `vla_baselines/small_vla.py`.
3. Train with `VLATrainer` using `configs/vla/small_vla.yaml` (small preset) for a few epochs.
4. Save: `trainer.save_checkpoint("results/checkpoints/small_vla/quick/small_vla_quick.pt")`.
5. Verify: `vla.predict_action(image, "Reach the green target in the plane while avoiding obstacles.")` returns a finite action sequence.

**README / model card frontmatter**
```yaml
---
title: "SmallVLA quick checkpoint — MPC vs VLA vs Diffusion study"
language: en
license: mit
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
  - robotics
  - robot-learning
  - imitation-learning
  - vla
  - vision-language-action
  - smallvla
  - diffusion-policy
  - mpc
  - reaching-task
datasets:
  - Ryukijano/mpc-expert-demos-quick-test
---
```

---

### 3.2 Tiny DDPM DiffusionPolicy quick checkpoint

**What to upload**
- One `diffusion_policy_quick.pt` model state file.
- A `config.json` / `config.yaml` with `action_dim`, `horizon`, `obs_dim`, `num_diffusion_steps`, `hidden_dim`, `num_layers`.
- `README.md` model card and an `example_inference.py` snippet.

**Format**
- PyTorch `.pt` containing `state_dict` and `config`.  The `DiffusionPolicy` class in `diffusion_baselines/ddpm_policy.py` does not currently wrap a `save()` method, so use `torch.save({"state_dict": policy.net.state_dict(), "config": {...}}, path)`.

**Estimated size**
- Quick config (`hidden_dim=16`, `num_layers=4`, `num_diffusion_steps=4`): ~5.6 MB.
- Full default config (`hidden_dim=256`, `num_layers=4`, `num_diffusion_steps=100`): ~641 MB.

> Measured on GB10: `ddpm_quick.pt` (hidden_dim=16, num_layers=4, 4 diffusion steps) = 5,644,181 bytes; `ddpm.pt` (hidden_dim=256, num_layers=4, 100 steps) = 641,478,489 bytes. Quick net = 1,390,930 params; full net = 160,352,770 params.

**Steps to create**
1. Run the quick demo collection: `python run_experiments.py --quick` (or `run_ablation.py --benchmark reaching --seeds 0 --episodes 5 --epochs 10 --num-demos 10`) — this already calls `collect_demonstrations()` in `run_experiments.py` (line 342-375).
2. Train a `DiffusionPolicy` from `diffusion_baselines/ddpm_policy.py` with quick hyperparameters.
3. Save the `ConditionalUnet1D` backbone:
   ```python
   torch.save({
       "state_dict": policy.net.state_dict(),
       "config": {
           "action_dim": 2,
           "horizon": 15,
           "obs_dim": 4,
           "num_diffusion_steps": 4,
           "hidden_dim": 16,
           "num_layers": 4,
       },
   }, "diffusion_policy_quick.pt")
   ```
4. Verify: `policy.sample(state, num_samples=1)` returns shape `(1, 15, 2)`.

**README / model card frontmatter**
```yaml
---
title: "Tiny DDPM DiffusionPolicy quick checkpoint — MPC vs VLA vs Diffusion study"
language: en
license: mit
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
  - robotics
  - robot-learning
  - imitation-learning
  - diffusion-policy
  - ddpm
  - action-generation
  - reaching-task
datasets:
  - Ryukijano/mpc-expert-demos-quick-test
---
```

---

### 3.3 MPC expert demonstration dataset

**What to upload**
- `mpc_expert_demos_quick_test.npz` (state-based observations and actions).
- `mpc_expert_demos_quick_test_images.npz` (optional image observations, for VLA training).
- `README.md` dataset card.
- Optional `convert_to_lerobot.py` to produce a LeRobot-compatible `data/` folder.

**Format**
- Primary: NumPy `.npz` archive with keys `observations`, `actions`, `next_observations`, `rewards`, `dones` (as produced by `DemonstrationCollector.save()` in `benchmarks/demonstration_collector.py`, line 297-312).
- Optional: LeRobot dataset (Parquet files + video / image frames), generated after upload or as a second dataset repo.

**Estimated size**
- State-only, 10 demos, horizon 15: ~3 KB.
- State-only, 10 episodes × 60 steps (600 transitions): ~28 KB.
- Image-based, 10 episodes × 60 steps, 64×64 RGB: ~15 MB.
- Image-based, 10 episodes × 60 steps, 96×96 RGB: ~33 MB.
- Full medium run (100 demos, 128×128, 60 steps): ~600 MB.

> Measured on GB10: `reaching_actual_state.npz` (600 transitions) = 28,278 bytes; `reaching_actual_64x64.npz` = 14,754,678 bytes; `reaching_actual_96x96.npz` = 33,186,678 bytes. Use `np.savez_compressed` if smaller on-disk size is required.

**Steps to create**
1. State dataset: run `run_experiments.py --quick` and persist the output of `collect_demonstrations()` with `DemonstrationCollector.save()`:
   ```python
   collector = DemonstrationCollector(image_mode=False)
   ds = collector.from_mpc(env, mpc, n_episodes=10, max_steps=60, seeds=range(10))
   collector.save("data/mpc_expert_demos_quick_test.npz")
   ```
2. Image dataset: repeat with `image_mode=True`:
   ```python
   collector = DemonstrationCollector(image_mode=True)
   ds = collector.from_mpc(env, mpc, n_episodes=10, max_steps=60, seeds=range(10))
   collector.save("data/mpc_expert_demos_quick_test_images.npz")
   ```
3. Optional LeRobot conversion: install `lerobot`, load each `.npz`, and write to `data/chunk-000/parquet/0000.parquet` with a `lerobot/README.md`.

**README / dataset card frontmatter**
```yaml
---
title: "MPC expert demonstrations — quick test set"
language: en
license: mit
tags:
  - robotics
  - robot-learning
  - imitation-learning
  - demonstrations
  - reaching-task
  - npz
  - lerobot
configs:
  - reaching
task_categories:
  - robot-learning
size_categories:
  - n<1K
---
```

---

### 3.4 Space: quick-test comparison plot gallery

**What to upload**
- `app.py` (Gradio static image gallery).
- `requirements.txt` (`gradio>=4.0`, `matplotlib`, `pandas`).
- `README.md` Space card.
- All PNG files from `results/quick_test/report/figures/` and `results/quick_test/ablation/figures/`.

**Format**
- HF Space, SDK `gradio`.
- Static gallery with short captions and a download link for the `master_comparison.csv`.

**Estimated size**
- App code: <50 KB.
- Report figures: 320 KB (`results/quick_test/report/figures/`).
- Ablation figures: 212 KB (`results/quick_test/ablation/figures/`).
- CSV tables: <20 KB.
- Total: ~600 KB of assets.

**Steps to create**
1. Copy the existing quick-test outputs:
   - `results/quick_test/report/figures/*.png`
   - `results/quick_test/ablation/figures/*.png`
   - `results/quick_test/tables/master_comparison.csv`
2. Create `app.py` with a `gr.Gallery`, `gr.DataFrame`, and a `gr.DownloadButton`.
3. Push to `Ryukijano/mpc-vla-diffusion-plot-gallery`.

**README / Space card frontmatter**
```yaml
---
title: "MPC vs VLA vs Diffusion — quick test plot gallery"
emoji: "/"
colorFrom: "purple"
colorTo: "blue"
sdk: gradio
sdk_version: "4.x"
app_file: app.py
pinned: false
license: mit
---
```

---

### 3.5 Space: interactive controller arena

**What to upload**
- `app.py` (Gradio front-end + a lightweight backend that runs one of the study controllers on a benchmark).
- `requirements.txt` (study dependencies: `torch`, `numpy`, `scipy`, `pyyaml`, `matplotlib`, etc.).
- `README.md` with usage and limitations.

**Format**
- HF Space, SDK `gradio`.
- Two tabs:
  1. **Config upload**: visitor uploads a YAML benchmark/config, picks controller families, and clicks "Run quick comparison".  The Space runs a cached / CPU version of `run_experiments.py --quick` and returns the `master_comparison.csv` and generated plots.
  2. **Model zoo**: drop-down to load a published checkpoint (SmallVLA / DDPM / MIP) and run a single rollout in the browser.

**Estimated size**
- App code: <1 MB.
- Runtime model downloads: SmallVLA ~340 MB (if loading from Hub); DDPM quick ~5.6 MB.

**Steps to create**
1. Wrap `run_experiments.py` into a callable function `run_quick_from_config(yaml_path, benchmark, controllers)`.
2. Add guards so the Space never runs heavy `num_episodes > 10` or `hidden_dim > 32`.
3. Use `huggingface_hub.hf_hub_download()` to pull the published checkpoints for the "Model zoo" tab.
4. Build Gradio UI: `gr.File` (YAML), `gr.Dropdown` (benchmark, controller family), `gr.Button`, `gr.DataFrame`, `gr.Image` (plot outputs).
5. Push to `Ryukijano/mpc-vla-diffusion-arena`.

> **Caveat:** running MPC solvers inside a free CPU Space may time out. The simplest robust version pre-computes quick-test results and *replays* them for the uploaded config, while heavier re-runs are queued behind a paid GPU Space or left as a local-only option.

**README / Space card frontmatter**
```yaml
---
title: "MPC vs VLA vs Diffusion — controller arena"
emoji: "🤖"
colorFrom: "red"
colorTo: "orange"
sdk: gradio
sdk_version: "4.x"
app_file: app.py
pinned: false
license: mit
---
```

---

## 4. Hugging Face Collection

**Suggested Collection ID:** `Ryukijano/mpc-vla-diffusion-study`

**Title:**  
`MPC vs VLA vs Diffusion: an open comparison of robot control families`

**Description:**  
This collection groups the code, checkpoints, demonstration data and interactive demos for the study "MPC vs VLA vs Diffusion: Do generative control policies actually beat classical MPC and VLA-style policies, and if so, why?"  It includes a small compute-matched VLA, a tiny DiffusionPolicy, expert demonstrations from MPC, and Gradio Spaces for comparing controllers on simple reaching and pushing tasks.

**Items to group in the Collection**
1. GitHub repo `github.com/Ryukijano/mpc-vla-diffusion-study`
2. Model `Ryukijano/smallvla-mpc-vla-diffusion-quick`
3. Model `Ryukijano/diffusion-policy-mpc-vla-diffusion-quick`
4. Dataset `Ryukijano/mpc-expert-demos-quick-test`
5. Space `Ryukijano/mpc-vla-diffusion-plot-gallery`
6. Space `Ryukijano/mpc-vla-diffusion-arena`
7. Later additions: full-size SmallVLA, flow-matching / MIP checkpoints, real-robot demo data, arXiv paper page.

---

## 5. Paper page vs blog

### If the work is later submitted to arXiv
- Publish the preprint on arXiv and add the arXiv ID to each model / dataset card.
- Create a HF paper page at `huggingface.co/papers/<arxiv_id>` (or let HF auto-create it from the arXiv abstract).
- On the paper page, add the study's model, dataset and Space repos as linked artifacts.
- In each artifact card, add a `citation` block that points to the arXiv BibTeX and, if applicable, a `paper:` link in the frontmatter.

### With no arXiv paper (current state)
- Use an HF **community blog post** at `huggingface.co/new-blog` as the primary public write-up.
- The blog will embed links to the GitHub repo, the HF Collection, and each artifact.
- Use the README of the GitHub repo as a permanent landing page with a "Cite this work" section (arXiv placeholder or a `CITATION.cff` file).
- When the paper is ready, update the blog with an edit linking the new arXiv / HF paper page.

### Recommended blog → Hub cross-links
| In the blog | Link target |
|---|---|
| "Try the demo" | `Ryukijano/mpc-vla-diffusion-arena` |
| "View the plots" | `Ryukijano/mpc-vla-diffusion-plot-gallery` |
| "Download the SmallVLA checkpoint" | `Ryukijano/smallvla-mpc-vla-diffusion-quick` |
| "Download the diffusion policy checkpoint" | `Ryukijano/diffusion-policy-mpc-vla-diffusion-quick` |
| "Get the expert demonstrations" | `Ryukijano/mpc-expert-demos-quick-test` |
| "Browse all artifacts" | `Ryukijano/mpc-vla-diffusion-study` (Collection) |
| "Full code and protocols" | `github.com/Ryukijano/mpc-vla-diffusion-study` |

---

## 6. Action checklist (pre-launch)

### Repository hygiene
- [ ] Add an `LICENSE` file to the study root (MIT is consistent with `mpc_baselines_repo/LICENSE` and `reference_diffusion_policy/LICENSE`).
- [ ] Add a `CITATION.cff` or a "Cite this work" section in `README.md`.
- [ ] Add a `docs/blogging/assets/` folder with the post thumbnail (1300×650 px) and any body figures.

### HF Hub setup
- [ ] Ensure the HF user/org `Ryukijano` exists and has a write token.
- [ ] Install `huggingface_hub` CLI: `pip install "huggingface_hub[cli]"`.
- [ ] Log in: `huggingface-cli login`.

### Generate artifacts
- [ ] Create `data/mpc_expert_demos_quick_test.npz` (state-based) using `DemonstrationCollector`.
- [ ] Create `data/mpc_expert_demos_quick_test_images.npz` (image-based) for VLA training.
- [ ] Train and save `results/checkpoints/small_vla/small_vla_quick.pt`.
- [ ] Train and save `results/checkpoints/diffusion/diffusion_policy_quick.pt`.
- [ ] Generate a `config.json` / `config.yaml` alongside each checkpoint.

### Upload to Hub
- [ ] Create model repo `Ryukijano/smallvla-mpc-vla-diffusion-quick`.
- [ ] Create model repo `Ryukijano/diffusion-policy-mpc-vla-diffusion-quick`.
- [ ] Create dataset repo `Ryukijano/mpc-expert-demos-quick-test`.
- [ ] Create Space `Ryukijano/mpc-vla-diffusion-plot-gallery`.
- [ ] Create Space `Ryukijano/mpc-vla-diffusion-arena`.
- [ ] Upload files via `huggingface-cli upload` or `huggingface_hub` Python API.
- [ ] Confirm LFS is used for files >10 MB (`.pt`, image `.npz`).

### Cards and metadata
- [ ] Write `README.md` (model / dataset / Space cards) with the frontmatter blocks above.
- [ ] Add `example_inference.py` to each model repo.
- [ ] Add `convert_to_lerobot.py` to the dataset repo.

### Collection and blog
- [ ] Create the Collection `Ryukijano/mpc-vla-diffusion-study` and add all items.
- [ ] Draft the HF community blog post (`huggingface.co/new-blog`) using `docs/blogging/hf_blog_research.md` as style guide.
- [ ] Cross-link every artifact in the blog and in the GitHub repo `README.md`.

### Verification
- [ ] Test `SmallVLA` inference from the uploaded checkpoint in a fresh environment.
- [ ] Test `DiffusionPolicy.sample()` from the uploaded checkpoint.
- [ ] Test the dataset `.npz` can be loaded by `DemonstrationCollector.load()`.
- [ ] Smoke-test both Spaces on a CPU runtime before announcing the blog.

---

## 7. Appendix: size measurement notes

All on-disk measurements below were taken from the DGX Spark (`mpc_vla` conda env, Python 3.11, PyTorch 2.12.0.dev20260408+cu128) using synthetic parameters or rendered images from `benchmarks/reaching_env.py`:

| Item | Path / file | Size |
|---|---|---|
| SmallVLA state dict (small preset) | `/tmp/hf_size_check/vla_small_state.pt` | 342,811,799 B (~327 MB) |
| SmallVLA full trainer checkpoint | `/tmp/hf_size_check/vla_small_full.pt` | 342,811,701 B (~327 MB) |
| DDPM quick net (`hidden_dim=16`) | `/tmp/hf_size_check/ddpm_quick.pt` | 5,644,181 B (~5.6 MB) |
| DDPM full net (`hidden_dim=256`) | `/tmp/hf_size_check/ddpm.pt` | 641,478,489 B (~612 MB) |
| 10 state-only episodes (600 transitions) | `/tmp/hf_size_check/reaching_actual_state.npz` | 28,278 B (~28 KB) |
| 10 image episodes, 64×64 | `/tmp/hf_size_check/reaching_actual_64x64.npz` | 14,754,678 B (~14.1 MB) |
| 10 image episodes, 96×96 | `/tmp/hf_size_check/reaching_actual_96x96.npz` | 33,186,678 B (~31.7 MB) |
| Quick-test report figures | `results/quick_test/report/figures/` | 320 KB |
| Quick-test ablation figures | `results/quick_test/ablation/figures/` | 212 KB |

These numbers are the basis for the size estimates in the artifact table.

---

## 8. Local packaging verification (performed)

The following dry-runs were executed in the `mpc_vla` conda environment on the DGX Spark to prove the HF packaging workflow and card generation work without uploading anything.

### 8.1 Model packaging dry-run

| Step | Command | Output directory / key files | Status |
|---|---|---|---|
| Single placeholder model card | `conda run -n mpc_vla python scripts/package_hf_models.py --checkpoint /tmp/hf_model_test/dummy.pt --repo-id Ryukijano/test-model --dry-run` | `dist/hf_models/test-model/README.md`, `dist/hf_models/test-model/config.yaml`, `dist/hf_models/test-model/dummy.pt` | PASS |
| All four quick baselines | `conda run -n mpc_vla python scripts/package_hf_models.py --output-dir dist/hf_models` | `dist/hf_models/small_vla/`, `dist/hf_models/ddpm/`, `dist/hf_models/flow_matching/`, `dist/hf_models/mip/` | PASS |

Each package contains a `README.md` model card, `config.yaml`, `example_inference.py`, and a verified `.pt`/`.npz` checkpoint.

### 8.2 Dataset packaging dry-run

| Step | Command | Output directory / key files | Status |
|---|---|---|---|
| Quick MPC demo dataset | `conda run -n mpc_vla python scripts/package_hf_dataset.py --output-dir results/hf_datasets/mpc_expert_demos --n-episodes 2 --max-steps 10` | `results/hf_datasets/mpc_expert_demos/README.md`, `mpc_expert_demos_state.npz`, `mpc_expert_demos_state.parquet`, `mpc_expert_demos_images.npz` | PASS |

### 8.3 Trained release checkpoints

`scripts/train_and_export_checkpoints.py` produced and verified the four trained PushT checkpoints:

| Checkpoint | File | Size | Params | Status |
|---|---|---|---|---|
| SmallVLA | `results/checkpoints/small_vla_pusht.pt` | 571 MB | 149,436,944 | PASS |
| DDPM Diffusion Policy | `results/checkpoints/ddpm_pusht.pt` | 43 MB | 11,058,050 | PASS |
| Flow Matching Policy | `results/checkpoints/flow_matching_pusht.pt` | 43 MB | 11,058,050 | PASS |
| Minimal Iterative Policy (MIP) | `results/checkpoints/mip_pusht.npz` | 59 KB | 7,072 | PASS |

The verification results are recorded in `results/checkpoints/release_manifest.json`.

### 8.4 Artifact card directories

Generated model and dataset cards are also copied to `results/hf_artifacts/` for local inspection:

- `results/hf_artifacts/MODEL_CARDS/`
  - `small_vla_README.md`
  - `ddpm_README.md`
  - `flow_matching_README.md`
  - `mip_README.md`
  - `test_model_dry_run_README.md`
- `results/hf_artifacts/DATASET_CARDS/`
  - `mpc_expert_demos_README.md`

### 8.5 Verification script

Run the bundled shell gate to re-check the artifacts and the Gradio Space:

```bash
bash /home/aimsgroupuol/AIMSgeneral/Gyanateet/mpc_vla_diffusion_study/scripts/verify_hf_artifacts.sh
```

This script:
1. Confirms the four checkpoints exist under `results/checkpoints/`.
2. Confirms the dataset package exists under `results/hf_datasets/mpc_expert_demos`.
3. Confirms `demo_space/app.py` exists and compiles with `python -m py_compile`.

It exits `0` when all checks pass and `1` otherwise.
