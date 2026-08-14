#!/usr/bin/env python3
"""Safe Hugging Face Hub upload helper for the MPC vs VLA vs Diffusion study.

Uploads:
  1. SmallVLA quick checkpoint
  2. DDPM Diffusion Policy quick checkpoint
  3. Flow Matching Policy quick checkpoint
  4. Minimal Iterative Policy (MIP) quick checkpoint
  5. MPC expert demonstrations dataset
  6. Gradio demo Space

Token safety:
  - Real uploads require the HF_TOKEN environment variable (write token).
  - If HF_TOKEN is not set and --dry-run is not used, the script exits with a
    clear, actionable message and does not attempt any network call.
  - --dry-run can be run without a token and only reports what would be uploaded.

Usage::

    export HF_TOKEN=hf_...  # required for real uploads
    conda run -n mpc_vla python scripts/upload_hf_artifacts.py --dry-run
    conda run -n mpc_vla python scripts/upload_hf_artifacts.py

The script always writes ``results/hf_artifacts/upload_report.json`` (or the
path passed with ``--report``) with a per-artifact status.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from huggingface_hub import HfApi

    _HFH = True
    _HFH_ERR = None
except Exception as err:  # pragma: no cover
    HfApi = None  # type: ignore
    _HFH = False
    _HFH_ERR = err


STUDY_ROOT = Path(__file__).resolve().parents[1]

MODELS: List[Dict[str, Any]] = [
    {
        "name": "SmallVLA quick checkpoint",
        "repo_id": "Ryukijano/smallvla-mpc-vla-diffusion-quick",
        "manifest_key": "small_vla_pusht",
        "card_dir": STUDY_ROOT / "results" / "hf_artifacts" / "model_cards" / "smallvla-mpc-vla-diffusion-quick",
        "checkpoint": STUDY_ROOT / "results" / "checkpoints" / "small_vla_pusht.pt",
    },
    {
        "name": "DDPM Diffusion Policy quick checkpoint",
        "repo_id": "Ryukijano/ddpm-mpc-vla-diffusion-quick",
        "manifest_key": "ddpm_pusht",
        "card_dir": STUDY_ROOT / "results" / "hf_artifacts" / "model_cards" / "ddpm-mpc-vla-diffusion-quick",
        "checkpoint": STUDY_ROOT / "results" / "checkpoints" / "ddpm_pusht.pt",
    },
    {
        "name": "Flow Matching Policy quick checkpoint",
        "repo_id": "Ryukijano/flow-matching-mpc-vla-diffusion-quick",
        "manifest_key": "flow_matching_pusht",
        "card_dir": STUDY_ROOT / "results" / "hf_artifacts" / "model_cards" / "flow-matching-mpc-vla-diffusion-quick",
        "checkpoint": STUDY_ROOT / "results" / "checkpoints" / "flow_matching_pusht.pt",
    },
    {
        "name": "Minimal Iterative Policy (MIP) quick checkpoint",
        "repo_id": "Ryukijano/mip-mpc-vla-diffusion-quick",
        "manifest_key": "mip_pusht",
        "card_dir": STUDY_ROOT / "results" / "hf_artifacts" / "model_cards" / "mip-mpc-vla-diffusion-quick",
        "checkpoint": STUDY_ROOT / "results" / "checkpoints" / "mip_pusht.npz",
    },
]

DATASET_REPO = "Ryukijano/mpc-expert-demos-quick-test"
DATASET_CARD_DIR = STUDY_ROOT / "results" / "hf_artifacts" / "dataset_cards" / "mpc-expert-demos-quick-test"
DATASET_FILES = [
    "mpc_expert_demos_state.parquet",
    "mpc_expert_demos_state.npz",
    "mpc_expert_demos_images.npz",
]
DATASET_CANDIDATES = [
    STUDY_ROOT / "dist" / "hf_datasets" / "mpc_expert_demos",
    STUDY_ROOT / "data",
    STUDY_ROOT / "data" / "quick_demo",
    STUDY_ROOT / "results" / "checkpoints",
]

SPACE_REPO = "Ryukijano/mpc-vla-diffusion-arena"
SPACE_DIR = STUDY_ROOT / "demo_space"

SPACE_IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git",
    ".pytest_cache",
    "*.ipynb",
    "upload_report.json",
]


def _human_size(num_bytes: int) -> str:
    """Return human-readable file size."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _get_token(dry_run: bool) -> Optional[str]:
    """Validate HF_TOKEN. Dry-runs are allowed without a token."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    if dry_run:
        print("[WARNING] HF_TOKEN is not set. Dry-run will continue without authentication.")
        return None
    print("=" * 72)
    print("ERROR: HF_TOKEN environment variable is required to upload to Hugging Face Hub.")
    print("=" * 72)
    print()
    print("  1. Create a write token at https://huggingface.co/settings/tokens")
    print("  2. Export it in your shell:")
    print("       export HF_TOKEN=hf_...")
    print("  3. Re-run this script without --dry-run.")
    print()
    print("No upload was attempted. Exiting.")
    sys.exit(1)


def _get_api(token: Optional[str]) -> HfApi:
    """Return an authenticated HfApi instance."""
    if not _HFH:
        print("ERROR: huggingface_hub is not installed.")
        print(f"Import error: {_HFH_ERR}")
        print("Install with: conda run -n mpc_vla pip install -U huggingface_hub")
        sys.exit(1)
    return HfApi(token=token)  # type: ignore


def _load_release_manifest() -> Dict[str, Any]:
    path = STUDY_ROOT / "results" / "checkpoints" / "release_manifest.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _extract_model_config(checkpoint: Path, manifest_key: str) -> Optional[Dict[str, Any]]:
    """Extract a small, JSON-serializable config from a saved checkpoint."""
    try:
        if checkpoint.suffix == ".pt":
            import torch

            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            cfg = dict(ckpt.get("config", {})) if isinstance(ckpt, dict) else {}
            # Convert tensors if any sneaked in
            for key, value in list(cfg.items()):
                if isinstance(value, torch.Tensor):
                    cfg[key] = value.detach().cpu().tolist()
            del ckpt
            return cfg
        if checkpoint.suffix == ".npz":
            import numpy as np

            data = np.load(checkpoint)
            keys = ("state_dim", "action_dim", "horizon", "hidden_dim", "noise_std")
            cfg = {}
            for key in keys:
                if key in data:
                    arr = data[key]
                    cfg[key] = arr.item() if arr.ndim == 0 else arr.tolist()
            return cfg
    except Exception as exc:  # pragma: no cover
        print(f"  [WARNING] Could not extract config from {checkpoint}: {exc}")
    return None


def _build_model_metadata(
    repo_id: str, manifest_key: str, checkpoint: Path, config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build a per-model metadata.json from the release manifest and checkpoint."""
    manifest = _load_release_manifest()
    return {
        "repo_id": repo_id,
        "study": "MPC vs VLA vs Diffusion: An Open-Source Study of Robot Control Families",
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_file": checkpoint.name,
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        "checkpoint_size_human": _human_size(checkpoint.stat().st_size) if checkpoint.exists() else None,
        "checkpoint_config": config,
        "release_manifest_entry": manifest.get("checkpoints", {}).get(manifest_key, {}),
    }


def _resolve_dataset_source(override: Optional[Path]) -> Path:
    """Pick the best local directory that contains the dataset files."""
    if override is not None:
        if override.is_dir():
            return override
        print(f"[WARNING] --dataset-dir {override} is not a directory; falling back to auto-detection.")

    def _has_data(path: Path) -> bool:
        return (
            path.is_dir()
            and (path / DATASET_FILES[0]).exists()
            and (path / DATASET_FILES[1]).exists()
        )

    for candidate in DATASET_CANDIDATES:
        if _has_data(candidate):
            return candidate

    # Best-effort default so the report still has a path to show.
    return STUDY_ROOT / "data"


def _model_files(model: Dict[str, Any], config: Optional[Dict[str, Any]], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Describe files that will be / were uploaded for a model."""
    files: Dict[str, Any] = {
        "README.md": str(model["card_dir"] / "README.md"),
        model["checkpoint"].name: str(model["checkpoint"]),
        "config.json": config,
        "metadata.json": metadata,
    }
    return files


def _dry_run_model(model: Dict[str, Any]) -> Dict[str, Any]:
    """Describe what a model upload would do."""
    readme = model["card_dir"] / "README.md"
    checkpoint = model["checkpoint"]
    missing: List[str] = [str(p) for p in (readme, checkpoint) if not p.exists()]
    files = _model_files(model, None, None)
    files["config.json"] = "(extracted from checkpoint during upload)"
    files["metadata.json"] = "(generated from release manifest)"
    if missing:
        return {
            "status": "missing_files",
            "missing_files": missing,
            "files": files,
            "message": f"Missing required files: {', '.join(missing)}",
        }
    return {
        "status": "dry_run",
        "missing_files": [],
        "files": files,
        "message": "Would upload README, checkpoint, config.json and metadata.json",
    }


def _upload_model(api: HfApi, token: str, model: Dict[str, Any]) -> Dict[str, Any]:
    """Upload a single model checkpoint and its card to the Hub."""
    readme = model["card_dir"] / "README.md"
    checkpoint = model["checkpoint"]
    repo_id = model["repo_id"]

    missing: List[str] = [str(p) for p in (readme, checkpoint) if not p.exists()]
    if missing:
        return {
            "status": "missing_files",
            "missing_files": missing,
            "files": _model_files(model, None, None),
            "message": f"Missing required files: {', '.join(missing)}",
        }

    config = _extract_model_config(checkpoint, model["manifest_key"])
    metadata = _build_model_metadata(repo_id, model["manifest_key"], checkpoint, config)
    files = _model_files(model, config, metadata)

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        api.upload_file(
            path_or_fileobj=str(checkpoint),
            path_in_repo=checkpoint.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add checkpoint {checkpoint.name}",
        )
        if config is not None:
            api.upload_file(
                path_or_fileobj=io.BytesIO(json.dumps(config, indent=2).encode("utf-8")),
                path_in_repo="config.json",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add config.json",
            )
        api.upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(metadata, indent=2).encode("utf-8")),
            path_in_repo="metadata.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add metadata.json",
        )

        return {
            "status": "success",
            "missing_files": [],
            "files": files,
            "message": f"Uploaded to https://huggingface.co/{repo_id}",
            "hub_url": f"https://huggingface.co/{repo_id}",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "missing_files": [],
            "files": files,
            "message": f"{type(exc).__name__}: {exc}",
        }


def _dataset_files(source: Path) -> Dict[str, Any]:
    """Describe dataset files."""
    files: Dict[str, Any] = {"README.md": str(DATASET_CARD_DIR / "README.md")}
    for fname in DATASET_FILES:
        files[fname] = str(source / fname)
    manifest = source / "manifest.json"
    if manifest.exists():
        files["manifest.json"] = str(manifest)
    return files


def _dry_run_dataset(source: Path) -> Dict[str, Any]:
    """Describe what a dataset upload would do."""
    readme = DATASET_CARD_DIR / "README.md"
    missing: List[str] = [str(p) for p in (readme,) if not p.exists()]
    files = _dataset_files(source)
    for fname in DATASET_FILES:
        if not (source / fname).exists():
            missing.append(str(source / fname))
    if missing:
        return {
            "status": "missing_files",
            "missing_files": missing,
            "files": files,
            "message": f"Missing required files: {', '.join(missing)}",
        }
    return {
        "status": "dry_run",
        "missing_files": [],
        "files": files,
        "message": f"Would upload dataset files from {source}",
    }


def _upload_dataset(api: HfApi, token: str, source: Path) -> Dict[str, Any]:
    """Upload the dataset README and data files to the Hub."""
    readme = DATASET_CARD_DIR / "README.md"
    missing: List[str] = [str(p) for p in (readme,) if not p.exists()]
    files = _dataset_files(source)
    for fname in DATASET_FILES:
        if not (source / fname).exists():
            missing.append(str(source / fname))
    if missing:
        return {
            "status": "missing_files",
            "missing_files": missing,
            "files": files,
            "message": f"Missing required files: {', '.join(missing)}",
        }

    try:
        api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)

        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        for fname in DATASET_FILES:
            path = source / fname
            if path.exists():
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=fname,
                    repo_id=DATASET_REPO,
                    repo_type="dataset",
                    commit_message=f"Add {fname}",
                )
        manifest = source / "manifest.json"
        if manifest.exists():
            api.upload_file(
                path_or_fileobj=str(manifest),
                path_in_repo="manifest.json",
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message="Add manifest",
            )

        return {
            "status": "success",
            "missing_files": [],
            "files": files,
            "message": f"Uploaded to https://huggingface.co/datasets/{DATASET_REPO}",
            "hub_url": f"https://huggingface.co/datasets/{DATASET_REPO}",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "missing_files": [],
            "files": files,
            "message": f"{type(exc).__name__}: {exc}",
        }


def _dry_run_space() -> Dict[str, Any]:
    """Describe what a Space upload would do."""
    if not SPACE_DIR.is_dir():
        return {
            "status": "missing_files",
            "missing_files": [str(SPACE_DIR)],
            "files": {"folder": str(SPACE_DIR)},
            "message": "Space directory not found",
        }
    files = {"folder": str(SPACE_DIR)}
    return {
        "status": "dry_run",
        "missing_files": [],
        "files": files,
        "message": "Would upload entire demo_space/ folder to a Gradio Space",
    }


def _upload_space(api: HfApi, token: str) -> Dict[str, Any]:
    """Upload the Gradio demo Space folder to the Hub."""
    if not SPACE_DIR.is_dir():
        return {
            "status": "missing_files",
            "missing_files": [str(SPACE_DIR)],
            "files": {"folder": str(SPACE_DIR)},
            "message": "Space directory not found",
        }

    try:
        api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(
            folder_path=str(SPACE_DIR),
            repo_id=SPACE_REPO,
            repo_type="space",
            ignore_patterns=SPACE_IGNORE_PATTERNS,
            commit_message="Upload Gradio Space",
        )
        return {
            "status": "success",
            "missing_files": [],
            "files": {"folder": str(SPACE_DIR)},
            "message": f"Uploaded to https://huggingface.co/spaces/{SPACE_REPO}",
            "hub_url": f"https://huggingface.co/spaces/{SPACE_REPO}",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "missing_files": [],
            "files": {"folder": str(SPACE_DIR)},
            "message": f"{type(exc).__name__}: {exc}",
        }


def _print_dry_run_item(name: str, repo_id: str, source: Any, status: str, message: str) -> None:
    print(f"\n[DRY-RUN] {name} -> {repo_id}")
    print(f"  Status: {status}")
    if isinstance(source, dict):
        for k, v in source.items():
            print(f"  {k}: {v}")
    else:
        print(f"  Source: {source}")
    if message:
        print(f"  Message: {message}")


def _print_upload_item(name: str, repo_id: str, status: str, message: str) -> None:
    print(f"\n[UPLOAD] {name} -> {repo_id}")
    print(f"  Status: {status}")
    if message:
        print(f"  Message: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload MPC vs VLA vs Diffusion artifacts to Hugging Face Hub"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Override the local directory used as the dataset source.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=STUDY_ROOT / "results" / "hf_artifacts" / "upload_report.json",
        help="Path to the upload_report.json output file.",
    )
    args = parser.parse_args()

    token = _get_token(args.dry_run)
    api = None if args.dry_run else _get_api(token)

    dataset_source = _resolve_dataset_source(args.dataset_dir)

    artifacts: List[Dict[str, Any]] = []

    for model in MODELS:
        if args.dry_run:
            print(f"\n{'='*72}\n[DRY-RUN] {model['name']}\n{'='*72}")
            res = _dry_run_model(model)
            _print_dry_run_item(
                model["name"],
                model["repo_id"],
                res["files"],
                res["status"],
                res["message"],
            )
        else:
            print(f"\n{'='*72}\n[UPLOAD] {model['name']}\n{'='*72}")
            res = _upload_model(api, token, model)  # type: ignore
            _print_upload_item(model["name"], model["repo_id"], res["status"], res["message"])

        artifacts.append({
            "artifact": model["name"],
            "repo_id": model["repo_id"],
            "repo_type": "model",
            "status": res["status"],
            "local_source": str(model["card_dir"]),
            "files": res["files"],
            "missing_files": res.get("missing_files", []),
            "message": res["message"],
            "hub_url": res.get("hub_url", ""),
        })

    # Dataset
    if args.dry_run:
        print(f"\n{'='*72}\n[DRY-RUN] Dataset\n{'='*72}")
        res = _dry_run_dataset(dataset_source)
        _print_dry_run_item(
            "MPC expert demonstrations dataset",
            DATASET_REPO,
            res["files"],
            res["status"],
            res["message"],
        )
    else:
        print(f"\n{'='*72}\n[UPLOAD] Dataset\n{'='*72}")
        res = _upload_dataset(api, token, dataset_source)  # type: ignore
        _print_upload_item("MPC expert demonstrations dataset", DATASET_REPO, res["status"], res["message"])

    artifacts.append({
        "artifact": "MPC expert demonstrations dataset",
        "repo_id": DATASET_REPO,
        "repo_type": "dataset",
        "status": res["status"],
        "local_source": str(dataset_source),
        "files": res["files"],
        "missing_files": res.get("missing_files", []),
        "message": res["message"],
        "hub_url": res.get("hub_url", ""),
    })

    # Space
    if args.dry_run:
        print(f"\n{'='*72}\n[DRY-RUN] Gradio Space\n{'='*72}")
        res = _dry_run_space()
        _print_dry_run_item(
            "MPC vs VLA vs Diffusion Arena",
            SPACE_REPO,
            res["files"],
            res["status"],
            res["message"],
        )
    else:
        print(f"\n{'='*72}\n[UPLOAD] Gradio Space\n{'='*72}")
        res = _upload_space(api, token)  # type: ignore
        _print_upload_item("MPC vs VLA vs Diffusion Arena", SPACE_REPO, res["status"], res["message"])

    artifacts.append({
        "artifact": "MPC vs VLA vs Diffusion Arena",
        "repo_id": SPACE_REPO,
        "repo_type": "space",
        "status": res["status"],
        "local_source": str(SPACE_DIR),
        "files": res["files"],
        "missing_files": res.get("missing_files", []),
        "message": res["message"],
        "hub_url": res.get("hub_url", ""),
    })

    # Write report
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": args.dry_run,
        "token_set": token is not None,
        "dataset_source": str(dataset_source),
        "report_path": str(args.report),
        "artifacts": artifacts,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("UPLOAD SUMMARY")
    print("=" * 72)
    for art in artifacts:
        print(f"[{art['status']}] {art['artifact']} -> {art['repo_id']}")
        if art.get("hub_url"):
            print(f"    URL: {art['hub_url']}")
        if art.get("missing_files"):
            print(f"    Missing: {', '.join(map(str, art['missing_files']))}")
        if art.get("message") and art["status"] not in ("success", "dry_run"):
            print(f"    Message: {art['message']}")
    print(f"\nReport written to: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
