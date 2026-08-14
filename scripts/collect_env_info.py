#!/usr/bin/env python3
"""Hardware and Environment Provenance Collector for MPC vs VLA vs Diffusion study.

Collects and records hardware configuration, GPU specs, CUDA/PyTorch versions,
operating system details, and git commit status.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_git_info(repo_dir: str) -> Dict[str, Any]:
    """Retrieve git metadata."""
    info: Dict[str, Any] = {}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.PIPE
        ).decode().strip()
        info["commit_hash"] = commit
    except Exception as e:
        info["commit_hash"] = f"unknown ({e})"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, stderr=subprocess.PIPE
        ).decode().strip()
        info["branch"] = branch
    except Exception as e:
        info["branch"] = f"unknown ({e})"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.PIPE
        ).decode().strip()
        info["clean"] = (len(status) == 0)
        info["status_summary"] = status if status else "clean"
    except Exception as e:
        info["clean"] = False
        info["status_summary"] = f"unknown ({e})"

    return info


def collect_env_info(repo_dir: str = REPO_ROOT) -> Dict[str, Any]:
    """Collect full hardware, OS, Python, PyTorch, CUDA, and Git provenance."""
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "git": get_git_info(repo_dir),
    }

    # CPU Information
    try:
        cpu_model = "unknown"
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        if cpu_model == "unknown":
            # For aarch64
            try:
                lscpu = subprocess.check_output(["lscpu"], stderr=subprocess.PIPE).decode()
                for line in lscpu.splitlines():
                    if "Model name:" in line or "Architecture:" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
            except Exception:
                pass
        info["cpu_model"] = cpu_model
    except Exception:
        info["cpu_model"] = platform.processor() or platform.machine()

    # CPU Count
    info["cpu_count_logical"] = os.cpu_count()

    # System RAM
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    mem_kb = int(line.split()[1])
                    info["ram_total_gb"] = round(mem_kb / (1024 * 1024), 2)
                    break
    except Exception:
        info["ram_total_gb"] = "unknown"

    # PyTorch and CUDA information
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["cuda_arch_list"] = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []

            gpus = []
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "index": i,
                    "name": prop.name,
                    "total_memory_gb": round(prop.total_memory / (1024**3), 2),
                    "multi_processor_count": prop.multi_processor_count,
                    "compute_capability": f"{prop.major}.{prop.minor}",
                }
                gpus.append(gpu_info)
            info["gpus"] = gpus
            info["primary_gpu"] = gpus[0]["name"] if gpus else "None"
        else:
            info["primary_gpu"] = "None (CPU only)"
    except ImportError:
        info["pytorch_version"] = "not installed"
        info["cuda_available"] = False
        info["primary_gpu"] = "None"

    # NVIDIA Driver Version via nvidia-smi
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            stderr=subprocess.PIPE,
        ).decode().strip()
        info["nvidia_driver_version"] = driver.splitlines()[0] if driver else "unknown"
    except Exception:
        info["nvidia_driver_version"] = "nvidia-smi unavailable"

    return info


def main():
    parser = argparse.ArgumentParser(description="Collect hardware and environment provenance.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "env_info.json"),
        help="Path to output JSON file (default: results/env_info.json)",
    )
    args = parser.parse_args()

    env_info = collect_env_info(REPO_ROOT)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(env_info, f, indent=2)

    print(f"[Provenance] Saved environment provenance to: {out_path}")
    print(f"  Hostname:     {env_info.get('hostname')}")
    print(f"  CPU:          {env_info.get('cpu_model')} ({env_info.get('cpu_count_logical')} cores)")
    print(f"  RAM:          {env_info.get('ram_total_gb')} GB")
    print(f"  Primary GPU:  {env_info.get('primary_gpu')}")
    print(f"  CUDA Version: {env_info.get('cuda_version')}")
    print(f"  Driver:       {env_info.get('nvidia_driver_version')}")
    print(f"  PyTorch:      {env_info.get('pytorch_version')}")
    print(f"  Git Commit:   {env_info.get('git', {}).get('commit_hash')}")


if __name__ == "__main__":
    main()
