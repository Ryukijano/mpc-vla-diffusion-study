"""VLA baselines package: OpenVLA inference, SmallVLA, TextEncoder, and VLATrainer.

Public API
----------
* :class:`OpenVLAInference` — inference wrapper for the OpenVLA-7B model.
* :class:`SmallVLA` — small compute-matched VLA for fair comparison.
* :class:`TextEncoder` — CLIP / BoW language instruction encoder.
* :class:`VLATrainer` — training loop for SmallVLA with early stopping, LR
  scheduling, and optional wandb logging.

Sub-modules also export their internal building blocks (e.g. ``SmallViT``,
``ActionHead``, ``BagOfWordsEncoder``) for advanced use.
"""

from __future__ import annotations

from .openvla_wrapper import OpenVLAInference
from .small_vla import SmallVLA
from .text_encoder import TextEncoder
from .vla_trainer import VLATrainer

__all__ = [
    "OpenVLAInference",
    "SmallVLA",
    "TextEncoder",
    "VLATrainer",
]

__version__ = "0.1.0"
