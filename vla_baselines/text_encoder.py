"""Text / language instruction encoder for VLA baselines.

Provides a :class:`TextEncoder` that produces a fixed-dimensional embedding
vector from a natural-language instruction string.  The encoder first tries to
use a HuggingFace CLIP text encoder (``clip-vit-base-patch32``); if CLIP or the
transformers library is unavailable it transparently falls back to a lightweight
bag-of-words + MLP encoder that requires no external downloads and is fully
trainable on-device.

This module is designed to be import-safe: no heavy models are loaded at import
time.  The CLIP model is loaded lazily on the first call to :meth:`encode` (only
when ``backend="clip"``).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback bag-of-words encoder
# ---------------------------------------------------------------------------

# A small but representative robot-instruction vocabulary.  Tokens not in this
# vocabulary are hashed to a stable bucket so the encoder is robust to unseen
# words while remaining deterministic.
_DEFAULT_VOCAB: List[str] = [
    "pick", "place", "put", "grab", "grasp", "release", "move", "push", "pull",
    "lift", "drop", "open", "close", "rotate", "turn", "slide", "stack",
    "unstack", "pour", "insert", "remove", "the", "a", "an", "to", "into",
    "onto", "from", "up", "down", "left", "right", "forward", "backward",
    "red", "blue", "green", "yellow", "black", "white", "block", "cup",
    "bowl", "plate", "bottle", "can", "box", "table", "shelf", "drawer",
    "and", "over", "near", "far", "top", "bottom", "middle", "object",
    "item", "container", "tray", "bin", "zone", "area", "target", "start",
    "end", "goal", "home", "arm", "gripper", "hand", "robot", "task",
]

_NUM_HASH_BUCKETS = 512  # total fallback vocab size = known + hash buckets


def _tokenize_simple(text: str) -> List[str]:
    """Lowercase, strip punctuation, and split on whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


class BagOfWordsEncoder(nn.Module):
    """Lightweight bag-of-words + MLP text encoder (no external downloads).

    Maps each token to an index (known vocab or a stable hash bucket), averages
    the token embeddings, then projects through a small MLP to ``output_dim``.
    """

    def __init__(self, output_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.vocab_size = len(_DEFAULT_VOCAB) + _NUM_HASH_BUCKETS
        self.token_to_idx = {tok: i for i, tok in enumerate(_DEFAULT_VOCAB)}
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _token_to_idx(self, token: str) -> int:
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        # Deterministic hash bucket in range [len(vocab), vocab_size)
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        return len(_DEFAULT_VOCAB) + (h % _NUM_HASH_BUCKETS)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: ``(batch, seq_len)`` LongTensor of token indices.
            attention_mask: ``(batch, seq_len)`` mask (1 = valid, 0 = pad).
                If ``None``, all tokens are considered valid.

        Returns:
            ``(batch, output_dim)`` embedding tensor.
        """
        emb = self.embedding(token_ids)  # (B, S, H)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            emb = emb * mask
            lengths = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            pooled = emb.sum(dim=1) / lengths
        else:
            pooled = emb.mean(dim=1)  # (B, H)
        return self.mlp(pooled)  # (B, output_dim)

    def encode_text(self, texts: Union[str, List[str]], device: Optional[torch.device] = None) -> torch.Tensor:
        """Encode one or more instruction strings.

        Args:
            texts: A single string or list of strings.
            device: Device to place tensors on.

        Returns:
            ``(batch, output_dim)`` embedding tensor.
        """
        if isinstance(texts, str):
            texts = [texts]
        if device is None:
            device = next(self.parameters()).device
        batch_ids = []
        batch_masks = []
        max_len = 1
        for text in texts:
            tokens = _tokenize_simple(text)
            if not tokens:
                tokens = ["unknown"]
            ids = [self._token_to_idx(t) for t in tokens]
            batch_ids.append(ids)
            batch_masks.append([1.0] * len(ids))
            max_len = max(max_len, len(ids))
        # Pad to max_len
        padded_ids = []
        padded_masks = []
        for ids, masks in zip(batch_ids, batch_masks):
            pad = max_len - len(ids)
            padded_ids.append(ids + [0] * pad)
            padded_masks.append(masks + [0.0] * pad)
        token_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(padded_masks, dtype=torch.float, device=device)
        return self.forward(token_tensor, mask_tensor)


# ---------------------------------------------------------------------------
# Main TextEncoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Language instruction encoder with CLIP backend and BoW fallback.

    Tries to load ``openai/clip-vit-base-patch32`` text encoder from HuggingFace
    transformers.  If that fails (offline, missing weights, etc.), falls back to
    a :class:`BagOfWordsEncoder` so the rest of the pipeline still works.

    Args:
        output_dim: Desired embedding dimension.  If using CLIP, a linear
            projection maps from CLIP's hidden size to ``output_dim``.
        backend: Force a specific backend: ``"clip"``, ``"bow"``, or ``"auto"``
            (default — tries CLIP, falls back to BoW).
        model_name: HuggingFace model name for the CLIP backend.
        device: Device to load the CLIP model on (ignored for BoW).
    """

    def __init__(
        self,
        output_dim: int = 256,
        backend: str = "auto",
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.requested_backend = backend
        self.model_name = model_name
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Lazy-loaded internals
        self._clip_model: Optional[nn.Module] = None
        self._clip_tokenizer = None
        self._clip_hidden_size: int = 0
        self._projection: Optional[nn.Module] = None
        self._bow_encoder: Optional[BagOfWordsEncoder] = None
        self._active_backend: str = "auto"  # resolved on first encode()

    # -- backend resolution -------------------------------------------------

    def _try_load_clip(self) -> bool:
        """Attempt to load the CLIP text model. Returns True on success."""
        try:
            from transformers import CLIPModel, CLIPTokenizer  # type: ignore
        except ImportError:
            logger.warning("transformers not available — cannot load CLIP text encoder.")
            return False

        try:
            tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            model = CLIPModel.from_pretrained(self.model_name)
            # We only need the text tower
            clip_hidden = model.config.text_config.hidden_size
            self._clip_model = model.text_model.to(self.device)
            self._clip_model.eval()
            self._clip_tokenizer = tokenizer
            self._clip_hidden_size = clip_hidden
            self._projection = nn.Linear(clip_hidden, self.output_dim).to(self.device)
            logger.info(
                "TextEncoder: loaded CLIP text encoder '%s' (hidden=%d -> %d).",
                self.model_name, clip_hidden, self.output_dim,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TextEncoder: could not load CLIP '%s' (%s). "
                "Falling back to bag-of-words encoder.",
                self.model_name, exc,
            )
            return False

    def _ensure_loaded(self) -> None:
        """Lazily resolve and load the active backend on first use."""
        if self._active_backend != "auto":
            return  # already resolved

        if self.requested_backend == "bow":
            self._bow_encoder = BagOfWordsEncoder(output_dim=self.output_dim).to(self.device)
            self._active_backend = "bow"
            logger.info("TextEncoder: using bag-of-words backend (forced).")
            return

        if self.requested_backend in ("clip", "auto"):
            if self._try_load_clip():
                self._active_backend = "clip"
                return
            if self.requested_backend == "clip":
                raise RuntimeError(
                    f"backend='clip' requested but CLIP model '{self.model_name}' "
                    f"could not be loaded."
                )

        # Fallback
        self._bow_encoder = BagOfWordsEncoder(output_dim=self.output_dim).to(self.device)
        self._active_backend = "bow"
        logger.info("TextEncoder: using bag-of-words fallback backend.")

    # -- public API ---------------------------------------------------------

    @property
    def backend(self) -> str:
        """The resolved backend name (``'clip'`` or ``'bow'``).

        Forces lazy loading if not yet resolved.
        """
        self._ensure_loaded()
        return self._active_backend

    def encode(self, instruction_text: Union[str, List[str]]) -> torch.Tensor:
        """Encode instruction text(s) into embedding vectors.

        Args:
            instruction_text: A single string or list of strings.

        Returns:
            ``(batch, output_dim)`` FloatTensor on the encoder's device.
        """
        self._ensure_loaded()
        if isinstance(instruction_text, str):
            texts = [instruction_text]
        else:
            texts = list(instruction_text)

        if self._active_backend == "clip":
            return self._encode_clip(texts)
        return self._bow_encoder.encode_text(texts, device=self.device)

    def _encode_clip(self, texts: List[str]) -> torch.Tensor:
        """Encode using the CLIP text tower + linear projection."""
        assert self._clip_model is not None
        assert self._clip_tokenizer is not None
        assert self._projection is not None
        inputs = self._clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            # last_hidden_state: (B, S, H); pool using CLIP's pooled output
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                # Mean-pool over last hidden state
                pooled = outputs.last_hidden_state.mean(dim=1)
            projected = self._projection(pooled)
        return projected

    # -- nn.Module plumbing -------------------------------------------------

    def forward(self, instruction_text: Union[str, List[str]]) -> torch.Tensor:
        """Alias for :meth:`encode` so the encoder can be called directly."""
        return self.encode(instruction_text)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        """Return trainable parameters.

        For the CLIP backend, only the projection layer is trainable (the CLIP
        text tower is frozen).  For BoW, all parameters are trainable.
        """
        self._ensure_loaded()
        if self._active_backend == "clip" and self._projection is not None:
            return self._projection.parameters(recurse=recurse)
        if self._bow_encoder is not None:
            return self._bow_encoder.parameters(recurse=recurse)
        return super().parameters(recurse=recurse)

    def train(self, mode: bool = True):  # type: ignore[override]
        """Set training mode.

        For CLIP, the frozen text tower stays in eval mode; only the projection
        is toggled.  For BoW, the whole encoder follows ``mode``.
        """
        self._ensure_loaded()
        if self._active_backend == "clip":
            if self._projection is not None:
                self._projection.train(mode)
            # Keep frozen CLIP tower in eval
            if self._clip_model is not None:
                self._clip_model.eval()
        elif self._bow_encoder is not None:
            self._bow_encoder.train(mode)
        return self

    def to(self, *args, **kwargs):
        """Move TextEncoder and any lazily-loaded modules to device."""
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = torch.device(device)
            if self._clip_model is not None:
                self._clip_model = self._clip_model.to(*args, **kwargs)
            if self._projection is not None:
                self._projection = self._projection.to(*args, **kwargs)
            if self._bow_encoder is not None:
                self._bow_encoder = self._bow_encoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)
