"""Small compute-matched VLA for fair comparison against diffusion policies.

The :class:`SmallVLA` is a lightweight Vision-Language-Action model designed to
be roughly the same parameter count as the diffusion policies used in the
comparison study, so that any performance difference is attributable to the
*method* (VLA vs diffusion) rather than raw model size.

Architecture
------------
* **Vision encoder** — a small ViT-Base (patch 16, 224×224) implemented in pure
  PyTorch (no torchvision dependency).  ~12 layers, hidden 768, ~86M params.
* **Language encoder** — :class:`~vla_baselines.text_encoder.TextEncoder`
  (CLIP if available, BoW fallback).  Projects to ``hidden_dim``.
* **Fusion** — concatenate vision and language embeddings, project through a
  small MLP.
* **Action head** — MLP that predicts ``(horizon, action_dim)`` actions.

The model is fully trainable on GB10 with the ``mpc_vla`` conda env.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small ViT (pure PyTorch, no torchvision)
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Convert an image into a sequence of patch embeddings."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block (ViT-style)."""

    def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    """A small ViT-Base encoder (patch16, 224) in pure PyTorch.

    Matches the ``vit-base-patch16-224`` configuration: 12 layers, 12 heads,
    hidden 768, MLP ratio 4.  Randomly initialised (no pretrained weights) so
    it trains from scratch alongside the rest of the VLA.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Initialise positional embedding and cls token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, C, H, W)`` image tensor.

        Returns:
            ``(B, embed_dim)`` pooled features (CLS token after final norm).
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token


# ---------------------------------------------------------------------------
# Action head
# ---------------------------------------------------------------------------


class ActionHead(nn.Module):
    """MLP action head predicting ``(horizon, action_dim)`` actions."""

    def __init__(self, input_dim: int, action_dim: int, horizon: int, hidden_dim: int = 256, num_layers: int = 4) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.output_dim = action_dim * horizon
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, input_dim)`` fused features.

        Returns:
            ``(B, horizon, action_dim)`` action tensor.
        """
        out = self.net(x)  # (B, horizon * action_dim)
        return out.view(-1, self.horizon, self.action_dim)


# ---------------------------------------------------------------------------
# SmallVLA
# ---------------------------------------------------------------------------


class SmallVLA(nn.Module):
    """Small compute-matched Vision-Language-Action model.

    Combines a small ViT vision encoder, a text encoder, and an MLP action head.
    Designed for fair comparison against small diffusion policies.

    Args:
        action_dim: Dimensionality of the action vector per timestep.
        horizon: Number of future timesteps to predict.
        hidden_dim: Hidden dimension of the fusion MLP and action head.
        num_layers: Number of layers in the action head MLP.
        img_size: Input image size (square).
        text_backend: Backend for :class:`TextEncoder` — ``"auto"``, ``"clip"``,
            or ``"bow"``.
        device: Device to place the model on.

    Example:
        >>> vla = SmallVLA(action_dim=7, horizon=4)
        >>> action = vla.predict_action(image, "pick up the red block")
    """

    def __init__(
        self,
        action_dim: int = 7,
        horizon: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 4,
        img_size: int = 224,
        text_backend: str = "auto",
        text_model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.img_size = img_size
        self._device = torch.device(device) if device is not None else torch.device("cpu")

        # Vision encoder (ViT-Base, ~86M params)
        self.vision_encoder = SmallViT(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )
        vision_dim = 768

        # Language encoder
        self.text_encoder = TextEncoder(
            output_dim=hidden_dim,
            backend=text_backend,
            model_name=text_model_name,
            device=self._device,
        )

        # Fusion: concatenate vision + language, project to hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Action head
        self.action_head = ActionHead(
            input_dim=hidden_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Move to device (text_encoder loads lazily, so move sub-modules)
        self.to(self._device)

    # -- device helpers -----------------------------------------------------

    @property
    def device(self) -> torch.device:
        """The device the model parameters live on."""
        try:
            return next(self.vision_encoder.parameters()).device
        except StopIteration:
            return self._device

    # -- preprocessing ------------------------------------------------------

    def _preprocess_image(self, image: Union[np.ndarray, torch.Tensor, "Image"]) -> torch.Tensor:
        """Convert an image to a normalised ``(1, 3, H, W)`` float tensor."""
        if hasattr(image, "convert"):
            # PIL Image
            image = image.convert("RGB").resize((self.img_size, self.img_size))
            image = np.array(image, dtype=np.float32) / 255.0
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if image.dtype != torch.float32:
            image = image.float()
        # Handle shape variations
        if image.ndim == 2:
            image = image.unsqueeze(-1).expand(-1, -1, 3)
        if image.ndim == 3:
            # (H, W, C) -> (C, H, W)
            if image.shape[-1] in (1, 3, 4):
                image = image.permute(2, 0, 1)
            # Now (C, H, W)
            if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
                image = F.interpolate(image.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
            image = image.unsqueeze(0)  # (1, C, H, W)
        elif image.ndim == 4:
            # (B, H, W, C) -> (B, C, H, W)
            if image.shape[-1] in (1, 3, 4):
                image = image.permute(0, 3, 1, 2)
            if image.shape[2] != self.img_size or image.shape[3] != self.img_size:
                image = F.interpolate(image, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"Unsupported image ndim={image.ndim}")
        # Normalise to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        # Ensure 3 channels
        if image.shape[1] == 1:
            image = image.expand(-1, 3, -1, -1)
        elif image.shape[1] == 4:
            image = image[:, :3]
        image = (image - mean) / std
        return image

    # -- forward ------------------------------------------------------------

    def forward(self, images: torch.Tensor, instructions: Union[str, List[str]]) -> torch.Tensor:
        """Forward pass.

        Args:
            images: ``(B, C, H, W)`` preprocessed image tensor.
            instructions: A single string (batch=1) or list of strings.

        Returns:
            ``(B, horizon, action_dim)`` predicted action tensor.
        """
        if isinstance(instructions, str):
            instructions = [instructions]
        # Ensure images is on the right device
        images = images.to(self.device)
        # Vision
        vis_feat = self.vision_encoder(images)  # (B, 768)
        # Language
        lang_feat = self.text_encoder.encode(instructions)  # (B, hidden_dim)
        lang_feat = lang_feat.to(self.device)
        # Fuse
        fused = torch.cat([vis_feat, lang_feat], dim=-1)  # (B, 768 + hidden)
        fused = self.fusion(fused)  # (B, hidden)
        # Action
        actions = self.action_head(fused)  # (B, horizon, action_dim)
        return actions

    # -- prediction ---------------------------------------------------------

    def predict_action(self, image: Union[np.ndarray, torch.Tensor, "Image"], instruction: str) -> np.ndarray:
        """Predict an action sequence from a single image + instruction.

        Args:
            image: A PIL Image, numpy array ``(H, W, C)``, or torch tensor.
            instruction: Natural-language instruction string.

        Returns:
            ``(horizon, action_dim)`` numpy array of predicted actions.
        """
        # Use eval_mode() instead of eval() — eval() calls self.train(False)
        # which would invoke our overridden train() training-loop method.
        self.eval_mode()
        with torch.no_grad():
            img_tensor = self._preprocess_image(image).to(self.device)
            action = self.forward(img_tensor, instruction)  # (1, H, A)
        return action.squeeze(0).cpu().numpy()

    # -- training -----------------------------------------------------------

    def train(
        self,
        demonstrations: Sequence[Dict[str, Any]],
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        verbose: bool = True,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """Train the SmallVLA on demonstration data.

        Each demonstration is a dict with keys:
            - ``"image"``: PIL Image / np.ndarray ``(H, W, C)`` / torch tensor
            - ``"instruction"``: str
            - ``"action"``: np.ndarray or torch tensor of shape
              ``(horizon, action_dim)`` or ``(action_dim,)``

        Args:
            demonstrations: Sequence of demonstration dicts.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate.
            weight_decay: Adam weight decay.
            verbose: If True, print progress.
            log_interval: Print loss every N batches.

        Returns:
            Dict with ``"epoch_loss"`` and ``"batch_loss"`` lists.
        """
        self.train_mode()
        device = self.device

        # Preprocess all demonstrations
        images: List[torch.Tensor] = []
        instructions: List[str] = []
        actions: List[torch.Tensor] = []
        for demo in demonstrations:
            img = self._preprocess_image(demo["image"])
            images.append(img.squeeze(0))  # (C, H, W)
            instructions.append(demo["instruction"])
            act = demo["action"]
            if isinstance(act, np.ndarray):
                act = torch.from_numpy(act).float()
            if act.ndim == 1:
                # (action_dim,) -> (1, action_dim) -> broadcast later
                act = act.unsqueeze(0)
            # Ensure shape (horizon, action_dim)
            if act.shape[0] != self.horizon:
                if act.shape[0] == 1:
                    act = act.expand(self.horizon, -1)
                else:
                    # Truncate or pad
                    if act.shape[0] > self.horizon:
                        act = act[: self.horizon]
                    else:
                        pad = torch.zeros(self.horizon - act.shape[0], act.shape[1])
                        act = torch.cat([act, pad], dim=0)
            actions.append(act)

        images_tensor = torch.stack(images).to(device)  # (N, C, H, W)
        actions_tensor = torch.stack(actions).to(device)  # (N, horizon, action_dim)
        n = len(demonstrations)

        # Collect trainable parameters (text encoder projection + vision + heads)
        params = list(self.vision_encoder.parameters())
        params += list(self.fusion.parameters())
        params += list(self.action_head.parameters())
        # Text encoder params (projection for CLIP, or full BoW)
        for p in self.text_encoder.parameters():
            params.append(p)

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        epoch_losses: List[float] = []
        batch_losses: List[float] = []

        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                batch_imgs = images_tensor[idx]
                batch_acts = actions_tensor[idx]
                batch_instr = [instructions[j] for j in idx.tolist()]
                pred = self.forward(batch_imgs, batch_instr)
                loss = criterion(pred, batch_acts)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                batch_losses.append(loss.item())
                if verbose and (num_batches % log_interval == 0):
                    logger.info(
                        "SmallVLA epoch %d/%d batch %d loss=%.6f",
                        epoch + 1, epochs, num_batches, loss.item(),
                    )
            avg = epoch_loss / max(num_batches, 1)
            epoch_losses.append(avg)
            if verbose:
                logger.info("SmallVLA epoch %d/%d avg_loss=%.6f", epoch + 1, epochs, avg)

        return {"epoch_loss": epoch_losses, "batch_loss": batch_losses}

    def train_mode(self) -> "SmallVLA":
        """Set the model to training mode (alias for ``self.train(True)``).

        Note: ``nn.Module.train`` is overridden by our ``train`` method above,
        so we provide this to set training mode without ambiguity.
        """
        super().train(True)
        self.text_encoder.train(True)
        return self

    def eval_mode(self) -> "SmallVLA":
        """Set the model to evaluation mode."""
        super().train(False)
        self.text_encoder.train(False)
        return self

    # -- save / load --------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to a ``.pt`` file.

        Args:
            path: File path (typically ending in ``.pt``).
        """
        # Force the text encoder to initialise its (lazily-loaded) sub-modules
        # so that their weights are included in the state dict.  Without this,
        # the BoW encoder's embedding/MLP don't exist yet and are silently
        # omitted from the saved checkpoint.
        self.text_encoder._ensure_loaded()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            "model_state_dict": self.state_dict(),
            "config": {
                "action_dim": self.action_dim,
                "horizon": self.horizon,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "img_size": self.img_size,
            },
        }
        torch.save(state, path)
        logger.info("SmallVLA saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[Union[str, torch.device]] = None, text_backend: str = "bow") -> "SmallVLA":
        """Load a SmallVLA from a ``.pt`` file.

        Uses ``text_backend="bow"`` by default to avoid requiring CLIP weights
        at load time.  Override with ``text_backend="auto"`` or ``"clip"`` to
        use CLIP.

        Args:
            path: Path to the saved ``.pt`` file.
            device: Device to load the model on.
            text_backend: Text encoder backend to use.

        Returns:
            A loaded :class:`SmallVLA` instance.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = state["config"]
        model = cls(
            action_dim=config["action_dim"],
            horizon=config["horizon"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            img_size=config.get("img_size", 224),
            text_backend=text_backend,
            device=device,
        )
        # Force the text encoder to initialise its (lazily-loaded) sub-modules
        # so that load_state_dict can populate their weights.  Without this,
        # the BoW encoder's embedding/MLP don't exist yet and strict=False
        # silently skips them, resulting in mismatched predictions.
        model.text_encoder._ensure_loaded()
        model.load_state_dict(state["model_state_dict"], strict=False)
        model.to(model._device)
        logger.info("SmallVLA loaded from %s", path)
        return model

    # -- parameter count (useful for compute matching) ----------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        # Force text encoder to load so we count its params too
        self.text_encoder._ensure_loaded()
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
