"""Training loop for SmallVLA with validation, early stopping, and LR scheduling.

The :class:`VLATrainer` provides a richer training loop than the built-in
:meth:`SmallVLA.train` method:

* **Train / validation split** — automatically holds out a fraction of
  demonstrations for validation.
* **Early stopping** — monitors validation loss and stops when it plateaus.
* **LR scheduling** — cosine-annealing or step-decay schedule.
* **Optional wandb logging** — if the ``wandb`` package is available and the
  user calls :meth:`init_wandb`, metrics are logged to Weights & Biases.
* **Checkpointing** — save / load full training state (model + optimizer +
  scheduler + epoch).

The trainer is designed for :class:`SmallVLA` (the OpenVLA-7B model is too
large to fine-tune on the GB10 and is used for inference only).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Optional wandb import — trainer works without it
try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _wandb = None
    _WANDB_AVAILABLE = False


class VLATrainer:
    """Training loop for :class:`SmallVLA` with val split, early stopping, and LR scheduling.

    Args:
        model: A :class:`SmallVLA` instance to train.
        lr: Initial learning rate for the Adam optimizer.
        weight_decay: Adam weight decay.
        device: Device to train on.  ``"auto"`` uses the model's current
            device, ``"cuda"`` / ``"cpu"`` are also accepted.
        scheduler_type: Learning-rate schedule — ``"cosine"`` or ``"step"``.
        scheduler_kwargs: Extra kwargs forwarded to the scheduler constructor.
            For ``"cosine"``: ``T_max`` (defaults to ``num_epochs``).
            For ``"step"``: ``step_size`` (default 30) and ``gamma`` (default 0.1).
        patience: Number of epochs with no val-loss improvement before early
            stopping triggers.
        min_delta: Minimum val-loss decrease to count as an improvement.
        grad_clip: Max gradient norm for clipping (0 to disable).
        verbose: If True, log progress to ``logging``.

    Example:
        >>> vla = SmallVLA(action_dim=7, horizon=4, img_size=64, text_backend="bow")
        >>> trainer = VLATrainer(vla, lr=1e-4)
        >>> history = trainer.train(demos, num_epochs=50, batch_size=16)
        >>> trainer.save_checkpoint("checkpoints/vla.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "auto",
        scheduler_type: str = "cosine",
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        patience: int = 10,
        min_delta: float = 1e-5,
        grad_clip: float = 1.0,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        self.verbose = verbose

        # Resolve device
        if device == "auto":
            if hasattr(model, "device"):
                self.device = model.device
            else:
                self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
        # Move model to device
        if hasattr(model, "to"):
            model.to(self.device)

        # Optimizer — collect trainable parameters
        params = self._collect_parameters(model)
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        # Scheduler (created lazily in train() so we know num_epochs)
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self._num_epochs_for_scheduler: int = 0

        # Loss
        self.criterion = nn.MSELoss()

        # State
        self.current_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.epochs_no_improve: int = 0
        self.stopped_early: bool = False
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        # wandb
        self._wandb_run: Optional[Any] = None

    # ------------------------------------------------------------------
    # Parameter collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
        """Collect trainable parameters, handling SmallVLA's text encoder quirks.

        SmallVLA's :class:`TextEncoder` overrides ``parameters()`` to return
        only the projection (CLIP) or the BoW encoder.  We iterate over
        ``model.named_parameters()`` to be safe and include everything with
        ``requires_grad=True``.
        """
        params: List[torch.nn.Parameter] = []
        for p in model.parameters():
            if p.requires_grad:
                params.append(p)
        if not params:
            # Fallback: try named_parameters to catch edge cases
            for _name, p in model.named_parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _build_scheduler(self, num_epochs: int) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Build the LR scheduler based on ``scheduler_type``."""
        if self.scheduler_type == "cosine":
            t_max = self.scheduler_kwargs.get("T_max", num_epochs)
            eta_min = self.scheduler_kwargs.get("eta_min", 0.0)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=eta_min,
            )
        elif self.scheduler_type == "step":
            step_size = self.scheduler_kwargs.get("step_size", 30)
            gamma = self.scheduler_kwargs.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma,
            )
        elif self.scheduler_type in ("none", None):
            return None
        else:
            logger.warning(
                "VLATrainer: unknown scheduler_type '%s' — no scheduler used.",
                self.scheduler_type,
            )
            return None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        demonstrations: Sequence[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Preprocess demonstrations into tensors.

        Each demo dict must have:
            - ``"image"``: PIL Image / np.ndarray (H,W,C) / torch tensor
            - ``"instruction"``: str
            - ``"action"``: np.ndarray or torch tensor ``(horizon, action_dim)``
              or ``(action_dim,)``
        """
        # Use SmallVLA's preprocessing if available
        has_preprocess = hasattr(self.model, "_preprocess_image")
        action_dim = getattr(self.model, "action_dim", 7)
        horizon = getattr(self.model, "horizon", 4)

        images: List[torch.Tensor] = []
        instructions: List[str] = []
        actions: List[torch.Tensor] = []

        for demo in demonstrations:
            # Image
            if has_preprocess:
                img = self.model._preprocess_image(demo["image"]).squeeze(0)
            else:
                img = torch.as_tensor(np.asarray(demo["image"]), dtype=torch.float32)
                if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                    img = img.permute(2, 0, 1)
                if img.ndim == 2:
                    img = img.unsqueeze(0).expand(3, -1, -1)
            images.append(img)

            instructions.append(demo["instruction"])

            # Action
            act = demo["action"]
            if isinstance(act, np.ndarray):
                act = torch.from_numpy(act).float()
            if act.ndim == 1:
                act = act.unsqueeze(0)
            # Ensure (horizon, action_dim)
            if act.shape[0] != horizon:
                if act.shape[0] == 1:
                    act = act.expand(horizon, -1)
                elif act.shape[0] > horizon:
                    act = act[:horizon]
                else:
                    pad = torch.zeros(horizon - act.shape[0], act.shape[1])
                    act = torch.cat([act, pad], dim=0)
            actions.append(act)

        images_tensor = torch.stack(images).to(self.device)
        actions_tensor = torch.stack(actions).to(self.device)
        return images_tensor, instructions, actions_tensor

    @staticmethod
    def _split_indices(
        n: int,
        val_split: float,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split ``n`` indices into train and validation sets."""
        indices = np.arange(n)
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)
        n_val = max(1, int(n * val_split)) if val_split > 0 else 0
        if n_val == 0 or n_val >= n:
            return indices, np.array([], dtype=int)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return train_idx, val_idx

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------

    def init_wandb(
        self,
        project: str = "mpc_vla_diffusion_study",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Initialise a wandb run for logging.

        Returns ``True`` if wandb was successfully initialised, ``False`` if
        wandb is not available.
        """
        if not _WANDB_AVAILABLE:
            logger.warning("VLATrainer: wandb not available — skipping logging.")
            return False
        self._wandb_run = _wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            reinit=True,
        )
        logger.info("VLATrainer: wandb run '%s' initialised.", run_name or "unnamed")
        return True

    def _wandb_log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb if a run is active."""
        if self._wandb_run is not None and _WANDB_AVAILABLE:
            _wandb.log(metrics, step=step)

    def finish_wandb(self) -> None:
        """Finish the wandb run if one is active."""
        if self._wandb_run is not None and _WANDB_AVAILABLE:
            _wandb.finish()
            self._wandb_run = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        demonstrations: Sequence[Dict[str, Any]],
        num_epochs: int,
        batch_size: int = 32,
        val_split: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Dict[str, List[float]]:
        """Run the full training loop with train/val tracking and early stopping.

        Args:
            demonstrations: Sequence of demo dicts (see :meth:`_prepare_data`).
            num_epochs: Maximum number of training epochs.
            batch_size: Mini-batch size.
            val_split: Fraction of data to use for validation (0–0.5).
            shuffle: Whether to shuffle data each epoch.
            seed: Random seed for reproducible splits.

        Returns:
            Dict with ``"train_loss"``, ``"val_loss"``, and ``"lr"`` lists
            (one entry per epoch).
        """
        n = len(demonstrations)
        if n == 0:
            raise ValueError("No demonstrations provided for training.")

        # Prepare data
        images_tensor, instructions, actions_tensor = self._prepare_data(demonstrations)

        # Split
        train_idx, val_idx = self._split_indices(n, val_split, shuffle=shuffle, seed=seed)
        has_val = len(val_idx) > 0

        if self.verbose:
            logger.info(
                "VLATrainer: %d demos -> %d train, %d val | epochs=%d batch=%d",
                n, len(train_idx), len(val_idx), num_epochs, batch_size,
            )

        # Build scheduler
        self.scheduler = self._build_scheduler(num_epochs)
        self._num_epochs_for_scheduler = num_epochs

        # Reset early-stopping state
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.stopped_early = False
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

        # Set model to training mode
        if hasattr(self.model, "train_mode"):
            self.model.train_mode()
        else:
            self.model.train(True)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            t0 = time.perf_counter()

            # --- Train phase ---
            train_loss = self._run_epoch(
                images_tensor, instructions, actions_tensor,
                train_idx, batch_size, train=True, shuffle=shuffle, seed=seed + epoch,
            )

            # --- Validation phase ---
            if has_val:
                if hasattr(self.model, "eval_mode"):
                    self.model.eval_mode()
                else:
                    self.model.train(False)
                val_loss = self._run_epoch(
                    images_tensor, instructions, actions_tensor,
                    val_idx, batch_size, train=False, shuffle=False,
                )
                if hasattr(self.model, "train_mode"):
                    self.model.train_mode()
                else:
                    self.model.train(True)
            else:
                val_loss = train_loss  # no val data

            # --- LR step ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Record ---
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            elapsed = time.perf_counter() - t0
            if self.verbose:
                logger.info(
                    "VLATrainer: epoch %d/%d | train_loss=%.6f val_loss=%.6f lr=%.2e (%.1fs)",
                    epoch + 1, num_epochs, train_loss, val_loss, current_lr, elapsed,
                )

            # --- wandb log ---
            self._wandb_log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "epoch": epoch + 1,
            }, step=epoch + 1)

            # --- Early stopping ---
            if has_val:
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.verbose:
                        logger.info(
                            "VLATrainer: no improvement for %d epoch(s) "
                            "(best=%.6f at epoch %d).",
                            self.epochs_no_improve, self.best_val_loss,
                            self.best_epoch + 1,
                        )
                    if self.epochs_no_improve >= self.patience:
                        self.stopped_early = True
                        if self.verbose:
                            logger.info(
                                "VLATrainer: early stopping at epoch %d "
                                "(patience=%d).",
                                epoch + 1, self.patience,
                            )
                        break

        if self.verbose:
            logger.info(
                "VLATrainer: training complete. Best val_loss=%.6f at epoch %d.",
                self.best_val_loss, self.best_epoch + 1,
            )

        return self.history

    def _run_epoch(
        self,
        images_tensor: torch.Tensor,
        instructions: List[str],
        actions_tensor: torch.Tensor,
        indices: np.ndarray,
        batch_size: int,
        train: bool,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> float:
        """Run one epoch over the given indices. Returns average loss."""
        n = len(indices)
        if n == 0:
            return 0.0

        if shuffle and train and seed is not None:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(indices)
        elif shuffle and train:
            perm = np.random.permutation(indices)
        else:
            perm = indices

        total_loss = 0.0
        num_batches = 0

        for i in range(0, n, batch_size):
            batch_idx = perm[i : i + batch_size]
            batch_imgs = images_tensor[batch_idx]
            batch_acts = actions_tensor[batch_idx]
            batch_instr = [instructions[j] for j in batch_idx.tolist()]

            if train:
                self.optimizer.zero_grad()
                pred = self.model.forward(batch_imgs, batch_instr)
                loss = self.criterion(pred, batch_acts)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip,
                    )
                self.optimizer.step()
            else:
                with torch.no_grad():
                    pred = self.model.forward(batch_imgs, batch_instr)
                    loss = self.criterion(pred, batch_acts)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save full training state to ``path``.

        Saves model state dict, optimizer state, scheduler state, and training
        metadata (epoch, best val loss, history).
        """
        dir_path = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir_path, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "scheduler_type": self.scheduler_type,
        }
        # Include model config if available (SmallVLA)
        if hasattr(self.model, "action_dim"):
            checkpoint["model_config"] = {
                "action_dim": self.model.action_dim,
                "horizon": self.model.horizon,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "img_size": self.model.img_size,
            }

        torch.save(checkpoint, path)
        logger.info("VLATrainer: checkpoint saved to %s", path)

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> Dict[str, Any]:
        """Load training state from ``path``.

        Args:
            path: Path to the checkpoint file.
            load_optimizer: If True, restore optimizer state.
            load_scheduler: If True, restore scheduler state (if present).

        Returns:
            The checkpoint dict (useful for inspecting metadata).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLATrainer: could not load optimizer state: %s", exc)

        if (
            load_scheduler
            and self.scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
        ):
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLATrainer: could not load scheduler state: %s", exc)

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": [], "lr": []})

        logger.info(
            "VLATrainer: checkpoint loaded from %s (epoch=%d, best_val=%.6f).",
            path, self.current_epoch, self.best_val_loss,
        )
        return checkpoint

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        demonstrations: Sequence[Dict[str, Any]],
        batch_size: int = 32,
    ) -> float:
        """Evaluate the model on a set of demonstrations (no gradient).

        Returns the average MSE loss.
        """
        images_tensor, instructions, actions_tensor = self._prepare_data(demonstrations)
        indices = np.arange(len(demonstrations))

        if hasattr(self.model, "eval_mode"):
            self.model.eval_mode()
        else:
            self.model.train(False)

        avg_loss = self._run_epoch(
            images_tensor, instructions, actions_tensor,
            indices, batch_size, train=False, shuffle=False,
        )
        return avg_loss
