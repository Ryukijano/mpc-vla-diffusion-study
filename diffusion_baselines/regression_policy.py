"""Regression Policy (RCP baseline).

A simple MLP policy that maps an *observation* directly to an *action
sequence* of length ``horizon``.  Unlike the diffusion and flow-matching
policies, this baseline performs **no iterative denoising** -- a single
forward pass through the network produces the entire action sequence.

This serves as the **lower-bound baseline** (RCP -- Regression Control
Policy) in the study: it tests how far a non-iterative, non-generative
policy can go with the same observation-conditioning interface as the
diffusion baselines.

Training
--------
Standard supervised regression: minimise the MSE between the predicted
action sequence and the ground-truth demonstration action sequence.

GPU support
-----------
The model auto-detects CUDA and moves itself to the GPU when available.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _RegressionMLP(nn.Module):
    """Simple MLP: ``obs (obs_dim) -> action_seq (horizon * action_dim)``."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.output_dim = self.horizon * self.action_dim

        layers: List[nn.Module] = []
        in_dim = self.obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs: (B, obs_dim)`` -> ``(B, horizon, action_dim)``."""
        out = self.net(obs)
        return out.view(-1, self.horizon, self.action_dim)


class RegressionPolicy:
    """Plain MLP regression policy (RCP baseline).

    Maps an observation directly to an action sequence with a single
    forward pass -- no diffusion, no iteration.

    Parameters
    ----------
    action_dim : dimensionality of each action in the sequence.
    horizon : number of actions in the generated sequence.
    obs_dim : dimensionality of the observation (state) vector.
    hidden_dim : width of each hidden layer.
    num_layers : number of hidden layers (excluding the output layer).
    device : device to place the model on.  ``'auto'`` selects CUDA if
        available, otherwise CPU.

    Attributes
    ----------
    net : the underlying :class:`_RegressionMLP`.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        if device == "auto" or device is None:
            self.device = _default_device()
        else:
            self.device = torch.device(device)

        self.net = _RegressionMLP(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            horizon=self.horizon,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        self._trained = False

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _prepare_demos(
        self,
        demonstrations: Sequence[Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert demonstrations to stacked tensors on the correct device."""
        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        for obs, act in demonstrations:
            obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
            act_arr = np.asarray(act, dtype=np.float32).reshape(self.horizon, self.action_dim)
            obs_list.append(obs_arr)
            act_list.append(act_arr)
        obs_tensor = torch.from_numpy(np.stack(obs_list)).to(self.device)
        act_tensor = torch.from_numpy(np.stack(act_list)).to(self.device)
        assert obs_tensor.shape[1] == self.obs_dim, (
            f"Observation dim mismatch: expected {self.obs_dim}, got {obs_tensor.shape[1]}"
        )
        return obs_tensor, act_tensor

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        demonstrations: Sequence[Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]],
        num_epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> List[float]:
        """Train the regression policy on ``(observation, action_sequence)`` pairs.

        Parameters
        ----------
        demonstrations : iterable of ``(obs, action_seq)`` where ``obs`` has
            shape ``(obs_dim,)`` and ``action_seq`` has shape
            ``(horizon, action_dim)``.
        num_epochs : number of passes over the data.
        batch_size : mini-batch size.
        lr : Adam learning rate.
        verbose : print the loss every 10 epochs.

        Returns
        -------
        losses : list of mean loss per epoch.
        """
        obs_tensor, act_tensor = self._prepare_demos(demonstrations)
        n = obs_tensor.shape[0]
        if n == 0:
            raise ValueError("Need at least one demonstration to train.")

        dataset = TensorDataset(obs_tensor, act_tensor)
        loader = DataLoader(dataset, batch_size=min(batch_size, n), shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        losses: List[float] = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for obs_b, act_b in loader:
                pred = self.net(obs_b)
                loss = F.mse_loss(pred, act_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"[RCP] epoch {epoch:4d}/{num_epochs}  loss={avg_loss:.6f}")

        self._trained = True
        return losses

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, observation: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Predict an action sequence from a single observation.

        Parameters
        ----------
        observation : observation vector of shape ``(obs_dim,)`` (or a batch
            ``(B, obs_dim)``).

        Returns
        -------
        actions : tensor of shape ``(horizon, action_dim)`` for a single
            observation, or ``(B, horizon, action_dim)`` for a batch.
        """
        self.net.eval()
        obs_arr = np.asarray(observation, dtype=np.float32)
        single = obs_arr.ndim == 1
        if single:
            obs_arr = obs_arr[None, :]
        obs_t = torch.from_numpy(obs_arr).to(self.device)
        out = self.net(obs_t)
        if single:
            return out[0]
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the policy to a ``.pt`` file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": {
                "action_dim": self.action_dim,
                "horizon": self.horizon,
                "obs_dim": self.obs_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
            },
            "trained": self._trained,
        }, path)

    def load(self, path: str, map_location: Optional[torch.device] = None) -> "RegressionPolicy":
        """Load weights from a ``.pt`` file (in-place)."""
        device = map_location or self.device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(device)
        self._trained = bool(ckpt.get("trained", True))
        return self

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> "RegressionPolicy":
        """Construct a policy from a saved checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        policy = cls(
            action_dim=cfg["action_dim"],
            horizon=cfg["horizon"],
            obs_dim=cfg["obs_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            device=device if device is not None else "auto",
        )
        policy.net.load_state_dict(ckpt["state_dict"])
        policy._trained = bool(ckpt.get("trained", True))
        return policy
