"""Iterative Regression Policy (MIP baseline).

This policy tests the **iterative compute hypothesis** of Simchowitz et al.:
that multi-step iterative refinement of an action sequence -- even with a
simple regression network -- can outperform a single-shot regressor.

The policy maintains a sequence of ``num_iterations`` regression networks.
At inference time the output of iteration *k* is perturbed with Gaussian
noise (scaled by ``noise_std``) and fed as additional conditioning to
iteration *k+1*.  This iterative refinement with noise injection is the
hallmark of the **Minimal Iterative Policy (MIP)** -- when
``num_iterations=2`` and ``noise_std > 0`` the policy *is* the MIP.

Training
--------
Each iteration step is trained independently to predict the ground-truth
action sequence, conditioned on the observation *and* (for steps > 0) a
noisy version of the previous step's prediction.  This teaches every step
to refine the noisy estimate toward the true target.

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


class _IterRegressionMLP(nn.Module):
    """MLP that maps ``[obs, prev_action]`` -> ``action_seq``.

    For the first iteration ``prev_action`` is zero (no prior estimate).
    For subsequent iterations it is a noise-perturbed prediction from the
    previous step.
    """

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
        # Input: observation + previous (noisy) action estimate.
        input_dim = self.obs_dim + self.horizon * self.action_dim

        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        """``obs: (B, obs_dim)``, ``prev_action: (B, horizon, action_dim)``.

        Returns ``(B, horizon, action_dim)``.
        """
        prev_flat = prev_action.reshape(prev_action.shape[0], -1)
        x = torch.cat([obs, prev_flat], dim=-1)
        out = self.net(x)
        return out.view(-1, self.horizon, self.action_dim)


class IterativeRegressionPolicy:
    """Multi-step regression with noise injection (MIP baseline).

    Parameters
    ----------
    action_dim : dimensionality of each action in the sequence.
    horizon : number of actions in the generated sequence.
    obs_dim : dimensionality of the observation (state) vector.
    num_iterations : number of iterative refinement steps.  When set to 2
        (with ``noise_std > 0``) this is the **Minimal Iterative Policy
        (MIP)**.
    hidden_dim : width of each hidden layer in every step's MLP.
    noise_std : standard deviation of Gaussian noise injected between
        iterations.  ``0`` disables noise (pure iterative refinement).
    device : device to place the model on.  ``'auto'`` selects CUDA if
        available, otherwise CPU.

    Attributes
    ----------
    nets : list of :class:`_IterRegressionMLP`, one per iteration step.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        obs_dim: int,
        num_iterations: int = 2,
        hidden_dim: int = 256,
        noise_std: float = 0.1,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.obs_dim = int(obs_dim)
        self.num_iterations = int(num_iterations)
        self.hidden_dim = int(hidden_dim)
        self.noise_std = float(noise_std)

        if device == "auto" or device is None:
            self.device = _default_device()
        else:
            self.device = torch.device(device)

        self.nets = nn.ModuleList([
            _IterRegressionMLP(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                horizon=self.horizon,
                hidden_dim=self.hidden_dim,
                num_layers=4,
            )
            for _ in range(self.num_iterations)
        ]).to(self.device)
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
        """Train each iteration step with noise injection.

        Each of the ``num_iterations`` networks is trained to predict the
        ground-truth action sequence.  For step *k > 0* the input includes a
        noise-perturbed prediction from step *k-1* (simulating the inference-
        time refinement chain).

        Parameters
        ----------
        demonstrations : iterable of ``(obs, action_seq)``.
        num_epochs : number of passes over the data **per iteration step**.
        batch_size : mini-batch size.
        lr : Adam learning rate.
        verbose : print the loss every 10 epochs.

        Returns
        -------
        losses : list of mean loss per epoch, concatenated across all
            iteration steps.
        """
        obs_tensor, act_tensor = self._prepare_demos(demonstrations)
        n = obs_tensor.shape[0]
        if n == 0:
            raise ValueError("Need at least one demonstration to train.")

        dataset = TensorDataset(obs_tensor, act_tensor)
        loader = DataLoader(dataset, batch_size=min(batch_size, n), shuffle=True)

        all_losses: List[float] = []

        for step in range(self.num_iterations):
            net = self.nets[step]
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            net.train()

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                n_batches = 0
                for obs_b, act_b in loader:
                    bs = obs_b.shape[0]
                    if step == 0:
                        # First step: no prior estimate (zeros).
                        prev = torch.zeros(bs, self.horizon, self.action_dim, device=self.device)
                    else:
                        # Run the previous step's network (no grad) to get
                        # a prediction, then inject noise.
                        with torch.no_grad():
                            prev_net = self.nets[step - 1]
                            prev_net.eval()
                            prev = prev_net(obs_b, torch.zeros(bs, self.horizon, self.action_dim, device=self.device))
                        if self.noise_std > 0:
                            prev = prev + self.noise_std * torch.randn_like(prev)

                    pred = net(obs_b, prev)
                    loss = F.mse_loss(pred, act_b)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                all_losses.append(avg_loss)
                if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                    print(f"[MIP step {step}] epoch {epoch:4d}/{num_epochs}  loss={avg_loss:.6f}")

        self._trained = True
        return all_losses

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, observation: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Predict an action sequence via iterative refinement.

        Runs ``num_iterations`` forward passes.  Between steps, Gaussian
        noise (scaled by ``noise_std``) is injected into the intermediate
        prediction, which the next step learns to refine.

        Parameters
        ----------
        observation : observation vector of shape ``(obs_dim,)`` (or a batch
            ``(B, obs_dim)``).

        Returns
        -------
        actions : tensor of shape ``(horizon, action_dim)`` for a single
            observation, or ``(B, horizon, action_dim)`` for a batch.
        """
        for net in self.nets:
            net.eval()

        obs_arr = np.asarray(observation, dtype=np.float32)
        single = obs_arr.ndim == 1
        if single:
            obs_arr = obs_arr[None, :]
        obs_t = torch.from_numpy(obs_arr).to(self.device)
        bs = obs_t.shape[0]

        prev = torch.zeros(bs, self.horizon, self.action_dim, device=self.device)

        for step in range(self.num_iterations):
            pred = self.nets[step](obs_t, prev)
            if step < self.num_iterations - 1 and self.noise_std > 0:
                # Inject noise between steps (the last step returns clean output).
                prev = pred + self.noise_std * torch.randn_like(pred)
            else:
                prev = pred

        if single:
            return prev[0]
        return prev

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the policy to a ``.pt`` file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "state_dict": self.nets.state_dict(),
            "config": {
                "action_dim": self.action_dim,
                "horizon": self.horizon,
                "obs_dim": self.obs_dim,
                "num_iterations": self.num_iterations,
                "hidden_dim": self.hidden_dim,
                "noise_std": self.noise_std,
            },
            "trained": self._trained,
        }, path)

    def load(self, path: str, map_location: Optional[torch.device] = None) -> "IterativeRegressionPolicy":
        """Load weights from a ``.pt`` file (in-place)."""
        device = map_location or self.device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.nets.load_state_dict(ckpt["state_dict"])
        self.nets.to(device)
        self._trained = bool(ckpt.get("trained", True))
        return self

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> "IterativeRegressionPolicy":
        """Construct a policy from a saved checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        policy = cls(
            action_dim=cfg["action_dim"],
            horizon=cfg["horizon"],
            obs_dim=cfg["obs_dim"],
            num_iterations=cfg["num_iterations"],
            hidden_dim=cfg["hidden_dim"],
            noise_std=cfg["noise_std"],
            device=device if device is not None else "auto",
        )
        policy.nets.load_state_dict(ckpt["state_dict"])
        policy._trained = bool(ckpt.get("trained", True))
        return policy
