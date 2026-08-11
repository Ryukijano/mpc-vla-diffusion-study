"""Flow Matching Policy (rectified flow).

This policy shares the same 1D temporal U-Net architecture as the DDPM
Diffusion Policy but is trained with a *flow matching* (rectified-flow)
objective instead of the DDPM noise-prediction objective.

Flow matching (Lipman et al., 2023; Liu et al., 2023) defines a deterministic
probability path between a simple source distribution (Gaussian noise) and the
data distribution via the linear interpolation::

    x_t = (1 - t) * noise + t * x_0,   t in [0, 1]

The corresponding velocity field is ``v(x_t, t) = x_0 - noise``.  Training
minimises the MSE between the network's prediction and this target velocity.

Sampling integrates the ODE ``dx/dt = v_theta(x_t, t)`` from ``t=0`` (noise)
to ``t=1`` (data) using a small number of Euler steps.  Because the path is
nearly straight (rectified), far fewer steps are needed than DDPM -- typically
10 steps suffice versus 100+ for DDPM, yielding **much faster inference**.

GPU support
-----------
The model auto-detects CUDA and moves itself to the GPU when available.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .conditional_unet1d import ConditionalUnet1D
from .noise_schedule import FlowSchedule


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowMatchingPolicy:
    """Flow-matching (rectified-flow) policy for action-sequence generation.

    Parameters
    ----------
    action_dim : dimensionality of each action in the sequence.
    horizon : number of actions in the generated sequence.
    obs_dim : dimensionality of the observation vector used for conditioning.
        Set to ``None`` for unconditional generation.
    num_flow_steps : number of Euler ODE steps used during sampling (typically
        10 is enough -- far fewer than DDPM).
    hidden_dim : base channel width of the U-Net.
    num_layers : number of U-Net encoder/decoder levels (depth).
    device : device to place the model on.  Auto-detected if ``None``.

    Attributes
    ----------
    net : the :class:`ConditionalUnet1D` velocity-prediction network.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        obs_dim: Optional[int] = None,
        num_flow_steps: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.obs_dim = int(obs_dim) if obs_dim is not None else None
        self.num_flow_steps = int(num_flow_steps)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.device = device or _default_device()

        down_dims = [self.hidden_dim * (2 ** i) for i in range(self.num_layers)]

        self.net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim,
            down_dims=down_dims,
        ).to(self.device)

        self.schedule = FlowSchedule(num_steps=self.num_flow_steps, device=self.device)
        self.schedule = self.schedule.to(self.device)
        self._trained = False

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _prepare_demos(
        self,
        demonstrations: Sequence[Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        for obs, act in demonstrations:
            obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
            act_arr = np.asarray(act, dtype=np.float32).reshape(self.horizon, self.action_dim)
            obs_list.append(obs_arr)
            act_list.append(act_arr)
        obs_tensor = torch.from_numpy(np.stack(obs_list)).to(self.device)
        act_tensor = torch.from_numpy(np.stack(act_list)).to(self.device)
        if self.obs_dim is not None:
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
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-4,
        verbose: bool = False,
    ) -> List[float]:
        """Train with the flow-matching (velocity-prediction) loss.

        Parameters
        ----------
        demonstrations : iterable of ``(obs, action_seq)`` where ``obs`` has
            shape ``(obs_dim,)`` and ``action_seq`` has shape
            ``(horizon, action_dim)``.
        epochs : number of passes over the data.
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

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for obs_b, act_b in loader:
                bs = act_b.shape[0]
                # Sample continuous timesteps in [0, 1).
                t = self.schedule.sample_t(bs)
                # Sample source noise.
                noise = torch.randn_like(act_b)
                # Interpolated state: x_t = (1-t) noise + t x_0
                x_t = self.schedule.add_noise(act_b, t, noise)
                # Target velocity: v = x_0 - noise
                t_expand = t.clone()
                while t_expand.dim() < act_b.dim():
                    t_expand = t_expand.unsqueeze(-1)
                target_v = act_b - noise
                # The U-Net expects integer-ish timesteps for its sinusoidal
                # embedding; we scale continuous t to [0, num_flow_steps).
                t_idx = (t * self.num_flow_steps).long().clamp(0, self.num_flow_steps - 1)
                pred_v = self.net(x_t, t_idx, global_cond=obs_b if self.obs_dim is not None else None)
                loss = F.mse_loss(pred_v, target_v)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"[Flow] epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}")

        self._trained = True
        return losses

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Generate action sequences by ODE integration from noise to data.

        Parameters
        ----------
        observation : observation vector of shape ``(obs_dim,)`` (or a batch
            ``(B, obs_dim)``).
        num_samples : number of action sequences to generate per observation.

        Returns
        -------
        actions : tensor of shape ``(num_samples, horizon, action_dim)``.
        """
        self.net.eval()
        obs = None
        if self.obs_dim is not None:
            obs_arr = np.asarray(observation, dtype=np.float32)
            if obs_arr.ndim == 1:
                obs_arr = np.broadcast_to(obs_arr, (num_samples, obs_arr.shape[0])).copy()
            else:
                obs_arr = np.repeat(obs_arr, num_samples, axis=0)
            obs = torch.from_numpy(obs_arr).to(self.device)

        batch = obs.shape[0] if obs is not None else num_samples
        # Start from pure noise (t = 0).
        x = torch.randn(batch, self.horizon, self.action_dim, device=self.device)

        dt = 1.0 / float(self.num_flow_steps)
        # Integrate from t = 0 to t = 1 with Euler steps.
        for i in range(self.num_flow_steps):
            t_cont = i * dt
            # Map continuous t to an integer index for the sinusoidal embedding.
            t_idx = torch.full((batch,), i, device=self.device, dtype=torch.long)
            pred_v = self.net(x, t_idx, global_cond=obs)
            x = x + dt * pred_v

        return x

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
                "num_flow_steps": self.num_flow_steps,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
            },
            "trained": self._trained,
        }, path)

    def load(self, path: str, map_location: Optional[torch.device] = None) -> "FlowMatchingPolicy":
        """Load weights from a ``.pt`` file (in-place)."""
        device = map_location or self.device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(device)
        self._trained = bool(ckpt.get("trained", True))
        return self

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> "FlowMatchingPolicy":
        """Construct a policy from a saved checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        policy = cls(
            action_dim=cfg["action_dim"],
            horizon=cfg["horizon"],
            obs_dim=cfg["obs_dim"],
            num_flow_steps=cfg["num_flow_steps"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            device=device,
        )
        policy.net.load_state_dict(ckpt["state_dict"])
        policy._trained = bool(ckpt.get("trained", True))
        return policy
