"""DDPM Diffusion Policy (Chi et al., RSS 2023).

A full PyTorch implementation of the Diffusion Policy baseline.  The policy
learns to map an *observation* to an *action sequence* of length ``horizon``
by learning to denoise Gaussian noise conditioned on the observation, using a
1D temporal U-Net (:class:`~diffusion_baselines.conditional_unet1d.ConditionalUnet1D`)
as the denoising network and a DDPM noise schedule (Ho et al., 2020).

Training
--------
For each demonstration ``(obs, action_seq)`` we sample a random diffusion
timestep ``t``, corrupt the action sequence with Gaussian noise according to
the forward process, and train the network to predict the noise ``epsilon``
that was added (the standard DDPM simplified objective).

Sampling
--------
Starting from pure Gaussian noise we iteratively apply the learned reverse
process for ``num_diffusion_steps`` steps, conditioned on the observation,
to produce an action sequence.

GPU support
-----------
The model auto-detects CUDA and moves itself to the GPU when available.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .conditional_unet1d import ConditionalUnet1D
from .noise_schedule import LinearSchedule, NoiseSchedule


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionPolicy:
    """DDPM Diffusion Policy for action-sequence generation.

    Parameters
    ----------
    action_dim : dimensionality of each action in the sequence.
    horizon : number of actions in the generated sequence.
    obs_dim : dimensionality of the observation (state) vector used for
        conditioning.  Set to ``None`` for unconditional generation.
    num_diffusion_steps : number of DDPM denoising steps ``T``.
    hidden_dim : base channel width of the U-Net (propagated to ``down_dims``
        as ``[hidden_dim, hidden_dim*2, hidden_dim*4]``).
    num_layers : number of U-Net encoder/decoder levels (depth).
    noise_schedule : optional :class:`NoiseSchedule` instance.  Defaults to a
        :class:`LinearSchedule` with ``num_diffusion_steps`` steps.
    device : device to place the model on.  Auto-detected if ``None``.

    Attributes
    ----------
    net : the :class:`ConditionalUnet1D` denoising network.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        obs_dim: Optional[int] = None,
        num_diffusion_steps: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        noise_schedule: Optional[NoiseSchedule] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.obs_dim = int(obs_dim) if obs_dim is not None else None
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.device = device or _default_device()

        # Build the channel widths for each U-Net level.
        down_dims = [self.hidden_dim * (2 ** i) for i in range(self.num_layers)]

        self.net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim,
            down_dims=down_dims,
        ).to(self.device)

        self.schedule = noise_schedule or LinearSchedule(
            num_steps=self.num_diffusion_steps, device=self.device
        )
        self.schedule = self.schedule.to(self.device)
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
        """Train the diffusion policy on ``(observation, action_sequence)`` pairs.

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
                # Sample random timesteps.
                t = self.schedule.sample_t(bs)
                # Sample noise.
                noise = torch.randn_like(act_b)
                # Corrupt the action sequence.
                x_t = self.schedule.add_noise(act_b, t, noise)
                # Predict the noise.
                pred = self.net(x_t, t, global_cond=obs_b if self.obs_dim is not None else None)
                loss = F.mse_loss(pred, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"[DDPM] epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}")

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
        """Generate action sequences by iterative denoising.

        Parameters
        ----------
        observation : observation vector of shape ``(obs_dim,)`` (or a batch
            ``(B, obs_dim)``).  Ignored if the policy was built with
            ``obs_dim=None``.
        num_samples : number of action sequences to generate per observation.

        Returns
        -------
        actions : tensor of shape ``(num_samples, horizon, action_dim)`` (or
            ``(B, num_samples, horizon, action_dim)`` if a batch of
            observations is given).
        """
        self.net.eval()
        obs = None
        if self.obs_dim is not None:
            obs_arr = np.asarray(observation, dtype=np.float32)
            if obs_arr.ndim == 1:
                obs_arr = np.broadcast_to(obs_arr, (num_samples, obs_arr.shape[0])).copy()
            else:
                # batch of observations -> replicate each num_samples times
                obs_arr = np.repeat(obs_arr, num_samples, axis=0)
            obs = torch.from_numpy(obs_arr).to(self.device)

        batch = obs.shape[0] if obs is not None else num_samples
        x = torch.randn(batch, self.horizon, self.action_dim, device=self.device)

        # Reverse process: iterate from t = T-1 down to 0.
        for t_val in reversed(range(self.num_diffusion_steps)):
            t = torch.full((batch,), t_val, device=self.device, dtype=torch.long)
            pred = self.net(x, t, global_cond=obs)
            x = self.schedule.step(x, t, pred)

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
                "num_diffusion_steps": self.num_diffusion_steps,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
            },
            "trained": self._trained,
        }, path)

    def load(self, path: str, map_location: Optional[torch.device] = None) -> "DiffusionPolicy":
        """Load weights from a ``.pt`` file (in-place)."""
        device = map_location or self.device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(device)
        self._trained = bool(ckpt.get("trained", True))
        return self

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> "DiffusionPolicy":
        """Construct a policy from a saved checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        policy = cls(
            action_dim=cfg["action_dim"],
            horizon=cfg["horizon"],
            obs_dim=cfg["obs_dim"],
            num_diffusion_steps=cfg["num_diffusion_steps"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            device=device,
        )
        policy.net.load_state_dict(ckpt["state_dict"])
        policy._trained = bool(ckpt.get("trained", True))
        return policy
