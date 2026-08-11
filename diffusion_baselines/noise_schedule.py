"""Noise schedules for diffusion and flow-matching policies.

This module provides three schedules used by the diffusion baselines:

* :class:`LinearSchedule`  -- linear beta schedule (Ho et al., 2020).
* :class:`CosineSchedule`  -- cosine beta schedule (Improved DDPM,
  Nichol & Dhariwal, 2021).
* :class:`FlowSchedule`    -- flow-matching / rectified-flow schedule
  (Lipman et al., 2023; Liu et al., 2023).

Every schedule exposes a common interface so that the policy classes can
swap them transparently:

* ``add_noise(x, t, noise)``  -- forward process: corrupt ``x`` to ``x_t``.
* ``compute_alpha(t)``        -- cumulative signal-retention coefficient
  ``alpha_bar`` at timestep ``t`` (DDPM) or interpolation weight (flow).
* ``step(x_t, t, pred)``      -- one reverse-process update producing
  ``x_{t-1}`` (DDPM) or an ODE step (flow).
* ``sample_t(batch_size, ...)``-- sample random timesteps for training.

All tensors are kept on the same device as the input data.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class NoiseSchedule:
    """Base class for noise schedules.

    Subclasses must populate ``betas``, ``alphas`` and ``alpha_bar`` (cumulative
    product of ``alphas``) as 1-D float tensors of length ``num_steps``.
    """

    def __init__(self, num_steps: int, device: torch.device | str = "cpu") -> None:
        self.num_steps = int(num_steps)
        self.device = torch.device(device)
        # Filled in by subclasses.
        self.betas: torch.Tensor = torch.empty(0, device=self.device)
        self.alphas: torch.Tensor = torch.empty(0, device=self.device)
        self.alpha_bar: torch.Tensor = torch.empty(0, device=self.device)

    # -- helpers ------------------------------------------------------------
    def to(self, device: torch.device | str) -> "NoiseSchedule":
        self.device = torch.device(device)
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_bar = self.alpha_bar.to(self.device)
        return self

    def _t_index(self, t: torch.Tensor) -> torch.Tensor:
        """Clamp/round a timestep tensor to valid integer indices."""
        t = torch.as_tensor(t, device=self.device, dtype=torch.long)
        t = t.clamp(0, self.num_steps - 1)
        return t

    # -- common API (overridden where needed) -------------------------------
    def sample_t(self, batch_size: int) -> torch.Tensor:
        """Sample uniform random timestep indices in ``[0, num_steps)``."""
        return torch.randint(0, self.num_steps, (batch_size,), device=self.device)

    def compute_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``alpha_bar`` at timestep ``t`` (shape ``(B,)`` or scalar)."""
        idx = self._t_index(t)
        return self.alpha_bar[idx]

    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward (noising) process: ``x_t = sqrt(a_bar) x + sqrt(1-a_bar) noise``."""
        idx = self._t_index(t)
        sqrt_ab = torch.sqrt(self.alpha_bar[idx]).to(x.dtype)
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bar[idx]).to(x.dtype)
        while sqrt_ab.dim() < x.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_one_minus_ab = sqrt_one_minus_ab.unsqueeze(-1)
        return sqrt_ab * x + sqrt_one_minus_ab * noise

    def step(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """One reverse-process step.

        For DDPM schedules ``pred`` is the predicted noise ``epsilon`` and this
        returns ``x_{t-1}``.  For flow schedules ``pred`` is the predicted
        velocity and this performs one Euler ODE step.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Linear beta schedule (Ho et al., 2020)
# ---------------------------------------------------------------------------
class LinearSchedule(NoiseSchedule):
    """Linear beta schedule as in Ho et al. (2020).

    Parameters
    ----------
    num_steps : number of diffusion timesteps ``T``.
    beta_start, beta_end : endpoints of the linear schedule.
    """

    def __init__(
        self,
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(num_steps, device=device)
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        # posterior variance q(x_{t-1}|x_t,x_0): beta_t * (1-a_bar_{t-1})/(1-a_bar_t)
        alpha_bar_prev = torch.cat([torch.ones(1, device=self.device), self.alpha_bar[:-1]])
        self.posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)
        # posterior mean coefficients
        self.posterior_mean_coef1 = torch.sqrt(self.posterior_var) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef2 = (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar) * torch.sqrt(1.0 - betas)

    def step(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """DDPM reverse step: ``x_{t-1} = mean + sigma * z``.

        ``pred`` is the predicted noise ``epsilon``.
        """
        idx = self._t_index(t)
        beta_t = self.betas[idx]
        sqrt_ab = torch.sqrt(self.alphas[idx]).to(x_t.dtype)
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bar[idx]).to(x_t.dtype)
        # Predicted x_0 from epsilon.
        coef = beta_t / sqrt_one_minus_ab
        while coef.dim() < x_t.dim():
            coef = coef.unsqueeze(-1)
            sqrt_ab = sqrt_ab.unsqueeze(-1)
        mean = (x_t - coef * pred) / sqrt_ab

        # Add stochastic noise (except at t=0).
        sigma = torch.sqrt(self.posterior_var[idx]).to(x_t.dtype)
        while sigma.dim() < x_t.dim():
            sigma = sigma.unsqueeze(-1)
        noise = torch.randn_like(x_t)
        # No noise at the final step (t == 0).
        not_last = (idx > 0).to(x_t.dtype)
        while not_last.dim() < x_t.dim():
            not_last = not_last.unsqueeze(-1)
        return mean + not_last * sigma * noise


# ---------------------------------------------------------------------------
# Cosine beta schedule (Improved DDPM, Nichol & Dhariwal, 2021)
# ---------------------------------------------------------------------------
class CosineSchedule(NoiseSchedule):
    """Cosine beta schedule from Improved DDPM (Nichol & Dhariwal, 2021).

    Parameters
    ----------
    num_steps : number of diffusion timesteps ``T``.
    s : small offset to prevent ``beta`` from being too small near ``t=0``.
    max_beta : clamp value for ``beta`` to keep the schedule stable.
    """

    def __init__(
        self,
        num_steps: int = 100,
        s: float = 0.008,
        max_beta: float = 0.999,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(num_steps, device=device)
        steps = num_steps + 1
        t = torch.linspace(0, num_steps, steps, device=self.device, dtype=torch.float32) / num_steps
        alpha_bar = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clamp(betas, max=max_beta)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=self.device), self.alpha_bar[:-1]])
        self.posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef1 = torch.sqrt(self.posterior_var) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef2 = (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar) * torch.sqrt(1.0 - betas)

    def step(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """DDPM reverse step (identical maths to :class:`LinearSchedule`)."""
        idx = self._t_index(t)
        beta_t = self.betas[idx]
        sqrt_ab = torch.sqrt(self.alphas[idx]).to(x_t.dtype)
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bar[idx]).to(x_t.dtype)
        coef = beta_t / sqrt_one_minus_ab
        while coef.dim() < x_t.dim():
            coef = coef.unsqueeze(-1)
            sqrt_ab = sqrt_ab.unsqueeze(-1)
        mean = (x_t - coef * pred) / sqrt_ab
        sigma = torch.sqrt(self.posterior_var[idx]).to(x_t.dtype)
        while sigma.dim() < x_t.dim():
            sigma = sigma.unsqueeze(-1)
        noise = torch.randn_like(x_t)
        not_last = (idx > 0).to(x_t.dtype)
        while not_last.dim() < x_t.dim():
            not_last = not_last.unsqueeze(-1)
        return mean + not_last * sigma * noise


# ---------------------------------------------------------------------------
# Flow-matching / rectified-flow schedule
# ---------------------------------------------------------------------------
class FlowSchedule(NoiseSchedule):
    """Flow-matching (rectified-flow) schedule.

    The forward process is a linear interpolation between noise (``t=0``) and
    data (``t=1``)::

        x_t = (1 - t) * noise + t * x_0

    The velocity field to be learned is ``v = x_0 - noise``.  Sampling
    integrates the ODE ``dx/dt = v_theta(x_t, t)`` from ``t=0`` to ``t=1``
    using ``num_steps`` Euler steps.

    Unlike the DDPM schedules, ``t`` here is a continuous value in ``[0, 1]``
    rather than an integer index.  ``num_steps`` controls the granularity of
    the ODE integrator used during sampling.
    """

    def __init__(self, num_steps: int = 10, device: torch.device | str = "cpu") -> None:
        super().__init__(num_steps, device=device)
        # Evenly spaced integration grid in (0, 1].
        self.t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device, dtype=torch.float32)
        # Placeholder tensors so the base interface is satisfied.
        self.betas = torch.zeros(num_steps, device=self.device)
        self.alphas = torch.ones(num_steps, device=self.device)
        self.alpha_bar = torch.ones(num_steps, device=self.device)

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """Sample continuous timesteps uniform in ``[0, 1)`` for training."""
        return torch.rand(batch_size, device=self.device)

    def compute_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Interpolation weight ``t`` (data fraction) at time ``t``."""
        return torch.as_tensor(t, device=self.device, dtype=torch.float32)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward interpolation: ``x_t = (1-t) noise + t x_0``."""
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        while t.dim() < x.dim():
            t = t.unsqueeze(-1)
        return (1.0 - t) * noise + t * x

    def step(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """One Euler ODE step: ``x_{t+dt} = x_t + dt * v_theta``.

        ``pred`` is the predicted velocity ``v = x_0 - noise``.
        ``t`` is the current continuous time and ``dt`` is inferred from the
        integration grid.
        """
        dt = 1.0 / float(self.num_steps)
        t = torch.as_tensor(t, device=x_t.device, dtype=x_t.dtype)
        while t.dim() < x_t.dim():
            t = t.unsqueeze(-1)
        return x_t + dt * pred
