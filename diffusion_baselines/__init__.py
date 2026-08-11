"""Diffusion baselines package.

This package provides self-contained PyTorch implementations of the policy
baselines used in the MPC-vs-VLA-vs-Diffusion study:

* :class:`ConditionalUnet1D` -- 1D temporal U-Net denoising network.
* :class:`DiffusionPolicy` -- DDPM Diffusion Policy (Chi et al., RSS 2023).
* :class:`FlowMatchingPolicy` -- Flow-matching / rectified-flow policy.
* :class:`RegressionPolicy` -- plain MLP regression baseline (RCP).
* :class:`IterativeRegressionPolicy` -- multi-step regression with noise
  injection (Minimal Iterative Policy / MIP when ``num_iterations=2``).

Noise schedules:

* :class:`NoiseSchedule` -- base interface.
* :class:`LinearSchedule` -- linear beta schedule (Ho et al., 2020).
* :class:`CosineSchedule` -- cosine beta schedule (Improved DDPM).
* :class:`FlowSchedule` -- flow-matching / rectified-flow schedule.
"""

from .conditional_unet1d import (
    ConditionalUnet1D,
    ConditionalResidualBlock1D,
    Conv1dBlock,
    Downsample1d,
    SinusoidalPosEmb,
    Upsample1d,
)
from .ddpm_policy import DiffusionPolicy
from .flow_matching_policy import FlowMatchingPolicy
from .iterative_regression_policy import IterativeRegressionPolicy
from .noise_schedule import (
    CosineSchedule,
    FlowSchedule,
    LinearSchedule,
    NoiseSchedule,
)
from .regression_policy import RegressionPolicy

__all__ = [
    # Network
    "ConditionalUnet1D",
    "ConditionalResidualBlock1D",
    "Conv1dBlock",
    "Downsample1d",
    "SinusoidalPosEmb",
    "Upsample1d",
    # Policies
    "DiffusionPolicy",
    "FlowMatchingPolicy",
    "RegressionPolicy",
    "IterativeRegressionPolicy",
    # Noise schedules
    "NoiseSchedule",
    "LinearSchedule",
    "CosineSchedule",
    "FlowSchedule",
]
