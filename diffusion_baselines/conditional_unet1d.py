"""1D temporal U-Net for Diffusion Policy (Chi et al., RSS 2023).

This is a self-contained PyTorch implementation of the
``ConditionalUnet1D`` denoising network used by Diffusion Policy.  It is a
faithful re-implementation of the architecture from the original
``diffusion_policy`` reference codebase, with the conv1d building blocks and
sinusoidal positional embedding inlined so the module has no external
dependencies beyond ``torch`` and ``einops``.

Architecture
------------
* **Encoder-decoder with skip connections** (U-Net over the temporal axis).
* **Sinusoidal timestep embedding** -- the diffusion step index is mapped to a
  Fourier feature vector and passed through a small MLP.
* **Global conditioning via FiLM** (Perez et al., 2018) -- the observation
  feature vector is concatenated with the timestep embedding and used to
  predict a per-channel bias (and optionally a scale) that modulates every
  residual block.
* **Configurable depth and channel width** via ``down_dims``.

The network operates on sequences of shape ``(B, horizon, input_dim)`` and
returns predictions of the same shape -- i.e. it predicts either the added
noise ``epsilon`` (DDPM) or the velocity field (flow matching) for every
element of the action sequence.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal positional embedding for scalar inputs.

    Parameters
    ----------
    dim : embedding dimension (should be even).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] != self.dim:  # pad if dim is odd
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class Downsample1d(nn.Module):
    """Strided 1D convolution that halves the temporal length."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Transposed 1D convolution that doubles the temporal length."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish activation block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        # Guard against n_groups larger than the channel count (common for
        # small test networks).
        n_groups = min(n_groups, out_channels)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual 1D conv block with FiLM conditioning.

    The conditioning vector (timestep + global features) is projected to a
    per-channel bias (and optionally a scale) that modulates the hidden
    activations -- this is the FiLM modulation of Perez et al. (2018).

    Parameters
    ----------
    in_channels, out_channels : channel counts of the input / output.
    cond_dim : dimensionality of the conditioning vector.
    kernel_size : convolution kernel size.
    n_groups : number of GroupNorm groups.
    cond_predict_scale : if ``True`` the FiLM encoder predicts both a scale
        and a bias (doubling its output width); otherwise only a bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        # Make the residual path dimension-compatible.
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : ``[batch, in_channels, horizon]``
        cond : ``[batch, cond_dim]``

        Returns
        -------
        out : ``[batch, out_channels, horizon]``
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# ---------------------------------------------------------------------------
# Conditional 1D U-Net
# ---------------------------------------------------------------------------
class ConditionalUnet1D(nn.Module):
    """1D temporal U-Net denoising network for Diffusion Policy.

    Parameters
    ----------
    input_dim : dimensionality of each element in the input sequence
        (e.g. the action dimension).
    local_cond_dim : optional dimensionality of a *per-timestep* local
        conditioning signal.  ``None`` disables local conditioning.
    global_cond_dim : optional dimensionality of a global conditioning
        vector (e.g. the observation features).  ``None`` disables global
        conditioning.
    diffusion_step_embed_dim : width of the sinusoidal timestep embedding.
    down_dims : list of channel widths for each level of the encoder
        (the decoder mirrors these).  Controls depth and capacity.
    kernel_size : convolution kernel size.
    n_groups : number of GroupNorm groups.
    cond_predict_scale : if ``True`` FiLM predicts both scale and bias.

    Forward signature
    -----------------
    ``forward(sample, timestep, local_cond=None, global_cond=None)``

    * ``sample``      : ``(B, horizon, input_dim)`` -- noisy action sequence.
    * ``timestep``    : ``(B,)`` tensor or scalar int -- diffusion step.
    * ``local_cond``  : ``(B, horizon, local_cond_dim)`` or ``None``.
    * ``global_cond`` : ``(B, global_cond_dim)`` or ``None``.

    Returns ``(B, horizon, input_dim)`` -- the predicted noise / velocity.
    """

    def __init__(
        self,
        input_dim: int,
        local_cond_dim: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.local_cond_dim = local_cond_dim
        self.global_cond_dim = global_cond_dim

        all_dims: List[int] = [self.input_dim] + list(down_dims)
        start_dim = int(down_dims[0])

        dsed = int(diffusion_step_embed_dim)
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += int(global_cond_dim)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Local conditioning encoder (two residual blocks producing skip
        # features that are added at the first encoder and last decoder
        # levels).
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = int(local_cond_dim)
            local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, self.input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Predict noise / velocity for a noisy action sequence.

        Parameters
        ----------
        sample : ``(B, horizon, input_dim)`` noisy action sequence.
        timestep : ``(B,)`` tensor or scalar -- diffusion step index.
        local_cond : ``(B, horizon, local_cond_dim)`` per-timestep conditioning.
        global_cond : ``(B, global_cond_dim)`` global conditioning vector.

        Returns
        -------
        ``(B, horizon, input_dim)`` prediction.
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. Timestep embedding -> global feature.
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # 2. Encode local features (produces skip connections).
        h_local: List[torch.Tensor] = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        # 3. Encoder path.
        x = sample
        h: List[torch.Tensor] = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # 4. Mid blocks.
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # 5. Decoder path with skip connections.
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) - 1 and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x
