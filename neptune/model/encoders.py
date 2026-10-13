r"""Deterministic encoders of the surface and ocean states into latent grids."""

__all__ = [
    "LatentEncoder",
    "SurfaceEncoder",
    "OceanEncoder",
]

import math
import torch
import torch.nn as nn

from collections.abc import Sequence
from shaggy.models.cae import ConvEncoder
from torch import Tensor
from typing import Any

from neptune.data import (
    DATASET_CONDITIONING,
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    X,
    Y,
    Z,
)
from neptune.data.weights import get_weights_mask, get_weights_mesh


class LatentEncoder(nn.Module):
    r"""Creates a deterministic encoder of states and their conditioning, stacked with static inputs, into a latent grid.

    Arguments:
        static       : Land/sea mask and sin/cos mesh (1 + C_m, L_1, ..., L_S).
        in_channels  : Number of channels of the states and their conditioning C_i.
        resolution   : Spatial shape of the states (L_1, ..., L_S).
        lat_channels : Number of latent channels C_z.
        config       : Keyword arguments of the ConvEncoder (hid_channels, ...).
    """

    def __init__(
        self,
        static: Tensor,
        in_channels: int,
        resolution: Sequence[int],
        lat_channels: int,
        **config: Any,
    ) -> None:
        super().__init__()

        self.lat_channels = lat_channels
        self.register_buffer("static", static, persistent=False)

        self.encoder = ConvEncoder(
            in_channels + len(static),
            lat_channels,
            spatial=len(resolution),
            mod_features=None,
            **config,
        )

        self.shape = [n // s for n, s in zip(resolution, self.encoder.scale, strict=True)]
        self.tokens = math.prod(self.shape)

    def forward(self, x: Tensor) -> Tensor:
        r"""Encode states and their conditioning into latent grids.

        Arguments:
            x : States and their conditioning (B, C_i, L_1, ..., L_S).

        Returns:
            z : Latent grids (B, C_z, L_1', ..., L_S').
        """
        return self.encoder(torch.cat([x, self.static.expand(len(x), *self.static.shape)], dim=1))


class SurfaceEncoder(LatentEncoder):
    r"""Creates the deterministic encoder of the surface states, conditioned on the date and ERA5.

    Arguments:
        lat_channels : Number of latent channels C_z.
        config       : Keyword arguments of the ConvEncoder (hid_channels, ...).
    """

    def __init__(
        self,
        lat_channels: int,
        **config: Any,
    ) -> None:
        (mask, _), (mesh, _) = get_weights_mask(), get_weights_mesh()
        channels = len(DATASET_VARIABLES_SURFACE) + 2 + len(DATASET_CONDITIONING)
        super().__init__(torch.cat([mask, mesh]), channels, (Y, X), lat_channels, **config)


class OceanEncoder(LatentEncoder):
    r"""Creates the deterministic encoder of the ocean states, conditioned on the date.

    Arguments:
        lat_channels : Number of latent channels C_z.
        config       : Keyword arguments of the ConvEncoder (hid_channels, ...).
    """

    def __init__(
        self,
        lat_channels: int,
        **config: Any,
    ) -> None:
        (_, mask), (_, mesh) = get_weights_mask(), get_weights_mesh()
        channels = len(DATASET_VARIABLES_OCEAN) + 2
        super().__init__(torch.cat([mask, mesh]), channels, (Z, Y, X), lat_channels, **config)
