r"""Deterministic encoders of the surface and ocean states into latent grids."""

__all__ = [
    "Encoder",
]

import math
import torch
import torch.nn as nn

from shaggy.models.cae import ConvEncoder
from torch import Tensor
from typing import Any

from neptune.data import (
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    X,
    Y,
    Z,
)
from neptune.data.weights import get_weights_mesh

# fmt: off
#
# ----- Variables and spatial shape of each group
#
GROUPS = {
    "surface" : (DATASET_VARIABLES_SURFACE, (Y, X)),
    "ocean"   : (DATASET_VARIABLES_OCEAN,   (Z, Y, X)),
}
# fmt: on


class Encoder(nn.Module):
    r"""Creates the deterministic encoder of a group of variables, its states stacked with the mesh of its grid.

    Arguments:
        group        : Group of variables, 'surface' or 'ocean'.
        lat_channels : Number of latent channels C_z.
        config       : Keyword arguments of the ConvEncoder (hid_channels, stride, ...).
    """

    def __init__(
        self,
        group: str,
        lat_channels: int,
        **config: Any,
    ) -> None:
        super().__init__()

        # Security
        assert group in GROUPS, f"ERROR - group must be 'surface' or 'ocean', got {group}."

        self.group = group
        self.lat_channels = lat_channels
        variables, shape = GROUPS[group]

        # The sin/cos mesh is rebuilt at loading (not saved)
        mesh_surface, mesh_ocean = get_weights_mesh()
        static = mesh_surface if group == "surface" else mesh_ocean
        self.register_buffer("static", static, persistent=False)

        # The date is not an input: two identical states would otherwise get different latents
        self.encoder = ConvEncoder(
            len(variables) + len(static),
            lat_channels,
            spatial=len(shape),
            mod_features=None,
            **config,
        )

        # Each spatial dimension of the latent grid is divided by the scale of the encoder
        self.shape = [n // s for n, s in zip(shape, self.encoder.scale, strict=True)]
        self.tokens = math.prod(self.shape)

    def forward(self, x: Tensor) -> Tensor:
        r"""Encode states into latent grids, the first convolution in float32 whatever the autocast.

        Arguments:
            x : States (B, C, [Z,] Y, X).

        Returns:
            z : Latent grids (B, C_z, [Z',] Y', X').
        """

        x = torch.cat([x, self.static.expand(len(x), *self.static.shape)], dim=1)

        # The daily changes of the deep levels are below the precision of bfloat16, so the inputs are read in float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = self.encoder.in_proj(self.encoder.patch(x.float()))

        h = self.encoder.descent(h)

        return self.encoder.out_proj(self.encoder.out_norm(h))
