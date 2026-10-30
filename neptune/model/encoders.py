r"""Deterministic encoders of the surface and ocean states into latent grids."""

__all__ = [
    "Encoder",
    "stack_dates",
]

import math
import torch
import torch.nn as nn

from collections.abc import Sequence
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
from neptune.data.weights import get_weights_date, get_weights_mesh

# fmt: off
#
# ----- Variables and spatial shape of each group
#
GROUPS = {
    "surface" : (DATASET_VARIABLES_SURFACE, (Y, X)),
    "ocean"   : (DATASET_VARIABLES_OCEAN,   (Z, Y, X)),
}
# fmt: on


def stack_dates(states: Tensor, dates: Sequence[str]) -> Tensor:
    r"""Stack states with the sin/cos encoding of their date, the inputs of an encoder.

    Arguments:
        states : Surface (B, C_s, Y, X) or ocean (B, C_o, Z, Y, X) states.
        dates  : Date strings 'YYYY-MM-DD' of the states (B,).

    Returns:
        x : States and their date (B, C + 2, [Z,] Y, X).
    """

    encodings = [get_weights_date(date, dim=2, device=states.device) for date in dates]
    surface = states.ndim == 4
    date = torch.cat([date_s if surface else date_o for date_s, date_o in encodings])

    return torch.cat([states, date.to(states.dtype)], dim=1)


class Encoder(nn.Module):
    r"""Creates the deterministic encoder of a group of variables, its states and date stacked with the mesh of its grid.

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

        # The states, their date and the mesh are first mixed into as many channels as there are variables
        self.encoder = ConvEncoder(
            len(variables) + 2 + len(static),
            lat_channels,
            mix_channels=len(variables),
            spatial=len(shape),
            mod_features=None,
            **config,
        )

        # Each spatial dimension of the latent grid is divided by the scale of the encoder
        self.shape = [n // s for n, s in zip(shape, self.encoder.scale, strict=True)]
        self.tokens = math.prod(self.shape)

    def forward(self, x: Tensor) -> Tensor:
        r"""Encode states and their date into latent grids.

        Arguments:
            x : States and their date (B, C + 2, [Z,] Y, X).

        Returns:
            z : Latent grids (B, C_z, [Z',] Y', X').
        """
        return self.encoder(torch.cat([x, self.static.expand(len(x), *self.static.shape)], dim=1))
