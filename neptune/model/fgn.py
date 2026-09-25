r"""Functional Generative Network (FGN)."""

__all__ = [
    "FGN",
]

import torch
import torch.nn as nn

from azula.nn.dit import DiT
from collections.abc import Sequence
from shaggy.models.cae import create_ConvAE
from torch import Tensor
from typing import Any

from neptune.data import (
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    X,
    Y,
    Z,
)
from neptune.data.weights import (
    get_weights_bounds,
    get_weights_increments,
    get_weights_mask,
    get_weights_mesh,
)


def _positions(shape: Sequence[int]) -> Tensor:
    r"""Compute the integer coordinates of every token of a latent grid.

    Arguments:
        shape : Spatial shape of the latent grid (L_1, ..., L_S).

    Returns:
        positions : Coordinates of each token, flattened in the order of Tensor.flatten (T, S).
    """

    positions = torch.cartesian_prod(*[torch.arange(n, dtype=torch.float32) for n in shape])

    return positions.reshape(-1, len(shape))


def _stack_inputs(x_inp: Tensor, cond: Tensor, mask: Tensor, mesh: Tensor) -> Tensor:
    r"""Stack the previous states, the land/sea mask, the mesh and the conditioning along channels.

    Arguments:
        x_inp : Previous states (B, N, C, L_1, ..., L_S).
        cond  : Conditioning (B, C_c, L_1, ..., L_S).
        mask  : Land/sea mask (1, L_1, ..., L_S).
        mesh  : Sin/cos mesh (C_m, L_1, ..., L_S).

    Returns:
        x : Encoder inputs (B, N * C + 1 + C_m + C_c, L_1, ..., L_S).
    """

    static = torch.cat([mask, mesh]).expand(len(x_inp), -1, *mask.shape[1:])

    return torch.cat([x_inp.flatten(1, 2), static, cond], dim=1)


class FGN(nn.Module):
    r"""Creates a Functional Generative Network (FGN) forecasting the next surface and ocean state.

    References:
        | Skillful joint probabilistic weather forecasting from marginals (Alet et al., 2025)
        | https://arxiv.org/abs/2506.10772

        | WeatherNext 3: Increasing resolution and performance of global weather models with raw
        | observations (Rasp et al., 2026)
        | https://arxiv.org/abs/2609.03582

    Arguments:
        input_states          : Number of previous states N given as input.
        cond_channels_surface : Number of surface conditioning channels C_cs.
        cond_channels_ocean   : Number of ocean conditioning channels C_co.
        lat_channels          : Number of latent channels, shared by surface and ocean latents.
        mod_features          : Dimension D of the modulation noise vector.
        config_surface        : Keyword arguments of the surface ConvAE (hid_channels, ...).
        config_ocean          : Keyword arguments of the ocean ConvAE (hid_channels, ...).
        config_processor      : Keyword arguments of the DiT processor (hid_channels, ...).
    """

    def __init__(
        self,
        input_states: int = 1,
        cond_channels_surface: int = 2,
        cond_channels_ocean: int = 2,
        lat_channels: int = 64,
        mod_features: int = 32,
        config_surface: dict[str, Any] | None = None,
        config_ocean: dict[str, Any] | None = None,
        config_processor: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.input_states = input_states
        self.lat_channels = lat_channels
        self.mod_features = mod_features

        # Static inputs | Land/sea masks and sin/cos meshes, rebuilt at loading (not saved)
        mask_surface, mask_ocean = get_weights_mask()
        mesh_surface, mesh_ocean = get_weights_mesh()
        self.register_buffer("mask_surface", mask_surface, persistent=False)
        self.register_buffer("mask_ocean", mask_ocean, persistent=False)
        self.register_buffer("mesh_surface", mesh_surface, persistent=False)
        self.register_buffer("mesh_ocean", mesh_ocean, persistent=False)

        # Increments | Statistics of the daily increments (standardized units), rebuilt at loading
        mean_surface, std_surface, mean_ocean, std_ocean = get_weights_increments()
        self.register_buffer("increments_mean_surface", mean_surface, persistent=False)
        self.register_buffer("increments_std_surface", std_surface, persistent=False)
        self.register_buffer("increments_mean_ocean", mean_ocean, persistent=False)
        self.register_buffer("increments_std_ocean", std_ocean, persistent=False)

        # Bounds | Physical bounds of the standardized states, rebuilt at loading
        lower_surface, upper_surface, lower_ocean, upper_ocean = get_weights_bounds()
        self.register_buffer("lower_surface", lower_surface, persistent=False)
        self.register_buffer("upper_surface", upper_surface, persistent=False)
        self.register_buffer("lower_ocean", lower_ocean, persistent=False)
        self.register_buffer("upper_ocean", upper_ocean, persistent=False)

        # Encoders and decoders | Inputs are the previous states, mask, mesh and conditioning
        channels_surface = len(DATASET_VARIABLES_SURFACE)
        channels_ocean = len(DATASET_VARIABLES_OCEAN)
        in_surface = (
            input_states * channels_surface + 1 + len(mesh_surface) + cond_channels_surface
        )
        in_ocean = input_states * channels_ocean + 1 + len(mesh_ocean) + cond_channels_ocean

        self.ae_surface = create_ConvAE(
            in_channels=in_surface,
            out_channels=channels_surface,
            lat_channels=lat_channels,
            spatial=2,
            mod_features=mod_features,
            **(config_surface or {}),
        )

        self.ae_ocean = create_ConvAE(
            in_channels=in_ocean,
            out_channels=channels_ocean,
            lat_channels=lat_channels,
            spatial=3,
            mod_features=mod_features,
            **(config_ocean or {}),
        )

        # Processor | Each token is positioned by its 3 coordinates (level, latitude, longitude)
        self.processor = DiT(
            in_channels=lat_channels,
            out_channels=lat_channels,
            mod_features=mod_features,
            pos_channels=3,
            **(config_processor or {}),
        )

        # Token positions (level, latitude, longitude), surface tokens lying at level -1
        _, *self.shape_surface = self.ae_surface.latent((Y, X))
        _, *self.shape_ocean = self.ae_ocean.latent((Z, Y, X))

        positions_surface = _positions((1, *self.shape_surface))
        positions_surface[:, 0] = -1
        positions_ocean = _positions(self.shape_ocean)

        self.tokens = [len(positions_surface), len(positions_ocean)]
        self.register_buffer(
            "positions",
            torch.cat([positions_surface, positions_ocean]),
            persistent=False,
        )

    def compression(self) -> tuple[float, float, float]:
        r"""Compute the compression factors of the states into latent tokens.

        Returns:
            surface : Size of a surface state (C_s, Y, X) over the size of its latent tokens.
            ocean   : Size of an ocean state (C_o, Z, Y, X) over the size of its latent tokens.
            total   : Size of both states over the size of all latent tokens.
        """

        size_surface = len(DATASET_VARIABLES_SURFACE) * Y * X
        size_ocean = len(DATASET_VARIABLES_OCEAN) * Z * Y * X
        latent_surface, latent_ocean = (self.lat_channels * tokens for tokens in self.tokens)

        return (
            size_surface / latent_surface,
            size_ocean / latent_ocean,
            (size_surface + size_ocean) / (latent_surface + latent_ocean),
        )

    def postprocess(self, x_s: Tensor, x_o: Tensor) -> tuple[Tensor, Tensor]:
        r"""Project standardized states onto their physical bounds.

        Arguments:
            x_s : Surface states (..., C_s, Y, X).
            x_o : Ocean states (..., C_o, Z, Y, X).

        Returns:
            x_s : Surface states within their bounds, same shape.
            x_o : Ocean states within their bounds, same shape.
        """

        x_s = x_s.clamp(self.lower_surface, self.upper_surface)
        x_o = x_o.clamp(self.lower_ocean, self.upper_ocean)

        return x_s, x_o

    def forward(
        self,
        x_inp_s: Tensor,
        x_inp_o: Tensor,
        cond_s: Tensor,
        cond_o: Tensor,
        members: int = 1,
    ) -> tuple[Tensor, Tensor]:
        r"""Forecast the next surface and ocean states, for each member of an ensemble.

        Arguments:
            x_inp_s : Previous surface states (B, N, C_s, Y, X).
            x_inp_o : Previous ocean states (B, N, C_o, Z, Y, X).
            cond_s  : Surface conditioning (B, C_cs, Y, X).
            cond_o  : Ocean conditioning (B, C_co, Z, Y, X).
            members : Number of ensemble members E, each with its own noise vector.

        Returns:
            x_out_s : Next surface states (B, E, C_s, Y, X).
            x_out_o : Next ocean states (B, E, C_o, Z, Y, X).
        """

        batch = len(x_inp_s)

        # Repeating the inputs for each member, (B, ...) → (B * E, ...)
        x_inp_s, x_inp_o, cond_s, cond_o = (
            t.repeat_interleave(members, dim=0) for t in (x_inp_s, x_inp_o, cond_s, cond_o)
        )

        # One noise vector per member, modulating every network
        noise = torch.randn(len(x_inp_s), self.mod_features, device=x_inp_s.device)

        # Encoding
        z_s = self.ae_surface.encode(
            _stack_inputs(x_inp_s, cond_s, self.mask_surface, self.mesh_surface), noise
        )

        z_o = self.ae_ocean.encode(
            _stack_inputs(x_inp_o, cond_o, self.mask_ocean, self.mesh_ocean), noise
        )

        # Processing | Surface and ocean latents as a single sequence of tokens (B * E, T, C)
        z = torch.cat([z_s.flatten(2), z_o.flatten(2)], dim=2).transpose(1, 2)
        z = self.processor(z, noise, pos=self.positions)
        z_s, z_o = z.transpose(1, 2).split(self.tokens, dim=2)

        # Decoding | Standardized increments, rescaled to the increments of the states
        dx_s = self.ae_surface.decode(z_s.unflatten(2, self.shape_surface), noise)
        dx_o = self.ae_ocean.decode(z_o.unflatten(2, self.shape_ocean), noise)
        dx_s = self.increments_mean_surface + self.increments_std_surface * dx_s
        dx_o = self.increments_mean_ocean + self.increments_std_ocean * dx_o

        # Next state | Last input state plus the increment, within physical bounds and zero on land
        x_out_s, x_out_o = self.postprocess(x_inp_s[:, -1] + dx_s, x_inp_o[:, -1] + dx_o)
        x_out_s = x_out_s * self.mask_surface
        x_out_o = x_out_o * self.mask_ocean

        return x_out_s.unflatten(0, (batch, members)), x_out_o.unflatten(0, (batch, members))
