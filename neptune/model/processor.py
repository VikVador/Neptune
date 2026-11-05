r"""Latent processor, a stochastic transformer forecasting the next latent states."""

__all__ = [
    "DiTCrossBlock",
    "MultiheadCrossAttention",
    "Processor",
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from azula.nn.attention import RoPEProjection, apply_rope
from azula.nn.dit import DiTBlock
from azula.nn.layers import SineEncoding
from azula.nn.utils import checkpoint
from collections.abc import Sequence
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Any

from neptune.model.fgn import _positions


class MultiheadCrossAttention(nn.Module):
    r"""Creates a multi-head cross-attention layer, the tokens attending to context tokens.

    Arguments:
        channels        : Number of channels C of the tokens and of the context.
        pos_channels    : Number of positional channels P, used by RoPE only.
        attention_heads : Number of attention heads H.
        rope            : Whether to use rotary positional embedding (RoPE).
    """

    def __init__(
        self,
        channels: int,
        pos_channels: int = 3,
        attention_heads: int = 1,
        rope: bool = False,
    ) -> None:
        super().__init__()

        # Security
        assert channels % attention_heads == 0, "ERROR - channels % attention_heads must be 0."

        self.heads = attention_heads
        self.q_proj = nn.Linear(channels, channels)
        self.kv_proj = nn.Linear(channels, 2 * channels)
        self.y_proj = nn.Linear(channels, channels, bias=False)
        self.qk_norm = nn.RMSNorm(channels // attention_heads, elementwise_affine=False, eps=1e-5)
        self.theta_proj = RoPEProjection(pos_channels, channels // 2) if rope else None

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        pos: Tensor | None = None,
        pos_context: Tensor | None = None,
    ) -> Tensor:
        r"""Attend from the tokens to the context tokens.

        Arguments:
            x           : Tokens (*, L, C).
            context     : Context tokens (*, L_c, C).
            pos         : Positions of the tokens (*, L, P), used by RoPE only.
            pos_context : Positions of the context tokens (*, L_c, P), used by RoPE only.

        Returns:
            y : Output tokens (*, L, C).
        """

        q = rearrange(self.q_proj(x), "... L (H C) -> ... H L C", H=self.heads)
        k, v = rearrange(self.kv_proj(context), "... L (n H C) -> n ... H L C", n=2, H=self.heads)
        q, k = self.qk_norm(q), self.qk_norm(k)

        # The queries and the keys are rotated by the angles of their own positions
        if self.theta_proj is not None:
            theta_q = rearrange(self.theta_proj(pos), "... L (H C) -> ... H L C", H=self.heads)
            theta_k = rearrange(
                self.theta_proj(pos_context), "... L (H C) -> ... H L C", H=self.heads
            )
            q, _ = apply_rope(q, q, theta_q)
            k, _ = apply_rope(k, k, theta_k)

        y = F.scaled_dot_product_attention(q, k, v)

        return self.y_proj(rearrange(y, "... H L C -> ... L (H C)"))


class DiTCrossBlock(DiTBlock):
    r"""Creates a modulated DiT block, with a cross-attention between its self-attention and FFN.

    Arguments:
        channels        : Number of channels C of the tokens and of the context.
        mod_features    : Dimension D of the modulation vector.
        pos_channels    : Number of positional channels P.
        attention_heads : Number of attention heads H.
        rope            : Whether to use rotary positional embedding (RoPE).
        kwargs          : Keyword arguments of the DiTBlock (ffn_factor, checkpointing, ...).
    """

    def __init__(
        self,
        channels: int,
        mod_features: int,
        pos_channels: int = 3,
        attention_heads: int = 1,
        rope: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            channels,
            mod_features,
            pos_channels=pos_channels,
            attention_heads=attention_heads,
            rope=rope,
            **kwargs,
        )

        self.mca = MultiheadCrossAttention(channels, pos_channels, attention_heads, rope)

        # Ada-Norm Zero of the cross-attention, initialized as the one of the DiTBlock
        self.ada_cross = nn.Sequential(
            nn.Linear(mod_features, 3 * channels),
            Rearrange("... (n C) -> n ... 1 C", n=3),
        )
        self.ada_cross[0].weight.data.mul_(1e-2)
        self.ada_cross[0].bias.data.zero_()

    def _forward(
        self,
        x: Tensor,
        context: Tensor,
        mod: Tensor,
        pos: Tensor | None = None,
        pos_context: Tensor | None = None,
    ) -> Tensor:
        r"""Update the tokens by self-attention, cross-attention to the context, then FFN.

        Arguments:
            x           : Tokens (*, L, C).
            context     : Context tokens (*, L_c, C).
            mod         : Modulation vector (*, D).
            pos         : Positions of the tokens (*, L, P).
            pos_context : Positions of the context tokens (*, L_c, P).

        Returns:
            y : Output tokens (*, L, C).
        """

        a1, b1, c1, a2, b2, c2 = self.ada_zero(mod)
        a3, b3, c3 = self.ada_cross(mod)

        x = x + c1 * self.msa((a1 + 1) * self.norm(x) + b1, pos)
        x = x + c3 * self.mca((a3 + 1) * self.norm(x) + b3, self.norm(context), pos, pos_context)
        x = x + c2 * self.ffn((a2 + 1) * self.norm(x) + b2)

        return x

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        mod: Tensor,
        pos: Tensor | None = None,
        pos_context: Tensor | None = None,
    ) -> Tensor:
        r"""Update the tokens, with activation checkpointing if enabled.

        Arguments:
            x           : Tokens (*, L, C).
            context     : Context tokens (*, L_c, C).
            mod         : Modulation vector (*, D).
            pos         : Positions of the tokens (*, L, P).
            pos_context : Positions of the context tokens (*, L_c, P).

        Returns:
            y : Output tokens (*, L, C).
        """

        if self.checkpointing:
            return checkpoint(self._forward, reentrant=not self.training)(
                x, context, mod, pos, pos_context
            )

        return self._forward(x, context, mod, pos, pos_context)


class Processor(nn.Module):
    r"""Creates the latent processor, an FGN transformer forecasting the next latent states.

    Arguments:
        variant              : Architecture, 'cross' (two transformers) or 'joint' (one sequence).
        input_states         : Number of previous latent states N given as input.
        lat_channels_surface : Number of latent channels C_s of the surface.
        lat_channels_ocean   : Number of latent channels C_o of the ocean.
        lat_channels_era5    : Number of latent channels C_f of the ERA5 forcing.
        shape_ocean          : Latent grid of the ocean (Z', Y', X'), surface and ERA5 on (Y', X').
        mod_features         : Dimension D of the modulation noise vector.
        date_features        : Dimension D_d of the encoded date, appended to the noise vector.
        increments           : Whether to predict the increment from the last input state.
        hid_channels         : Number of hidden channels C_h of the tokens.
        hid_blocks           : Number of transformer blocks (in each transformer for 'cross').
        attention_heads      : Number of attention heads H.
        kwargs               : Keyword arguments of the blocks (ffn_factor, dropout, rope, ...).
    """

    def __init__(
        self,
        variant: str = "cross",
        input_states: int = 2,
        lat_channels_surface: int = 32,
        lat_channels_ocean: int = 256,
        lat_channels_era5: int = 32,
        shape_ocean: Sequence[int] = (6, 32, 72),
        mod_features: int = 32,
        date_features: int = 2,
        increments: bool = True,
        hid_channels: int = 1024,
        hid_blocks: int = 16,
        attention_heads: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Security
        if variant not in ("cross", "joint"):
            raise ValueError(f"ERROR - variant must be 'cross' or 'joint', got {variant}.")

        self.variant = variant
        self.mod_features = mod_features
        self.increments = increments
        self.shape_ocean = list(shape_ocean)
        self.shape_surface = self.shape_ocean[1:]

        # Each type of token has its own projection, its bias telling the types apart
        self.in_surface = nn.Linear(input_states * lat_channels_surface, hid_channels)
        self.in_ocean = nn.Linear(input_states * lat_channels_ocean, hid_channels)
        self.in_era5 = nn.Linear(lat_channels_era5, hid_channels)
        self.out_surface = nn.Linear(hid_channels, lat_channels_surface)
        self.out_ocean = nn.Linear(hid_channels, lat_channels_ocean)

        # Absolute positional embedding, as in the DiT of azula
        self.pos_embedding = nn.Sequential(
            SineEncoding(hid_channels, omega=1e2),
            Rearrange("... P C -> ... (P C)"),
            nn.Linear(3 * hid_channels, hid_channels, bias=False),
        )
        self.pos_embedding[-1].weight.data.mul_(1e-2)

        # 'cross': ERA5 tokens in their own transformer, read by the state blocks (cross-attention)
        # 'joint': all the tokens in a single sequence, with full self-attention in every block
        # In both, the noise and the date modulate every normalization layer
        block = DiTCrossBlock if variant == "cross" else DiTBlock
        config = {
            "mod_features": mod_features + date_features,
            "pos_channels": 3,
            "attention_heads": attention_heads,
            **kwargs,
        }

        self.blocks = nn.ModuleList([block(hid_channels, **config) for _ in range(hid_blocks)])
        if variant == "cross":
            self.blocks_era5 = nn.ModuleList([
                DiTBlock(hid_channels, **config) for _ in range(hid_blocks)
            ])

        self.out_norm = nn.RMSNorm(hid_channels, elementwise_affine=False, eps=1e-5)
        self.out_ada_zero = nn.Sequential(
            nn.Linear(mod_features + date_features, 2 * hid_channels),
            Rearrange("... (n C) -> n ... 1 C", n=2),
        )
        self.out_ada_zero[0].weight.data.mul_(1e-2)
        self.out_ada_zero[0].bias.data.zero_()

        # Token positions (level, latitude, longitude), surface at level -1, ERA5 above at level -2
        positions_surface = _positions((1, *self.shape_surface))
        positions_surface[:, 0] = -1
        positions_ocean = _positions(self.shape_ocean)
        positions_era5 = _positions((1, *self.shape_surface))
        positions_era5[:, 0] = -2

        self.tokens = [len(positions_surface), len(positions_ocean)]
        self.register_buffer(
            "positions",
            torch.cat([positions_surface, positions_ocean]),
            persistent=False,
        )
        self.register_buffer("positions_era5", positions_era5, persistent=False)

    def forward(
        self,
        z_inp_s: Tensor,
        z_inp_o: Tensor,
        z_f: Tensor,
        date: Tensor,
        members: int = 1,
    ) -> tuple[Tensor, Tensor]:
        r"""Forecast the next latent surface and ocean states, for each member of an ensemble.

        Arguments:
            z_inp_s : Previous latent surface states, oldest first (B, N, C_s, Y', X').
            z_inp_o : Previous latent ocean states, oldest first (B, N, C_o, Z', Y', X').
            z_f     : Latent ERA5 forcing of the forecasted day (B, C_f, Y', X').
            date    : Encoded date of the forecasted day, e.g. (sin, cos) of its year progress (B, D_d).
            members : Number of ensemble members E, each with its own noise vector.

        Returns:
            z_out_s : Next latent surface states (B, E, C_s, Y', X').
            z_out_o : Next latent ocean states (B, E, C_o, Z', Y', X').
        """

        batch = len(z_inp_s)

        # Repeating the inputs for each member, (B, ...) → (B * E, ...)
        z_inp_s, z_inp_o, z_f, date = (
            t.repeat_interleave(members, dim=0) for t in (z_inp_s, z_inp_o, z_f, date)
        )

        # One noise vector per member, joined to the date to modulate every block
        noise = torch.randn(len(z_inp_s), self.mod_features, device=z_inp_s.device)
        mod = torch.cat([noise, date], dim=-1)

        # The previous states of a token are stacked along its channels, (B * E, T, N * C)
        h = torch.cat(
            [
                self.in_surface(z_inp_s.flatten(1, 2).flatten(2).transpose(1, 2)),
                self.in_ocean(z_inp_o.flatten(1, 2).flatten(2).transpose(1, 2)),
            ],
            dim=1,
        )
        h = h + self.pos_embedding(self.positions)
        f = self.in_era5(z_f.flatten(2).transpose(1, 2)) + self.pos_embedding(self.positions_era5)

        # Each state block reads the ERA5 tokens output by the ERA5 block of the same depth
        if self.variant == "cross":
            for block_era5, block in zip(self.blocks_era5, self.blocks, strict=True):
                f = block_era5(f, mod, pos=self.positions_era5)
                h = block(h, f, mod, pos=self.positions, pos_context=self.positions_era5)

        # All the tokens attend to each other, the ERA5 tokens being dropped at the end
        else:
            h = torch.cat([h, f], dim=1)
            positions = torch.cat([self.positions, self.positions_era5])
            for block in self.blocks:
                h = block(h, mod, pos=positions)
            h = h[:, : sum(self.tokens)]

        a, b = self.out_ada_zero(mod)
        h_s, h_o = ((a + 1) * self.out_norm(h) + b).split(self.tokens, dim=1)
        z_out_s = self.out_surface(h_s).transpose(1, 2).unflatten(2, self.shape_surface)
        z_out_o = self.out_ocean(h_o).transpose(1, 2).unflatten(2, self.shape_ocean)

        # The increment is added to the last input state
        if self.increments:
            z_out_s = z_inp_s[:, -1] + z_out_s
            z_out_o = z_inp_o[:, -1] + z_out_o

        return z_out_s.unflatten(0, (batch, members)), z_out_o.unflatten(0, (batch, members))
