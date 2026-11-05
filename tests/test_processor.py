r"""Tests for neptune.model.processor."""

import pytest
import torch

from neptune.model.processor import MultiheadCrossAttention, Processor

N, C_S, C_O, C_F, D_D = 2, 4, 8, 3, 2
Z, Y, X = 2, 3, 4

CONFIG = {
    "input_states": N,
    "lat_channels_surface": C_S,
    "lat_channels_ocean": C_O,
    "lat_channels_era5": C_F,
    "shape_ocean": (Z, Y, X),
    "mod_features": 4,
    "date_features": D_D,
    "hid_channels": 16,
    "hid_blocks": 2,
    "attention_heads": 2,
    "rope": True,
}


def _inputs(batch: int) -> tuple[torch.Tensor, ...]:
    r"""Draw random latent states, ERA5 forcing and date for a tiny grid.

    Arguments:
        batch : Number of samples B.

    Returns:
        inputs : Previous surface and ocean latents, ERA5 latent and encoded date.
    """

    return (
        torch.randn(batch, N, C_S, Y, X),
        torch.randn(batch, N, C_O, Z, Y, X),
        torch.randn(batch, C_F, Y, X),
        torch.randn(batch, D_D),
    )


@pytest.fixture(scope="module", params=["cross", "joint"])
def processor(request: pytest.FixtureRequest) -> Processor:
    r"""Tiny processor of each variant, in evaluation mode."""

    torch.manual_seed(0)

    return Processor(variant=request.param, **CONFIG).eval()


def test_cross_attention_typical() -> None:
    r"""Determines if the cross-attention keeps the shape of the tokens for a longer context."""

    mca = MultiheadCrossAttention(16, attention_heads=4)
    y = mca(torch.randn(2, 5, 16), torch.randn(2, 7, 16))

    assert y.shape == (2, 5, 16)


def test_processor_tokens(processor: Processor) -> None:
    r"""Determines if the surface and ERA5 tokens lie above the ocean levels."""

    assert processor.tokens == [Y * X, Z * Y * X]
    assert (processor.positions[: Y * X, 0] == -1).all()
    assert (processor.positions[Y * X :, 0] >= 0).all()
    assert (processor.positions_era5[:, 0] == -2).all()


def test_processor_forward(processor: Processor) -> None:
    r"""Determines if the forecast has one latent state per member, differing between members."""

    with torch.no_grad():
        z_out_s, z_out_o = processor(*_inputs(2), members=3)

    assert z_out_s.shape == (2, 3, C_S, Y, X)
    assert z_out_o.shape == (2, 3, C_O, Z, Y, X)
    assert not torch.allclose(z_out_o[:, 0], z_out_o[:, 1])
    assert not torch.allclose(z_out_s[:, 0], z_out_s[:, 1])


def test_processor_single_member(processor: Processor) -> None:
    r"""Determines if a single member with the same noise is reproducible."""

    inputs = _inputs(1)
    with torch.no_grad():
        torch.manual_seed(1)
        z_out_s, z_out_o = processor(*inputs)
        torch.manual_seed(1)
        z_out_s_bis, z_out_o_bis = processor(*inputs)

    assert z_out_s.shape == (1, 1, C_S, Y, X)
    assert z_out_o.shape == (1, 1, C_O, Z, Y, X)
    assert torch.equal(z_out_s, z_out_s_bis)
    assert torch.equal(z_out_o, z_out_o_bis)


def test_processor_era5(processor: Processor) -> None:
    r"""Determines if the forecast changes with the ERA5 forcing, the noise being the same."""

    z_inp_s, z_inp_o, z_f, date = _inputs(1)
    with torch.no_grad():
        torch.manual_seed(1)
        _, z_out_o = processor(z_inp_s, z_inp_o, z_f, date)
        torch.manual_seed(1)
        _, z_out_o_bis = processor(z_inp_s, z_inp_o, z_f + 1, date)

    assert not torch.allclose(z_out_o, z_out_o_bis)


def test_processor_increments(processor: Processor) -> None:
    r"""Determines if the forecast is the last input state when the output projections are zero."""

    processor_zero = Processor(variant=processor.variant, **CONFIG)
    for layer in (processor_zero.out_surface, processor_zero.out_ocean):
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

    z_inp_s, z_inp_o, z_f, date = _inputs(1)
    with torch.no_grad():
        z_out_s, z_out_o = processor_zero(z_inp_s, z_inp_o, z_f, date, members=2)

    assert torch.equal(z_out_s[:, 1], z_inp_s[:, -1])
    assert torch.equal(z_out_o[:, 1], z_inp_o[:, -1])


def test_processor_backward(processor: Processor) -> None:
    r"""Determines if every parameter of the processor receives a gradient."""

    processor.zero_grad()
    z_out_s, z_out_o = processor(*_inputs(1), members=2)
    (z_out_s.mean() + z_out_o.mean()).backward()

    assert all(p.grad is not None for p in processor.parameters())
