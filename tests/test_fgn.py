r"""Tests for neptune.model.fgn."""

import pytest
import torch

from pathlib import Path
from shaggy.tools import load, save

from neptune.data import (
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    X,
    Y,
    Z,
)
from neptune.model.fgn import FGN, _positions

C_S = len(DATASET_VARIABLES_SURFACE)
C_O = len(DATASET_VARIABLES_OCEAN)

CONFIG = {
    "input_states": 2,
    "cond_channels_surface": 2,
    "cond_channels_ocean": 3,
    "lat_channels": 8,
    "mod_features": 4,
    "config_surface": {"hid_channels": [4, 8], "hid_blocks": [1, 1], "patch_size": 8, "stride": 2},
    "config_ocean": {"hid_channels": [4, 8], "hid_blocks": [1, 1], "patch_size": 8, "stride": 2},
    "config_processor": {"hid_channels": 16, "hid_blocks": 1, "attention_heads": 2},
}


@pytest.fixture(scope="module")
def fgn() -> FGN:
    r"""Tiny FGN, downsampling every spatial dimension by 16."""
    return FGN(**CONFIG)


@pytest.fixture(scope="module")
def inputs() -> tuple[torch.Tensor, ...]:
    r"""Random previous states and conditioning, for a batch of 1."""

    torch.manual_seed(0)

    return (
        torch.randn(1, 2, C_S, Y, X),
        torch.randn(1, 2, C_O, Z, Y, X),
        torch.randn(1, 2, Y, X),
        torch.randn(1, 3, Z, Y, X),
    )


def test_positions_typical() -> None:
    r"""Determines if the token coordinates follow the order of Tensor.flatten."""

    positions = _positions((2, 3))
    assert positions.shape == (6, 2)
    assert positions.tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]


@pytest.mark.integration
def test_fgn_tokens(fgn: FGN) -> None:
    r"""Determines if the latents are flattened into the expected tokens and compression factors."""

    surface, ocean, total = fgn.compression()

    assert fgn.shape_surface == [Y // 16, X // 16]
    assert fgn.shape_ocean == [Z // 16, Y // 16, X // 16]
    assert fgn.tokens == [(Y // 16) * (X // 16), (Z // 16) * (Y // 16) * (X // 16)]
    assert (fgn.positions[: fgn.tokens[0], 0] == -1).all()
    assert surface == C_S * 16**2 / 8
    assert ocean == C_O * 16**3 / 8
    assert total == (C_S * Y * X + C_O * Z * Y * X) / (8 * sum(fgn.tokens))


@pytest.mark.integration
def test_fgn_forward(fgn: FGN, inputs: tuple[torch.Tensor, ...]) -> None:
    r"""Determines if the forecast has one state per member, zero on land, differing between members."""

    with torch.no_grad():
        x_out_s, x_out_o = fgn(*inputs, members=3)

    assert x_out_s.shape == (1, 3, C_S, Y, X)
    assert x_out_o.shape == (1, 3, C_O, Z, Y, X)
    assert (x_out_s * (1 - fgn.mask_surface) == 0).all()
    assert (x_out_o * (1 - fgn.mask_ocean) == 0).all()
    assert not torch.equal(x_out_o[:, 0], x_out_o[:, 1])


@pytest.mark.integration
def test_fgn_postprocess(fgn: FGN) -> None:
    r"""Determines if the states are clipped to their bounds, constants being set to 0."""

    x_s = torch.full((1, C_S, Y, X), -100.0)
    x_o = torch.full((1, C_O, Z, Y, X), -100.0)
    y_s, y_o = fgn.postprocess(x_s, x_o)

    assert torch.equal(y_s, torch.maximum(x_s, fgn.lower_surface))
    assert torch.equal(y_o, torch.maximum(x_o, fgn.lower_ocean))
    assert (y_o[:, fgn.upper_ocean[..., 0, 0] == 0] == 0).all()


@pytest.mark.integration
def test_fgn_backward(fgn: FGN, inputs: tuple[torch.Tensor, ...]) -> None:
    r"""Determines if every parameter of the encoders, processor and decoders receives a gradient."""

    fgn.zero_grad()
    x_out_s, x_out_o = fgn(*inputs, members=2)
    (x_out_s.mean() + x_out_o.mean()).backward()
    assert all(p.grad is not None for p in fgn.parameters())


@pytest.mark.integration
def test_fgn_save_load(fgn: FGN, inputs: tuple[torch.Tensor, ...], tmp_path: Path) -> None:
    r"""Determines if a saved FGN is reloaded with the same weights and forecasts."""

    save(fgn, CONFIG, tmp_path)
    fgn_loaded = load(tmp_path, FGN, device="cpu")

    with torch.no_grad():
        torch.manual_seed(1)
        x_out_s, _ = fgn.eval()(*inputs)
        torch.manual_seed(1)
        x_out_s_loaded, _ = fgn_loaded(*inputs)

    assert torch.equal(x_out_s, x_out_s_loaded)
