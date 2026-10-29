r"""Tests for neptune.model.encoders."""

import pytest
import torch

from neptune.data import DATASET_VARIABLES_OCEAN, DATASET_VARIABLES_SURFACE, X, Y, Z
from neptune.model.encoders import Encoder, stack_dates

CONFIG = {
    "lat_channels": 8,
    "hid_channels": [4, 8],
    "hid_blocks": [1, 1],
    "patch_size": 8,
    "stride": 2,
}


@pytest.mark.integration
def test_encoder_typical() -> None:
    r"""Determines if the surface and ocean encoders map their states and dates onto latent grids downsampled by 16."""

    surface, ocean = Encoder("surface", **CONFIG), Encoder("ocean", **CONFIG)
    dates = ["2000-01-01", "2000-07-01"]
    z_s = surface(stack_dates(torch.randn(2, len(DATASET_VARIABLES_SURFACE), Y, X), dates))
    z_o = ocean(stack_dates(torch.randn(2, len(DATASET_VARIABLES_OCEAN), Z, Y, X), dates))

    assert z_s.shape == (2, 8, *surface.shape) and surface.shape == [Y // 16, X // 16]
    assert z_o.shape == (2, 8, *ocean.shape) and ocean.shape == [Z // 16, Y // 16, X // 16]
    assert ocean.tokens == (Z // 16) * (Y // 16) * (X // 16)
    assert surface.encoder.mix.out_channels == len(DATASET_VARIABLES_SURFACE)
    assert ocean.encoder.mix.out_channels == len(DATASET_VARIABLES_OCEAN)


@pytest.mark.integration
def test_encoder_wrong_group() -> None:
    r"""Determines if an unknown group of variables is rejected."""

    with pytest.raises(AssertionError, match="group"):
        Encoder("atmosphere", **CONFIG)


def test_stack_dates_typical() -> None:
    r"""Determines if the sin/cos encoding of each date is stacked after the channels of its state."""

    states = torch.zeros(2, 3, Y, X)
    x = stack_dates(states, ["2000-01-01", "2000-07-01"])

    assert x.shape == (2, 5, Y, X)
    assert torch.all(x[:, :3] == 0)
    assert x[0, 4, 0, 0] > 0.99 and x[1, 4, 0, 0] < -0.99
