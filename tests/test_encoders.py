r"""Tests for neptune.model.encoders."""

import pytest
import torch

from neptune.data import (
    DATASET_CONDITIONING,
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    X,
    Y,
    Z,
)
from neptune.model.encoders import OceanEncoder, SurfaceEncoder

CONFIG = {
    "lat_channels": 8,
    "hid_channels": [4, 8],
    "hid_blocks": [1, 1],
    "patch_size": 8,
    "stride": 2,
}


@pytest.mark.integration
def test_encoders_typical() -> None:
    r"""Determines if the surface and ocean encoders map their states onto latent grids downsampled by 16."""

    surface, ocean = SurfaceEncoder(**CONFIG), OceanEncoder(**CONFIG)
    z_s = surface(
        torch.randn(2, len(DATASET_VARIABLES_SURFACE) + 2 + len(DATASET_CONDITIONING), Y, X)
    )
    z_o = ocean(torch.randn(2, len(DATASET_VARIABLES_OCEAN) + 2, Z, Y, X))

    assert z_s.shape == (2, 8, *surface.shape) and surface.shape == [Y // 16, X // 16]
    assert z_o.shape == (2, 8, *ocean.shape) and ocean.shape == [Z // 16, Y // 16, X // 16]
    assert ocean.tokens == (Z // 16) * (Y // 16) * (X // 16)
