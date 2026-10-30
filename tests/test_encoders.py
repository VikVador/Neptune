r"""Tests for neptune.model.encoders."""

import pytest
import torch

from neptune.data import DATASET_VARIABLES_OCEAN, DATASET_VARIABLES_SURFACE, X, Y, Z
from neptune.data.weights import get_weights_distance
from neptune.model.encoders import Encoder

CONFIG = {
    "lat_channels": 8,
    "hid_channels": [4, 8],
    "hid_blocks": [1, 1],
    "patch_size": 8,
    "stride": 2,
}


@pytest.mark.integration
def test_encoder_typical() -> None:
    r"""Determines if the surface and ocean encoders map their states onto latent grids downsampled by 16."""

    surface, ocean = Encoder("surface", **CONFIG), Encoder("ocean", **CONFIG)
    z_s = surface(torch.randn(2, len(DATASET_VARIABLES_SURFACE), Y, X))
    z_o = ocean(torch.randn(2, len(DATASET_VARIABLES_OCEAN), Z, Y, X))

    assert z_s.shape == (2, 8, *surface.shape) and surface.shape == [Y // 16, X // 16]
    assert z_o.shape == (2, 8, *ocean.shape) and ocean.shape == [Z // 16, Y // 16, X // 16]
    assert ocean.tokens == (Z // 16) * (Y // 16) * (X // 16)
    assert isinstance(ocean.encoder.mix, torch.nn.Identity)


@pytest.mark.integration
def test_encoder_float32_inputs() -> None:
    r"""Determines if the first convolution reads the inputs in float32 under a bfloat16 autocast."""

    surface = Encoder("surface", **CONFIG)
    seen = []
    surface.encoder.in_proj.register_forward_hook(
        lambda module, inputs, output: seen.append(output.dtype)
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = surface(torch.randn(2, len(DATASET_VARIABLES_SURFACE), Y, X))

    assert seen == [torch.float32]
    assert z.dtype == torch.bfloat16


@pytest.mark.integration
def test_encoder_wrong_group() -> None:
    r"""Determines if an unknown group of variables is rejected."""

    with pytest.raises(AssertionError, match="group"):
        Encoder("atmosphere", **CONFIG)


@pytest.mark.integration
def test_get_weights_distance_typical() -> None:
    r"""Determines if the weights are positive, and zero exactly on the constant levels."""

    weights_surface, weights_ocean = get_weights_distance()

    assert weights_surface.shape == (len(DATASET_VARIABLES_SURFACE), 1, 1)
    assert weights_ocean.shape == (len(DATASET_VARIABLES_OCEAN), Z, 1, 1)
    assert (weights_surface > 0).all() and (weights_ocean >= 0).all()
    assert weights_ocean[DATASET_VARIABLES_OCEAN.index("votemper")].min() > 1
