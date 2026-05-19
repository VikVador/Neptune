r"""Tests for neptune.data.weights."""

import pytest
import torch

from neptune.data import DATASET_VARIABLES_SURFACE, C, X, Y, Z
from neptune.data.weights import (
    _depth_dim,
    _prepare,
    get_weights_loss,
    get_weights_mask,
    get_weights_state_mask,
    get_weights_stats,
)


@pytest.mark.integration
def test_get_weights_mask_shape() -> None:
    r"""Determines if the mask has the expected depth and spatial dimensions."""
    assert get_weights_mask().shape == (Z, Y, X)


@pytest.mark.integration
def test_get_weights_mask_dtype() -> None:
    r"""Determines if the mask is a float32 tensor."""
    assert get_weights_mask().dtype == torch.float32


@pytest.mark.integration
def test_get_weights_mask_binary() -> None:
    r"""Determines if the mask contains only 0 (land) and 1 (sea)."""
    assert set(get_weights_mask().unique().tolist()) <= {0.0, 1.0}


@pytest.mark.integration
def test_get_weights_mask_dim2() -> None:
    r"""Determines if dim=2 prepends a batch dimension."""
    assert get_weights_mask(dim=2).shape == (1, Z, Y, X)


@pytest.mark.integration
def test_get_weights_state_mask_shape() -> None:
    r"""Determines if the state mask has the expected channel count."""
    assert get_weights_state_mask().shape == (C, Y, X)


@pytest.mark.integration
def test_get_weights_state_mask_binary() -> None:
    r"""Determines if the state mask contains only 0 (land) and 1 (sea)."""
    assert set(get_weights_state_mask().unique().tolist()) <= {0.0, 1.0}


@pytest.mark.integration
def test_get_weights_state_mask_dim2() -> None:
    r"""Determines if dim=2 prepends a batch dimension."""
    assert get_weights_state_mask(dim=2).shape == (1, C, Y, X)


@pytest.mark.integration
def test_get_weights_loss_shape() -> None:
    r"""Determines if loss weights have the expected channel layout."""
    assert get_weights_loss().shape == (C, 1, 1)


@pytest.mark.integration
def test_get_weights_loss_range() -> None:
    r"""Determines if all loss weights are in [0, 1]."""
    weights = get_weights_loss()
    assert (weights >= 0).all() and (weights <= 1).all()


@pytest.mark.integration
def test_get_weights_loss_ocean_sum() -> None:
    r"""Determines if the Z depth weights of one ocean variable sum to 1."""
    weights = get_weights_loss()
    n_surface = len(DATASET_VARIABLES_SURFACE)
    ocean_weights = weights[n_surface : n_surface + Z, 0, 0]
    assert torch.isclose(ocean_weights.sum(), torch.tensor(1.0), atol=1e-5)


@pytest.mark.integration
def test_get_weights_loss_dim2() -> None:
    r"""Determines if dim=2 prepends a batch dimension."""
    assert get_weights_loss(dim=2).shape == (1, C, 1, 1)


@pytest.mark.integration
def test_get_weights_stats_shape() -> None:
    r"""Determines if mean and std have the expected channel layout."""
    mean, std = get_weights_stats()
    assert mean.shape == (C, 1, 1)
    assert std.shape == (C, 1, 1)


@pytest.mark.integration
def test_get_weights_stats_positive_std() -> None:
    r"""Determines if all standard deviations are strictly positive."""
    _, std = get_weights_stats()
    assert (std > 0).all()


@pytest.mark.integration
def test_get_weights_stats_roundtrip() -> None:
    r"""Determines if standardization followed by unstandardization is the identity."""
    mean, std = get_weights_stats()
    x = torch.randn(C, Y, X)
    assert torch.allclose((x - mean) / std * std + mean, x, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
def test_get_weights_stats_dim2() -> None:
    r"""Determines if dim=2 prepends a batch dimension to both tensors."""
    mean, std = get_weights_stats(dim=2)
    assert mean.shape == (1, C, 1, 1)
    assert std.shape == (1, C, 1, 1)


def test_prepare_rank() -> None:
    r"""Determines if _prepare correctly unsqueezes for dim=2."""
    t = torch.zeros(3, 8, 8)
    assert _prepare(t, dim=1, device=None).shape == (3, 8, 8)
    assert _prepare(t, dim=2, device=None).shape == (1, 3, 8, 8)


def test_depth_dim() -> None:
    r"""Determines if _depth_dim returns the correct dimension name per variable."""
    assert _depth_dim("uo") == "depthu"
    assert _depth_dim("vo") == "depthv"
    assert _depth_dim("votemper") == "deptht"
    assert _depth_dim("vosaline") == "deptht"
