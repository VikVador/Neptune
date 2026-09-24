r"""Tests for neptune.data.weights."""

import pytest
import torch

from neptune.data import (
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    X,
    Y,
    Z,
)
from neptune.data.weights import (
    _encode_sin_cos,
    _prepare,
    get_weights_date,
    get_weights_mask,
    get_weights_mesh,
    get_weights_stats,
)


@pytest.mark.integration
def test_get_weights_mask_shape() -> None:
    r"""Determines if the surface and ocean masks have the expected dimensions."""
    mask_surface, mask_ocean = get_weights_mask()
    assert mask_surface.shape == (1, Y, X)
    assert mask_ocean.shape == (1, Z, Y, X)


@pytest.mark.integration
def test_get_weights_mask_binary() -> None:
    r"""Determines if both masks are float32 and contain only 0 (land) and 1 (sea)."""
    for mask in get_weights_mask():
        assert mask.dtype == torch.float32
        assert set(mask.unique().tolist()) <= {0.0, 1.0}


@pytest.mark.integration
def test_get_weights_mask_surface_is_top_level() -> None:
    r"""Determines if the surface mask is the top level of the ocean mask."""
    mask_surface, mask_ocean = get_weights_mask()
    assert torch.equal(mask_surface, mask_ocean[:, 0])


@pytest.mark.integration
def test_get_weights_mask_dim2() -> None:
    r"""Determines if dim=2 prepends a batch dimension to both masks."""
    mask_surface, mask_ocean = get_weights_mask(dim=2)
    assert mask_surface.shape == (1, 1, Y, X)
    assert mask_ocean.shape == (1, 1, Z, Y, X)


@pytest.mark.integration
def test_get_weights_mesh_shape() -> None:
    r"""Determines if the surface and ocean meshes have the expected dimensions."""
    mesh_surface, mesh_ocean = get_weights_mesh()
    assert mesh_surface.shape == (4, Y, X)
    assert mesh_ocean.shape == (6, Z, Y, X)


@pytest.mark.integration
def test_get_weights_mesh_structure() -> None:
    r"""Determines if latitude varies along Y only, longitude along X only and depth along Z only."""
    mesh_surface, mesh_ocean = get_weights_mesh()
    latitude, longitude, depth = mesh_ocean[0], mesh_ocean[2], mesh_ocean[4]
    assert torch.equal(mesh_surface, mesh_ocean[:4, 0])
    assert torch.equal(latitude, latitude[:1, :, :1].expand(Z, Y, X))
    assert torch.equal(longitude, longitude[:1, :1, :].expand(Z, Y, X))
    assert torch.equal(depth, depth[:, :1, :1].expand(Z, Y, X))


def test_encode_sin_cos_typical() -> None:
    r"""Determines if the encoding maps the range to [0, π] and lies on the unit circle."""
    sin, cos = _encode_sin_cos(torch.tensor([10.0, 15.0, 20.0]))
    assert torch.allclose(sin, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(cos, torch.tensor([1.0, 0.0, -1.0]), atol=1e-6)
    assert torch.allclose(sin.square() + cos.square(), torch.ones(3))


def test_encode_sin_cos_shape() -> None:
    r"""Determines if the encoding prepends a (sin, cos) dimension to any input shape."""
    assert _encode_sin_cos(torch.rand(3, 4, 5)).shape == (2, 3, 4, 5)


def test_get_weights_date_shape() -> None:
    r"""Determines if the date encoding is broadcast to the surface and ocean grids."""
    date_surface, date_ocean = get_weights_date("2000-06-15")
    assert date_surface.shape == (2, Y, X)
    assert date_ocean.shape == (2, Z, Y, X)
    assert torch.equal(date_ocean[:, 0], date_surface)


def test_get_weights_date_continuity() -> None:
    r"""Determines if the encoding is on the unit circle and continuous across the new year."""
    dec_31, _ = get_weights_date("1999-12-31")
    jan_01, _ = get_weights_date("2000-01-01")
    jul_01, _ = get_weights_date("2000-07-01")
    assert torch.allclose(dec_31[:, 0, 0].square().sum(), torch.tensor(1.0))
    assert (dec_31[:, 0, 0] - jan_01[:, 0, 0]).norm() < 0.05
    assert (jan_01[:, 0, 0] - jul_01[:, 0, 0]).norm() > 1.9


def test_get_weights_date_dim2() -> None:
    r"""Determines if dim=2 prepends a batch dimension to both encodings."""
    date_surface, date_ocean = get_weights_date("2000-06-15", dim=2)
    assert date_surface.shape == (1, 2, Y, X)
    assert date_ocean.shape == (1, 2, Z, Y, X)


@pytest.mark.integration
def test_get_weights_stats_shape() -> None:
    r"""Determines if the surface and ocean statistics have the expected dimensions."""
    mean_surface, std_surface, mean_ocean, std_ocean = get_weights_stats()
    assert mean_surface.shape == std_surface.shape == (len(DATASET_VARIABLES_SURFACE), 1, 1)
    assert mean_ocean.shape == std_ocean.shape == (len(DATASET_VARIABLES_OCEAN), Z, 1, 1)


@pytest.mark.integration
def test_get_weights_stats_positive_std() -> None:
    r"""Determines if all standard deviations are strictly positive."""
    _, std_surface, _, std_ocean = get_weights_stats()
    assert (std_surface > 0).all()
    assert (std_ocean > 0).all()


@pytest.mark.integration
def test_get_weights_stats_broadcast() -> None:
    r"""Determines if the statistics broadcast over surface (C_s, Y, X) and ocean (C_o, Z, Y, X) states."""
    mean_surface, std_surface, mean_ocean, std_ocean = get_weights_stats()
    x_s = torch.randn(len(DATASET_VARIABLES_SURFACE), Y, X)
    x_o = torch.randn(len(DATASET_VARIABLES_OCEAN), Z, Y, X)
    assert ((x_s - mean_surface) / std_surface).shape == x_s.shape
    assert ((x_o - mean_ocean) / std_ocean).shape == x_o.shape


@pytest.mark.integration
def test_get_weights_stats_dim2() -> None:
    r"""Determines if dim=2 prepends a batch dimension to every statistic."""
    mean_surface, std_surface, mean_ocean, std_ocean = get_weights_stats(dim=2)
    assert mean_surface.shape == std_surface.shape == (1, len(DATASET_VARIABLES_SURFACE), 1, 1)
    assert mean_ocean.shape == std_ocean.shape == (1, len(DATASET_VARIABLES_OCEAN), Z, 1, 1)


def test_prepare_rank() -> None:
    r"""Determines if _prepare correctly unsqueezes for dim=2."""
    t = torch.zeros(3, 8, 8)
    assert _prepare(t, dim=1, device=None).shape == (3, 8, 8)
    assert _prepare(t, dim=2, device=None).shape == (1, 3, 8, 8)
