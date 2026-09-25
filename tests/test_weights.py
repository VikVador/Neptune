r"""Tests for neptune.data.weights."""

import pytest
import torch
import xarray as xr

from neptune.config import PATH_STATS_INCREMENTS
from neptune.data import (
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    X,
    Y,
    Z,
)
from neptune.data.weights import (
    _constant_levels,
    _encode_sin_cos,
    get_weights_bounds,
    get_weights_date,
    get_weights_increments,
    get_weights_loss,
    get_weights_mask,
    get_weights_mesh,
    get_weights_stats,
)

C_S = len(DATASET_VARIABLES_SURFACE)
C_O = len(DATASET_VARIABLES_OCEAN)


def test_encode_sin_cos_typical() -> None:
    r"""Determines if a coordinate range is mapped to [0, π], on the unit circle."""

    sin, cos = _encode_sin_cos(torch.tensor([10.0, 15.0, 20.0]))

    assert torch.allclose(sin, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(cos, torch.tensor([1.0, 0.0, -1.0]), atol=1e-6)
    assert _encode_sin_cos(torch.rand(3, 4, 5)).shape == (2, 3, 4, 5)


def test_get_weights_date_typical() -> None:
    r"""Determines if the date encoding covers both grids and is continuous across the new year."""

    date_surface, date_ocean = get_weights_date("1999-12-31", dim=2)
    jan_01, _ = get_weights_date("2000-01-01")
    jul_01, _ = get_weights_date("2000-07-01")

    assert date_surface.shape == (1, 2, Y, X)
    assert date_ocean.shape == (1, 2, Z, Y, X)
    assert (date_surface[0, :, 0, 0] - jan_01[:, 0, 0]).norm() < 0.05
    assert (jan_01[:, 0, 0] - jul_01[:, 0, 0]).norm() > 1.9


@pytest.mark.integration
def test_get_weights_mask_typical() -> None:
    r"""Determines if the masks are binary, with the surface mask being the top ocean level."""

    mask_surface, mask_ocean = get_weights_mask()

    assert mask_surface.shape == (1, Y, X)
    assert mask_ocean.shape == (1, Z, Y, X)
    assert set(mask_ocean.unique().tolist()) == {0.0, 1.0}
    assert torch.equal(mask_surface, mask_ocean[:, 0])


@pytest.mark.integration
def test_get_weights_mesh_typical() -> None:
    r"""Determines if latitude varies along Y only, longitude along X only and depth along Z only."""

    mesh_surface, mesh_ocean = get_weights_mesh()
    latitude, longitude, depth = mesh_ocean[0], mesh_ocean[2], mesh_ocean[4]

    assert mesh_surface.shape == (4, Y, X)
    assert mesh_ocean.shape == (6, Z, Y, X)
    assert torch.equal(mesh_surface, mesh_ocean[:4, 0])
    assert torch.equal(latitude, latitude[:1, :, :1].expand(Z, Y, X))
    assert torch.equal(longitude, longitude[:1, :1, :].expand(Z, Y, X))
    assert torch.equal(depth, depth[:, :1, :1].expand(Z, Y, X))


@pytest.mark.integration
def test_get_weights_stats_typical() -> None:
    r"""Determines if the statistics broadcast over the surface and ocean states, with positive stds."""

    mean_surface, std_surface, mean_ocean, std_ocean = get_weights_stats()

    assert mean_surface.shape == std_surface.shape == (C_S, 1, 1)
    assert mean_ocean.shape == std_ocean.shape == (C_O, Z, 1, 1)
    assert (std_surface > 0).all() and (std_ocean > 0).all()


@pytest.mark.integration
def test_get_weights_increments_typical() -> None:
    r"""Determines if the increments statistics are the physical ones divided by the states std."""

    _, std_surface, _, std_ocean = get_weights_stats()
    _, std_inc_surface, _, std_inc_ocean = get_weights_increments()
    physical = xr.open_zarr(PATH_STATS_INCREMENTS)

    i_windsp = DATASET_VARIABLES_SURFACE.index("windsp")
    i_votemper = DATASET_VARIABLES_OCEAN.index("votemper")
    windsp = torch.tensor(float(physical["windsp"].sel(statistic="std")))
    votemper = torch.tensor(physical["votemper"].sel(statistic="std").values, dtype=torch.float32)

    assert std_inc_ocean.shape == (C_O, Z, 1, 1)
    assert torch.isclose(std_inc_surface[i_windsp] * std_surface[i_windsp], windsp)
    assert torch.allclose(
        std_inc_ocean[i_votemper] * std_ocean[i_votemper], votemper[:, None, None]
    )


@pytest.mark.integration
def test_get_weights_bounds_typical() -> None:
    r"""Determines if the bounds map 0 to the positive variables and pin constant levels to 0."""

    mean_surface, std_surface, _, _ = get_weights_stats()
    lower_surface, upper_surface, lower_ocean, upper_ocean = get_weights_bounds()
    _, constant_ocean = _constant_levels()

    i_windsp = DATASET_VARIABLES_SURFACE.index("windsp")
    windsp = lower_surface[i_windsp] * std_surface[i_windsp] + mean_surface[i_windsp]

    assert torch.allclose(windsp, torch.zeros(1, 1), atol=1e-5)
    assert (upper_surface == float("inf")).all()
    assert constant_ocean[DATASET_VARIABLES_OCEAN.index("DOX")].any()
    assert (lower_ocean[constant_ocean] == 0).all() and (upper_ocean[constant_ocean] == 0).all()


@pytest.mark.integration
def test_get_weights_loss_typical() -> None:
    r"""Determines if the loss averages the CRPS of normalized increments over non-constant levels."""

    _, constant_ocean = _constant_levels()
    _, std_inc_surface, _, std_inc_ocean = get_weights_increments()
    weights_surface, weights_ocean = get_weights_loss()

    assert weights_surface.shape == (C_S,)
    assert weights_ocean.shape == (C_O, Z)
    assert (weights_ocean[constant_ocean[..., 0, 0]] == 0).all()

    # Weights times the increments std are all equal to 1 / (number of non-constant levels)
    scaled = torch.cat([
        weights_surface * std_inc_surface[:, 0, 0],
        (weights_ocean * std_inc_ocean[..., 0, 0]).flatten(),
    ])
    scaled = scaled[scaled > 0]

    assert torch.allclose(scaled, torch.full_like(scaled, 1 / len(scaled)))
