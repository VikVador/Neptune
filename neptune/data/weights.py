r"""Spatial masks and loss weights for the Black Sea dataset."""

__all__ = [
    "get_weights_mask",
    "get_weights_state_mask",
    "get_weights_loss",
    "get_weights_stats",
]

import numpy as np
import torch
import xarray as xr

from functools import cache
from torch import Tensor

from neptune.config import (
    PATH_MASK,
    PATH_STATS,
)
from neptune.data import (
    DATASET_REGION,
    DATASET_VARIABLES,
    DATASET_VARIABLES_SURFACE,
)


def _prepare(
    t: Tensor,
    dim: int,
    device: torch.device | str | None,
) -> Tensor:
    r"""Utility to prepare output tensors with the desired rank and device."""
    if dim not in {1, 2}:
        raise ValueError(f"ERROR - dim must be 1 or 2, got {dim!r}")
    if dim == 2:
        t = t.unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


def _depth_dim(var: str) -> str:
    r"""Return the depth dimension name corresponding to a given variable."""
    if var == "uo":
        return "depthu"
    if var == "vo":
        return "depthv"
    return "deptht"


@cache
def _mask_array() -> np.ndarray:
    r"""Load and cache the ocean mask array from the zarr store."""
    ds = xr.open_zarr(PATH_MASK)
    try:
        return ds.mask.isel(
            longitude=DATASET_REGION["x"],
            latitude=DATASET_REGION["y"],
            level=DATASET_REGION["depthu"],
        ).values
    finally:
        ds.close()


@cache
def _stats_arrays() -> tuple[list[float], list[float]]:
    r"""Load and cache per-channel mean and std from the statistics zarr store."""
    ds = xr.open_zarr(PATH_STATS)
    try:
        means: list[float] = []
        stds: list[float] = []
        for var in DATASET_VARIABLES:
            if var in DATASET_VARIABLES_SURFACE:
                means.append(float(ds[var].sel(statistic="mean")))
                stds.append(float(ds[var].sel(statistic="std")))
            else:
                depth = _depth_dim(var)
                m = ds[var].sel(statistic="mean").isel({depth: DATASET_REGION[depth]}).values
                s = ds[var].sel(statistic="std").isel({depth: DATASET_REGION[depth]}).values
                means.extend(m.tolist())
                stds.extend(s.tolist())
        return means, stds
    finally:
        ds.close()


def get_weights_mask(
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Load the Black Sea 3D ocean mask.

    Arguments:
        dim    : Output rank. 1 → (Z, Y, X), 2 → (1, Z, Y, X).
        device : Target device ("cpu" or "cuda").

    Returns:
        mask : Binary tensor with 1 on sea and 0 on land.
    """
    z_u = DATASET_REGION["depthu"].stop
    z_v = DATASET_REGION["depthv"].stop
    z_t = DATASET_REGION["deptht"].stop
    if not (z_u == z_v == z_t):
        raise ValueError(f"Depth slices must match: got depthu={z_u}, depthv={z_v}, deptht={z_t}")

    mask = _mask_array()
    return _prepare(torch.as_tensor(mask, dtype=torch.float32), dim, device)


def get_weights_state_mask(
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Build the ocean mask aligned with the dataset channel layout.

    Arguments:
        dim    : Output rank. 1 → (C, Y, X), 2 → (1, C, Y, X).
        device : Target device ("cpu" or "cuda").

    Returns:
        mask : Binary tensor with 1 on sea and 0 on land.
    """
    mask = get_weights_mask()

    channels = []
    for var in DATASET_VARIABLES:
        if var in DATASET_VARIABLES_SURFACE:
            channels.append(mask[0])
        else:
            channels.extend(mask.unbind(0))

    return _prepare(torch.stack(channels, dim=0), dim, device)


def get_weights_loss(
    *,
    dim: int = 1,
    scale: float = 1.0,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Build per-channel loss weights that compensate for depth-varying sea coverage.

    Arguments:
        dim    : Output rank. 1 → (C, 1, 1), 2 → (1, C, 1, 1).
        scale  : Optional multiplier for the loss weights.
        device : Target device ("cpu" or "cuda").

    Returns:
        weights : Tensor with one weight per channel.
    """
    mask = get_weights_mask()

    sum_total = mask.sum()
    sum_level = mask.sum(dim=(1, 2))
    weight_z = 1.0 - sum_level / sum_total
    weight_z = weight_z / weight_z.sum()

    weights = []
    for var in DATASET_VARIABLES:
        if var in DATASET_VARIABLES_SURFACE:
            weights.append(weight_z[0])
        else:
            weights.extend(weight_z.unbind(0))

    return _prepare(torch.stack(weights, dim=0)[:, None, None] * scale, dim, device)


def get_weights_stats(
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Load per-channel mean and standard deviation from the precomputed statistics.

    Arguments:
        dim    : Output rank. 1 → (C, 1, 1), 2 → (1, C, 1, 1).
        device : Target device ("cpu" or "cuda").

    Returns:
        mean : Per-channel mean.
        std  : Per-channel standard deviation.
    """
    means, stds = _stats_arrays()

    mean = torch.tensor(means, dtype=torch.float32)[:, None, None]
    std = torch.tensor(stds, dtype=torch.float32)[:, None, None]
    return _prepare(mean, dim, device), _prepare(std, dim, device)
