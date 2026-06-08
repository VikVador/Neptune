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

from neptune.config import PATH_MASK, PATH_STATS
from neptune.data import (
    DATASET_REGION,
    DATASET_VARIABLES,
    DATASET_VARIABLES_OCEAN_BIO,
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
    range: tuple[float, float],
    depths: tuple[int, int],
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Build per-channel, per-location loss weights combining two components.

    Weights:
        Column-depth weight (Y, X): Shallow coastal columns receive higher weights.
        Vertical weight (Z): Constant from the surface down to a depth index then linearly decaying.

    Arguments:
        dim    : Output rank. 1 → (C, Y, X), 2 → (1, C, Y, X).
        range  : Interval of output weights.
        depths : Depth-level indices at which the linear decay begins.
        device : Target device ("cpu" or "cuda").

    Returns:
        weights : Per-channel, per-location weight tensor.
    """

    range_min, range_max = range
    dpt_phys, dpt_bio = depths
    mask = get_weights_mask()
    mask_state = get_weights_state_mask()
    z_dim = mask.shape[0]

    # Column-depth weight (Y, X)
    w_col = range_min + (1.0 - mask.sum(dim=0).float() / z_dim) * (range_max - range_min)

    # Vertical weight (Z,)
    def _vertical(dpt_cutoff: int) -> Tensor:
        r"""Build the vertical weight component for a given cutoff depth index."""
        w = torch.full((z_dim,), range_max)
        if dpt_cutoff < z_dim - 1:
            n_decay = z_dim - 1 - dpt_cutoff
            decay = torch.linspace(0.0, 1.0, n_decay + 1)[1:]  # (n_decay,)
            w[dpt_cutoff + 1 :] = range_max - decay * (range_max - range_min)
        return w  # (Z,)

    w_vert_phy = _vertical(dpt_phys)
    w_vert_bio = _vertical(dpt_bio)

    # Assembling weights
    channels = []
    for var in DATASET_VARIABLES:
        if var in DATASET_VARIABLES_SURFACE:
            channels.append((w_col + range_max) * 0.5)
        else:
            w_vert = w_vert_bio if var in DATASET_VARIABLES_OCEAN_BIO else w_vert_phy
            for w_z in w_vert.unbind(0):
                channels.append((w_col + w_z) * 0.5)

    weights = torch.stack(channels, dim=0) * mask_state
    return _prepare(weights, dim, device)


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
