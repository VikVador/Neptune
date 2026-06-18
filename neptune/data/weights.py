r"""Spatial masks and loss weights for the Black Sea dataset."""

__all__ = [
    "get_weights_mask",
    "get_weights_state_mask",
    "get_weights_stats",
    "get_weights_loss",
]

import numpy as np
import torch
import xarray as xr

from torch import Tensor

from neptune.config import (
    PATH_MASK,
    PATH_STATS,
)
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


def _mask_array() -> np.ndarray:
    r"""Load the ocean mask array from the zarr store.

    Returns:
        mask : Binary numpy array of shape (Z, Y, X).
    """
    ds = xr.open_zarr(PATH_MASK)
    try:
        return ds.mask.isel(
            longitude=DATASET_REGION["x"],
            latitude=DATASET_REGION["y"],
            level=DATASET_REGION["z"],
        ).values
    finally:
        ds.close()


def _stats_arrays() -> tuple[list[float], list[float]]:
    r"""Load per-channel mean and std from the statistics zarr store.

    Returns:
        means : Per-channel mean values.
        stds  : Per-channel standard deviation values.
    """

    z_slice = DATASET_REGION["z"]
    ds = xr.open_zarr(PATH_STATS)
    try:
        means: list[float] = []
        stds: list[float] = []
        z_len = z_slice.stop - z_slice.start
        for var in DATASET_VARIABLES:
            if var in DATASET_VARIABLES_SURFACE:
                means.append(float(ds[var].sel(statistic="mean")))
                stds.append(float(ds[var].sel(statistic="std")))
            else:
                da_mean = ds[var].sel(statistic="mean")
                da_std = ds[var].sel(statistic="std")
                depth = next((d for d in da_mean.dims if d.startswith("depth")), None)
                if depth:
                    m = da_mean.isel({depth: z_slice}).values
                    s = da_std.isel({depth: z_slice}).values
                else:
                    # stats zarr has no depth dim for this var: replicate to match preprocess
                    m = np.full(z_len, float(da_mean))
                    s = np.full(z_len, float(da_std))
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


def get_weights_loss(
    *,
    dim: int = 1,
    depths: tuple[int, int],
    w_min: float = 0.1,
    scaling: float = 100.0,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Build per-channel, per-location loss weights with equal per-column contribution.

    Arguments:
        dim     : Output rank. 1 → (C, Y, X), 2 → (1, C, Y, X).
        depths  : Level indices at which the linear decay begins for ocean variables.
        w_min   : Minimum (unnormalised) weight at the deepest level of the decay zone.
        scaling : Global multiplier applied to the final weight tensor.
        device  : Target device ("cpu" or "cuda").

    Returns:
        weights : Per-channel, per-location weight tensor. Land pixels are zero.
    """

    dpt_phys, dpt_bio = depths

    mask = get_weights_mask()
    z_dim = mask.shape[0]
    col_depth = mask.sum(dim=0)

    def _column_weights(d_cutoff: int) -> Tensor:
        r"""Normalised per-column vertical weights (sum_z = 1)."""
        n_decay = (col_depth - 1 - d_cutoff).clamp(min=0).float()
        z_idx = torch.arange(z_dim, dtype=torch.float32).view(z_dim, 1, 1)
        z_above = (z_idx - d_cutoff).clamp(min=0)
        t = torch.where(n_decay > 0, z_above / n_decay.unsqueeze(0), torch.zeros_like(z_above))
        w_shape = (1.0 - (1.0 - w_min) * t.clamp(max=1.0)) * mask
        col_sum = w_shape.sum(dim=0, keepdim=True).clamp(min=1e-8)
        return w_shape / col_sum

    w_ocean_phy = _column_weights(dpt_phys)
    w_ocean_bio = _column_weights(dpt_bio)
    w_surface = w_ocean_phy[0]

    channels = []
    for var in DATASET_VARIABLES:
        if var in DATASET_VARIABLES_SURFACE:
            channels.append(w_surface)
        else:
            w_ocean = w_ocean_bio if var in DATASET_VARIABLES_OCEAN_BIO else w_ocean_phy
            channels.extend(w_ocean.unbind(0))

    return _prepare(scaling * torch.stack(channels, dim=0), dim, device)
