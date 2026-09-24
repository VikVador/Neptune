r"""Spatial masks, mesh and standardization statistics for the Black Sea dataset."""

__all__ = [
    "get_weights_mask",
    "get_weights_mesh",
    "get_weights_stats",
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
    DATASET_VARIABLES_SURFACE,
    Z,
)


def _prepare(
    t: Tensor,
    dim: int,
    device: torch.device | str | None,
) -> Tensor:
    r"""Optionally add a batch dimension to a tensor and move it to a device.

    Arguments:
        t      : Tensor to prepare.
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device, or None to keep the current one.

    Returns:
        t : Prepared tensor.
    """
    if dim not in {1, 2}:
        raise ValueError(f"ERROR - dim must be 1 or 2, got {dim!r}")
    if dim == 2:
        t = t.unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


def _mask_dataarray() -> xr.DataArray:
    r"""Load the ocean mask given its coordinates from the zarr store.

    Returns:
        mask : Binary data array of shape (Z, Y, X).
    """
    ds = xr.open_zarr(PATH_MASK)
    try:
        return ds.mask.isel(
            longitude=DATASET_REGION["x"],
            latitude=DATASET_REGION["y"],
            level=DATASET_REGION["z"],
        ).load()
    finally:
        ds.close()


def _encode_sin_cos(values: Tensor) -> Tensor:
    r"""Rescale a coordinate to [0, π] over its range and encode it with sin/cos.

    Arguments:
        values : Coordinate values of any shape (*).

    Returns:
        encoding : Sine and cosine of the rescaled coordinate (2, *).
    """
    angle = torch.pi * (values - values.min()) / (values.max() - values.min())
    return torch.stack([angle.sin(), angle.cos()])


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
) -> tuple[Tensor, Tensor]:
    r"""Load the Black Sea land/sea masks for surface and ocean variables.

    Arguments:
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device ("cpu" or "cuda").

    Returns:
        mask_surface : Binary mask (1, Y, X), with 1 on sea and 0 on land.
        mask_ocean   : Binary mask (1, Z, Y, X), with 1 on sea and 0 on land.
    """
    mask_ocean = torch.as_tensor(_mask_dataarray().values, dtype=torch.float32)[None]
    mask_surface = mask_ocean[:, 0].clone()
    return _prepare(mask_surface, dim, device), _prepare(mask_ocean, dim, device)


def get_weights_mesh(
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Build the sin/cos encoded mesh of the surface and ocean variables.

    Arguments:
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device ("cpu" or "cuda").

    Returns:
        mesh_surface, mesh_ocean : (sin, cos) tensors of dimensions (4, Y, X) and (6, Z, Y, X).
    """
    mask = _mask_dataarray()
    depth, latitude, longitude = torch.meshgrid(
        torch.tensor(mask.level.values, dtype=torch.float32).log(),
        torch.tensor(mask.latitude.values, dtype=torch.float32),
        torch.tensor(mask.longitude.values, dtype=torch.float32),
        indexing="ij",
    )

    mesh_ocean = torch.cat([
        _encode_sin_cos(latitude),
        _encode_sin_cos(longitude),
        _encode_sin_cos(depth),
    ])
    mesh_surface = mesh_ocean[:4, 0].clone()
    return _prepare(mesh_surface, dim, device), _prepare(mesh_ocean, dim, device)


def get_weights_stats(
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Load the standardization statistics of the surface and ocean variables.

    Arguments:
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device ("cpu" or "cuda").

    Returns:
        mean_surface : Mean of each surface variable (C_s, 1, 1).
        std_surface  : Standard deviation of each surface variable (C_s, 1, 1).
        mean_ocean   : Mean of each ocean variable, per level (C_o, Z, 1, 1).
        std_ocean    : Standard deviation of each ocean variable, per level (C_o, Z, 1, 1).
    """
    means, stds = _stats_arrays()
    mean = torch.tensor(means, dtype=torch.float32)
    std = torch.tensor(stds, dtype=torch.float32)

    # Surface variables come first, followed by the Z levels of each ocean variable
    n_surface = len(DATASET_VARIABLES_SURFACE)
    mean_surface, mean_ocean = mean[:n_surface, None, None], mean[n_surface:].reshape(-1, Z, 1, 1)
    std_surface, std_ocean = std[:n_surface, None, None], std[n_surface:].reshape(-1, Z, 1, 1)

    return (
        _prepare(mean_surface, dim, device),
        _prepare(std_surface, dim, device),
        _prepare(mean_ocean, dim, device),
        _prepare(std_ocean, dim, device),
    )
