r"""Spatial masks, mesh, statistics and loss weights for the Black Sea dataset."""

__all__ = [
    "get_weights_mask",
    "get_weights_mesh",
    "get_weights_date",
    "get_weights_conditioning",
    "get_weights_stats",
    "get_weights_stats_conditioning",
    "get_weights_increments",
    "get_weights_bounds",
    "get_weights_loss",
]

import numpy as np
import torch
import xarray as xr

from collections.abc import Sequence
from datetime import date as Date
from pathlib import Path
from torch import Tensor

from neptune.config import (
    PATH_MASK,
    PATH_STATS,
    PATH_STATS_CONDITIONING,
    PATH_STATS_INCREMENTS,
)
from neptune.data import (
    DATASET_CONDITIONING,
    DATASET_REGION,
    DATASET_VARIABLES,
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    VARIABLES_CLIPPING,
    VARIABLES_CONDITIONING_STANDARDIZATION,
    VARIABLES_STANDARDIZATION,
    X,
    Y,
    Z,
)

# Increments standard deviation below which a variable-level is constant
STD_CONSTANT = 1e-7


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

    with xr.open_zarr(PATH_MASK) as ds:
        return ds.mask.isel(
            longitude=DATASET_REGION["x"],
            latitude=DATASET_REGION["y"],
            level=DATASET_REGION["z"],
        ).load()


def _encode_sin_cos(values: Tensor) -> Tensor:
    r"""Rescale a coordinate to [0, π] over its range and encode it with sin/cos.

    Arguments:
        values : Coordinate values of any shape (*).

    Returns:
        encoding : Sine and cosine of the rescaled coordinate (2, *).
    """

    angle = torch.pi * (values - values.min()) / (values.max() - values.min())

    return torch.stack([angle.sin(), angle.cos()])


def _stats_array(path: Path, var: str) -> np.ndarray:
    r"""Load the mean and standard deviation of one variable from a statistics zarr store.

    Arguments:
        path : Path to the statistics zarr store.
        var  : Name of the variable.

    Returns:
        stats : Mean and standard deviation, (2,) for a surface and (2, Z) for an ocean variable.
    """

    with xr.open_zarr(path) as ds:
        da = ds[var].sel(statistic=["mean", "std"])
        depth = next((d for d in da.dims if d.startswith("depth")), None)

        return (da if depth is None else da.isel({depth: DATASET_REGION["z"]})).values


def _stats_paths(
    standardization: str | None = None,
    conditioning: bool = False,
) -> dict[str, Path]:
    r"""Find the statistics store of each variable, given its standardization.

    Arguments:
        standardization : Standardization of every variable ('global', 'monthly' or 'daily').
        conditioning    : Whether to find the stores of the conditioning variables instead.

    Returns:
        paths : Path to the statistics zarr store of each variable.
    """

    variables, standardizations, paths = (
        (DATASET_CONDITIONING, VARIABLES_CONDITIONING_STANDARDIZATION, PATH_STATS_CONDITIONING)
        if conditioning
        else (DATASET_VARIABLES, VARIABLES_STANDARDIZATION, PATH_STATS)
    )
    default = standardizations["default"]

    return {var: paths[standardization or standardizations.get(var, default)] for var in variables}


def _stats_tensors(paths: dict[str, Path]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Load the mean and standard deviation of every variable, each from its own statistics store.

    Arguments:
        paths : Path to the statistics zarr store of each variable.

    Returns:
        mean_surface, std_surface : Surface statistics (C_s, 1, 1).
        mean_ocean, std_ocean     : Ocean statistics, per level (C_o, Z, 1, 1).
    """

    surface = np.stack([_stats_array(paths[var], var) for var in DATASET_VARIABLES_SURFACE])
    ocean = np.stack([_stats_array(paths[var], var) for var in DATASET_VARIABLES_OCEAN])
    surface = torch.tensor(surface, dtype=torch.float32)  # (C_s, 2)
    ocean = torch.tensor(ocean, dtype=torch.float32)  # (C_o, 2, Z)

    return (
        surface[:, 0, None, None],
        surface[:, 1, None, None],
        ocean[:, 0, :, None, None],
        ocean[:, 1, :, None, None],
    )


def _physical_bounds(variables: list[str]) -> tuple[Tensor, Tensor]:
    r"""Gather the physical bounds of variables, infinite when unbounded.

    Arguments:
        variables : Names of the variables.

    Returns:
        lower : Lower bound of each variable (C,).
        upper : Upper bound of each variable (C,).
    """

    bounds = [VARIABLES_CLIPPING.get(var, (None, None)) for var in variables]
    lower = torch.tensor([-torch.inf if lo is None else lo for lo, _ in bounds])
    upper = torch.tensor([torch.inf if hi is None else hi for _, hi in bounds])

    return lower, upper


def _constant_levels() -> tuple[Tensor, Tensor]:
    r"""Find the variable-levels whose values never change from one day to the next.

    Returns:
        constant_surface : Whether each surface variable is constant (C_s, 1, 1).
        constant_ocean   : Whether each ocean variable is constant, per level (C_o, Z, 1, 1).
    """

    paths = dict.fromkeys(DATASET_VARIABLES, PATH_STATS_INCREMENTS)
    _, std_surface, _, std_ocean = _stats_tensors(paths)

    return std_surface <= STD_CONSTANT, std_ocean <= STD_CONSTANT


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


def get_weights_date(
    date: str,
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Encode the year progress of a date with sin/cos, broadcast to the surface and ocean grids.

    Arguments:
        date   : Date string 'YYYY-MM-DD'.
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device ("cpu" or "cuda").

    Returns:
        date_surface : (sin, cos) of the year progress (2, Y, X).
        date_ocean   : (sin, cos) of the year progress (2, Z, Y, X).
    """

    progress = Date.fromisoformat(date).timetuple().tm_yday / 366
    angle = torch.tensor(2 * torch.pi * progress)
    encoding = torch.stack([angle.sin(), angle.cos()])

    date_surface = encoding[:, None, None].expand(2, Y, X)
    date_ocean = encoding[:, None, None, None].expand(2, Z, Y, X)

    return _prepare(date_surface, dim, device), _prepare(date_ocean, dim, device)


def get_weights_conditioning(
    dates: Sequence[str],
    c_inp: Tensor,
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Build the conditioning of the FGN, from the last input state x_t of each sample.

    Arguments:
        dates  : Date strings 'YYYY-MM-DD' of the last input states (B,).
        c_inp  : ERA5 conditioning of the last input states (B, C_c, Y, X).
        device : Target device ("cpu" or "cuda").

    Returns:
        cond_s : Surface conditioning, encoded date and ERA5 (B, 2 + C_c, Y, X).
        cond_o : Ocean conditioning, encoded date (B, 2, Z, Y, X).
    """

    encodings = [get_weights_date(date, dim=2, device=device) for date in dates]
    date_s = torch.cat([date_s for date_s, _ in encodings])
    date_o = torch.cat([date_o for _, date_o in encodings])

    return torch.cat([date_s, c_inp.to(date_s.device)], dim=1), date_o


def get_weights_stats(
    *,
    standardization: str | None = None,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Load the standardization statistics of the surface and ocean variables.

    Arguments:
        standardization : Standardization of every variable ('global', 'monthly', 'daily' or 'none')
        dim             : Use 2 to add a leading batch dimension, 1 otherwise.
        device          : Target device ("cpu" or "cuda").

    Returns:
        mean_surface : Mean of each surface variable (C_s, 1, 1).
        std_surface  : Standard deviation of each surface variable (C_s, 1, 1).
        mean_ocean   : Mean of each ocean variable, per level (C_o, Z, 1, 1).
        std_ocean    : Standard deviation of each ocean variable, per level (C_o, Z, 1, 1).
    """
    return tuple(_prepare(t, dim, device) for t in _stats_tensors(_stats_paths(standardization)))


def get_weights_stats_conditioning(
    *,
    standardization: str | None = None,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Load the standardization statistics of the conditioning variables.

    Arguments:
        standardization : Standardization of every variable ('global', 'monthly' or 'daily' or 'none')
        dim             : Use 2 to add a leading batch dimension, 1 otherwise.
        device          : Target device ("cpu" or "cuda").

    Returns:
        mean_conditioning : Mean of each conditioning variable (C_c, 1, 1).
        std_conditioning  : Standard deviation of each conditioning variable (C_c, 1, 1).
    """

    paths = _stats_paths(standardization, conditioning=True)
    stats = np.stack([_stats_array(paths[var], var) for var in DATASET_CONDITIONING])
    stats = torch.tensor(stats, dtype=torch.float32)  # (C_c, 2)
    mean, std = stats[:, 0, None, None], stats[:, 1, None, None]

    return _prepare(mean, dim, device), _prepare(std, dim, device)


def get_weights_increments(
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Load the mean and standard deviation of the daily increments, in standardized units.

    References:
        | WeatherNext 3: Increasing resolution and performance of global weather models with raw
        | observations (Rasp et al., 2026)
        | https://arxiv.org/abs/2609.03582

    Arguments:
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device ("cpu" or "cuda").

    Returns:
        mean_surface : Mean increment of each surface variable (C_s, 1, 1).
        std_surface  : Standard deviation of the increment of each surface variable (C_s, 1, 1).
        mean_ocean   : Mean increment of each ocean variable, per level (C_o, Z, 1, 1).
        std_ocean    : Standard deviation of the increment of each ocean variable (C_o, Z, 1, 1).
    """

    _, std_surface, _, std_ocean = _stats_tensors(_stats_paths())
    paths = dict.fromkeys(DATASET_VARIABLES, PATH_STATS_INCREMENTS)
    mean_inc_s, std_inc_s, mean_inc_o, std_inc_o = _stats_tensors(paths)

    return (
        _prepare(mean_inc_s / std_surface, dim, device),
        _prepare(std_inc_s / std_surface, dim, device),
        _prepare(mean_inc_o / std_ocean, dim, device),
        _prepare(std_inc_o / std_ocean, dim, device),
    )


def get_weights_bounds(
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Compute the physical bounds of the variables, per level and in standardized units.

    Arguments:
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device ("cpu" or "cuda").

    Returns:
        lower_surface, upper_surface : Bounds of each surface variable (C_s, 1, 1).
        lower_ocean, upper_ocean     : Bounds of each ocean variable, per level (C_o, Z, 1, 1).
    """

    mean_surface, std_surface, mean_ocean, std_ocean = _stats_tensors(_stats_paths())
    constant_surface, constant_ocean = _constant_levels()

    lower_surface, upper_surface = _physical_bounds(DATASET_VARIABLES_SURFACE)
    lower_ocean, upper_ocean = _physical_bounds(DATASET_VARIABLES_OCEAN)
    lower_surface, upper_surface = lower_surface[:, None, None], upper_surface[:, None, None]
    lower_ocean, upper_ocean = lower_ocean[:, None, None, None], upper_ocean[:, None, None, None]

    bounds = (
        torch.where(constant_surface, 0.0, (lower_surface - mean_surface) / std_surface),
        torch.where(constant_surface, 0.0, (upper_surface - mean_surface) / std_surface),
        torch.where(constant_ocean, 0.0, (lower_ocean - mean_ocean) / std_ocean),
        torch.where(constant_ocean, 0.0, (upper_ocean - mean_ocean) / std_ocean),
    )

    return tuple(_prepare(t, dim, device) for t in bounds)


def get_weights_loss(
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Compute the weight of each surface variable and each ocean variable-level in the loss.

    Arguments:
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device ("cpu" or "cuda").

    Returns:
        weights_surface : Weight of each surface variable (C_s,).
        weights_ocean   : Weight of each ocean variable, per level (C_o, Z).
    """

    constant_surface, constant_ocean = _constant_levels()
    _, std_inc_surface, _, std_inc_ocean = get_weights_increments()

    weights_surface = torch.where(constant_surface, 0.0, 1 / std_inc_surface)[:, 0, 0]
    weights_ocean = torch.where(constant_ocean, 0.0, 1 / std_inc_ocean)[..., 0, 0]
    channels = (~constant_surface).sum() + (~constant_ocean).sum()

    return (
        _prepare(weights_surface / channels, dim, device),
        _prepare(weights_ocean / channels, dim, device),
    )
