r"""Geography helpers for map visualizations."""

__all__ = [
    "load_grid",
    "load_depth",
    "draw_geography",
]

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from neptune.config import PATH_MASK
from neptune.data import DATASET_REGION


def load_grid() -> tuple[np.ndarray, np.ndarray]:
    r"""Load the longitude and latitude grids of the dataset region.

    Returns:
        lon : Longitude of each grid point (Y, X) [°].
        lat : Latitude of each grid point (Y, X) [°].
    """

    with xr.open_zarr(PATH_MASK) as ds:
        lat = ds["latitude"].values[DATASET_REGION["y"]]
        lon = ds["longitude"].values[DATASET_REGION["x"]]

    return np.meshgrid(lon, lat)


def load_depth() -> np.ndarray:
    r"""Load the depth of each level of the dataset region.

    Returns:
        depth : Depth of each level (Z,) [m].
    """

    with xr.open_zarr(PATH_MASK) as ds:
        return ds["level"].values[DATASET_REGION["z"]]


def draw_geography(ax: plt.Axes) -> None:
    r"""Add the Natural Earth coastline and rivers to a cartopy GeoAxes.

    Arguments:
        ax : Cartopy GeoAxes to draw on.
    """

    properties = dict(edgecolor="k", linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), **properties)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), **properties)
