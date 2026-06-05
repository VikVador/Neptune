r"""Geography helpers for map visualizations."""

__all__ = [
    "load_grid",
    "draw_geography",
]

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from neptune.config import PATH_MASK
from neptune.data import DATASET_REGION


def load_grid() -> tuple[np.ndarray, np.ndarray]:
    r"""Loading longitude and latitude arrays for the dataset region."""
    ds = xr.open_zarr(PATH_MASK)
    lat = ds["latitude"].values[DATASET_REGION["y"]]
    lon = ds["longitude"].values[DATASET_REGION["x"]]
    return np.meshgrid(lon, lat)


def draw_geography(ax: plt.Axes) -> None:
    r"""Add Natural Earth coastline and rivers to a cartopy GeoAxes."""
    properties = dict(edgecolor="k", linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), **properties.values())
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), **properties.values())
