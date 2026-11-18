r"""Errors of the operational forecast: grid, reading of the states, ingredients and comparisons."""

import json
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from functools import cache
from pathlib import Path

from neptune.config import PATH_FORECAST_ERRORS, PATH_MASK, PATH_MESH, SIMULATION_INTERIM
from neptune.data import (
    DATASET_VARIABLES,
    VARIABLES_BOUNDS,
    VARIABLES_CLIPPING,
    VARIABLES_LOG10,
)

# fmt: off
#
# ----- Files and variables
#
PATH_PREPARATION  = PATH_FORECAST_ERRORS / "preparation"
PATH_OBSERVATIONS = PATH_FORECAST_ERRORS / "observations"
PATH_TRANSIT      = PATH_FORECAST_ERRORS / "transit"
PATH_RUNS         = PATH_FORECAST_ERRORS / "runs"
PATH_DAYS         = PATH_FORECAST_ERRORS / "days"
PATH_RESULTS      = PATH_FORECAST_ERRORS / "results"

# Variables of Neptune in each kind of NEMO file, and the grid point of those not on the T points
GRIDS = {
    "grid_T" : ["ssh", "votemper", "vosaline"],
    "grid_U" : ["tauuo", "uo"],
    "grid_V" : ["tauvo", "vo"],
    "ptrc_T" : ["CHL", "DOX", "PAR", "PHO", "SIO", "NOS"],
}
POINTS = {"tauuo": "u", "uo": "u", "tauvo": "v", "vo": "v"}

# ----- Ingredients
#
LEADS       = 11
LEVELS      = 59
ZONES       = ["basin", "shelf", "open_sea"]
WEIGHTINGS  = ["cell", "none"]
TRANSFORMS  = ["raw", "log10"]
CORRECTIONS = ["clipped_to_zero", "out_of_bounds"]
STATISTICS  = [
    "count",
    "sum_weights",
    "sum_error",
    "sum_squared_error",
    "sum_absolute_error",
    "sum_forecast",
    "sum_reference",
    "sum_squared_forecast",
    "sum_squared_reference",
    "sum_product",
    "sum_forecast_anomaly",
    "sum_reference_anomaly",
    "sum_squared_forecast_anomaly",
    "sum_squared_reference_anomaly",
    "sum_product_anomaly",
]

# ----- Hypoxia on the shelf, with the oxygen threshold [mmol/m3] and the classes of the histograms
#
HYPOXIA_THRESHOLD = 62.5
HYPOXIA_BINS      = np.r_[np.arange(0, 402.5, 2.5), np.inf]
# fmt: on


@cache
def load_structure() -> dict[str, np.ndarray]:
    r"""Load the grid of the model: masks, weights, zones and depths.

    Returns:
        structure : Arrays of the grid, read from the preparation folder.
    """

    with xr.open_dataset(PATH_PREPARATION / "structure.nc") as ds:
        structure = {name: ds[name].values for name in ds.variables}

    # Masks are stored as int8
    for name in ("mask_t", "mask_u", "mask_v", "shelf"):
        structure[name] = structure[name].astype(bool)

    return structure


def build_structure() -> None:
    r"""Build the grid of the model once: masks and weights of the T, U and V points, zones and bottom."""

    with netCDF4.Dataset(PATH_MESH) as mesh:
        m = {name: mesh[name][0] for name in mesh.variables if mesh[name].ndim >= 3}
        depth = mesh["gdept_1d"][0]

    with xr.open_zarr(PATH_MASK) as ds:
        mask_t = ds["mask"].values.astype(bool)
        lon, lat = ds["longitude"].values, ds["latitude"].values

    bathymetry = np.asarray(m["bathy_metry"], dtype=np.float64)
    lon2d, lat2d = np.meshgrid(lon, lat)

    structure = {
        "mask_t": mask_t,
        "mask_u": np.asarray(m["umask"], dtype=bool),
        "mask_v": np.asarray(m["vmask"], dtype=bool),
        "bathymetry": bathymetry,
        "bottom": np.asarray(m["mbathy"], dtype=np.int32) - 1,
        "shelf": mask_t[0] & (bathymetry < 200),
        "region": kopelevich_regions(lon2d, lat2d, bathymetry) * mask_t[0],
    }
    for p in "tuv":
        area = np.asarray(m[f"e1{p}"] * m[f"e2{p}"], dtype=np.float64)
        structure[f"area_{p}"] = area
        structure[f"volume_{p}"] = area * np.asarray(m[f"e3{p}_0"], dtype=np.float64)

    dims = {2: ("y", "x"), 3: ("level", "y", "x")}
    ds = xr.Dataset(
        {
            name: (dims[x.ndim], x.astype(np.int8) if x.dtype == bool else x)
            for name, x in structure.items()
        },
        coords={"lon": ("x", lon), "lat": ("y", lat), "depth": ("level", np.asarray(depth))},
        attrs={
            "description": "Grid of NEMO for the forecast errors: masks, weights, zones and bottom level."
        },
    )
    PATH_PREPARATION.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(PATH_PREPARATION / "structure.nc")


def kopelevich_regions(lon: np.ndarray, lat: np.ndarray, bathymetry: np.ndarray) -> np.ndarray:
    r"""Approximate the 11 sub-regions of the chlorophyll quality metrics (QUID 007-010, Figure 1).

    Arguments:
        lon        : Longitude of each point [°E].
        lat        : Latitude of each point [°N].
        bathymetry : Depth of each point [m].

    Returns:
        region : Sub-region of each point, from 1 to 11.
    """

    deep, mid, shallow = (
        bathymetry >= 200,
        (bathymetry >= 50) & (bathymetry < 200),
        bathymetry < 50,
    )
    west = lon < 34.0 + (44.6 - lat) * 0.1
    east = lon > 38.5 + (lat - 41.0) * 0.86
    north_west = (lat > 44.3) & (lon < 33.6)

    region = np.select(
        [
            deep & west,
            deep & east,
            deep,
            north_west & shallow & (lon + lat > 76.3),
            north_west & shallow,
            shallow & (lon < 30.0),
            mid & (lon < 34.0),
            (lat < 42.3) & (lon < 34.5),
            (lat < 42.3) & (lon < 41.0),
            (lon > 40.0) & (lat < 43.3),
        ],
        [1, 3, 2, 4, 5, 6, 7, 8, 9, 10],
        default=11,
    )

    return region.astype(np.int8)


def sea_points(point: str) -> np.ndarray:
    r"""Flat indices of the sea columns of a grid point.

    Arguments:
        point : Grid point, 't', 'u' or 'v'.

    Returns:
        index : Flat indices (Y * X) of the columns with sea at the surface.
    """

    return np.flatnonzero(load_structure()[f"mask_{point}"][0])


def read_state(paths: dict[str, Path]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    r"""Read the variables of Neptune from the NEMO files of one day, with the corrections of the data audit.

    Arguments:
        paths : Path of the file of each kind ('grid_T', 'grid_U', 'grid_V', 'ptrc_T').

    Returns:
        fields      : Each variable in float64, NaN on land and outside its physical bounds, (Z, Y, X) or (Y, X).
        corrections : Number of values clipped to zero and out of bounds, per variable (V, 2).
    """

    structure = load_structure()
    fields, corrections = {}, np.zeros((len(DATASET_VARIABLES), 2))
    for kind, names in GRIDS.items():
        with netCDF4.Dataset(paths[kind]) as dataset:
            dataset.set_auto_mask(False)
            for name in names:
                x = np.asarray(dataset[name][0], dtype=np.float64)
                mask = structure[f"mask_{POINTS.get(name, 't')}"]
                x[~(mask if x.ndim == 3 else mask[0])] = np.nan
                x[np.abs(x) > 1e19] = np.nan

                v = DATASET_VARIABLES.index(name)
                if name in VARIABLES_CLIPPING:
                    corrections[v, 0] = np.sum(x < 0)
                    x[x < 0] = 0.0
                lo, hi = VARIABLES_BOUNDS[name]
                outside = (x < lo) | (x > hi)
                corrections[v, 1] = np.sum(outside)
                x[outside] = np.nan
                fields[name] = x

    return fields, corrections


@cache
def reanalysis_paths() -> dict[str, dict[str, str]]:
    r"""Paths of the files of the interim reanalysis, per date 'YYYY-MM-DD' and kind of file.

    Returns:
        paths : Path of each kind of file, per date.
    """

    with open(PATH_PREPARATION / "reanalysis.json") as file:
        return json.load(file)


def index_reanalysis() -> None:
    r"""List the daily files of the interim reanalysis once, per date and kind."""

    paths = {}
    for kind in GRIDS:
        for path in sorted(SIMULATION_INTERIM.glob(f"*/BS_1d_*_{kind}_*.nc")):
            d = path.name.split("_")[2]
            paths.setdefault(f"{d[:4]}-{d[4:6]}-{d[6:]}", {})[kind] = str(path)

    with open(PATH_PREPARATION / "reanalysis.json", "w") as file:
        json.dump(paths, file, indent=1)


def read_reanalysis(date: str) -> dict[str, np.ndarray]:
    r"""Read the interim reanalysis of a date.

    Arguments:
        date : Date string 'YYYY-MM-DD'.

    Returns:
        fields : Each variable of Neptune, corrected, (Z, Y, X) or (Y, X).
    """

    return read_state({kind: Path(p) for kind, p in reanalysis_paths()[date].items()})[0]


def read_climatology(date: str) -> dict[str, np.ndarray]:
    r"""Read the daily climatology of the day of the year of a date.

    Arguments:
        date : Date string 'YYYY-MM-DD'.

    Returns:
        fields : Each variable of Neptune in float64, (Z, Y, X) or (Y, X).
    """

    with xr.open_zarr(PATH_PREPARATION / "climatology.zarr") as ds:
        ds = ds.sel(day=date[5:]).load()

    return {name: ds[name].values.astype(np.float64) for name in DATASET_VARIABLES}


def ingredients(
    f: np.ndarray, o: np.ndarray, c: np.ndarray, w: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    r"""Sum the 15 ingredients of the diagnostics over the points of each group.

    Arguments:
        f      : Values of the evaluated system (..., N).
        o      : Values of the reference (..., N).
        c      : Values of the climatology (..., N).
        w      : Weights of the points, broadcastable to f.
        groups : Membership of the points to each group, 0 or 1 (N, G).

    Returns:
        sums : Ingredients of each group, in the order of STATISTICS (..., G, 15).
    """

    valid = np.isfinite(f) & np.isfinite(o) & np.isfinite(c) & (w > 0)
    w = np.where(valid, w, 0.0)
    f, o, c = (np.where(valid, x, 0.0) for x in (f, o, c))
    d, fa, oa = f - o, f - c, o - c

    terms = (
        valid, w, w * d, w * d**2, w * np.abs(d), w * f, w * o, w * f**2, w * o**2, w * f * o,
        w * fa, w * oa, w * fa**2, w * oa**2, w * fa * oa,
    )  # fmt: skip

    return np.stack([np.asarray(t, dtype=np.float64) @ groups for t in terms], axis=-1)


def transform(name: str, x: np.ndarray, kind: str) -> np.ndarray:
    r"""Transform the values of a variable before computing the ingredients.

    Arguments:
        name : Name of the variable.
        x    : Values, clipped to zero for the log-normal variables.
        kind : 'raw' or 'log10'.

    Returns:
        x : Values, or log10(x + epsilon), NaN for 'log10' if the variable is not log-normal.
    """

    if kind == "raw":
        return x
    if name not in VARIABLES_LOG10:
        return np.full_like(x, np.nan)
    return np.log10(x + VARIABLES_LOG10[name])


def compare_fields(
    f: dict[str, np.ndarray], o: dict[str, np.ndarray], c: dict[str, np.ndarray]
) -> np.ndarray:
    r"""Ingredients of a system against a gridded reference of the same day, per level, zone and variant.

    Arguments:
        f : Fields of the evaluated system, per variable.
        o : Fields of the reference (reanalysis), per variable.
        c : Fields of the climatology of the day, per variable.

    Returns:
        sums : Ingredients (variable, level + volume, zone, weighting, transform, statistic).
    """

    structure = load_structure()
    sums = np.zeros((
        len(DATASET_VARIABLES),
        LEVELS + 1,
        len(ZONES),
        len(WEIGHTINGS),
        len(TRANSFORMS),
        len(STATISTICS),
    ))
    for v, name in enumerate(DATASET_VARIABLES):
        p = POINTS.get(name, "t")
        index = sea_points(p)
        shelf = structure["shelf"].ravel()[index]
        zones = np.stack([np.ones_like(shelf), shelf, ~shelf], axis=1).astype(np.float64)

        area = structure[f"area_{p}"].ravel()[index]
        volume = structure[f"volume_{p}"].reshape(LEVELS, -1)[:, index]
        for t, kind in enumerate(TRANSFORMS):
            ff, oo, cc = (
                transform(name, x[name], kind).reshape(-1, x[name].shape[-2] * x[name].shape[-1])[
                    :, index
                ]
                for x in (f, o, c)
            )
            if not np.isfinite(ff).any():
                continue
            n = ff.shape[0]
            sums[v, :n, :, 0, t] = ingredients(ff, oo, cc, area, zones)
            sums[v, :n, :, 1, t] = ingredients(ff, oo, cc, np.ones_like(area), zones)
            if n == LEVELS:
                sums[v, LEVELS, :, 0, t] = ingredients(ff, oo, cc, volume, zones).sum(axis=0)
                sums[v, LEVELS, :, 1, t] = sums[v, :LEVELS, :, 1, t].sum(axis=0)

    return sums


def compare_hypoxia(f: np.ndarray, o: np.ndarray) -> dict[str, np.ndarray]:
    r"""Hypoxia of a system against the reanalysis, at the bottom of the shelf.

    Arguments:
        f : Oxygen of the evaluated system (Z, Y, X) [mmol/m3].
        o : Oxygen of the reanalysis (Z, Y, X) [mmol/m3].

    Returns:
        contingency : Area of the hits, false alarms, misses and correct negatives (4,) [m2].
        extent      : Hypoxic area [m2] and volume [m3] of the system and of the reanalysis (2, 2).
        histogram   : Area of the bottom oxygen of the system per class, reanalysis hypoxic or not (2, B) [m2].
    """

    structure = load_structure()
    shelf, bottom = structure["shelf"], structure["bottom"]
    j, i = np.nonzero(shelf)
    k = bottom[j, i]
    fb, ob, area = f[k, j, i], o[k, j, i], structure["area_t"][j, i]

    valid = np.isfinite(fb) & np.isfinite(ob)
    fb, ob, area = fb[valid], ob[valid], area[valid]
    yes_f, yes_o = fb < HYPOXIA_THRESHOLD, ob < HYPOXIA_THRESHOLD

    volume = np.where(shelf[None], structure["volume_t"], 0.0)
    extent = np.array([
        [area[yes_f].sum(), np.sum(volume * (f < HYPOXIA_THRESHOLD))],
        [area[yes_o].sum(), np.sum(volume * (o < HYPOXIA_THRESHOLD))],
    ])

    return {
        "contingency" : np.array([area[yes_f & yes_o].sum(), area[yes_f & ~yes_o].sum(), area[~yes_f & yes_o].sum(), area[~yes_f & ~yes_o].sum()]),
        "extent"      : extent,
        "histogram"   : np.stack([np.histogram(fb[yes_o == truth], HYPOXIA_BINS, weights=area[yes_o == truth])[0] for truth in (True, False)]),
    }  # fmt: skip


@cache
def load_product(product: str) -> dict[str, np.ndarray]:
    r"""Load the observations of a product, prepared once on the grid of the model.

    Arguments:
        product : Name of the product, e.g. 'l3_sst_1km'.

    Returns:
        observations : Units sorted by day, their subsets, and the operator from the model to the units.
    """

    path = PATH_OBSERVATIONS / "prepared" / product

    return {p.stem: np.load(p, mmap_mode="r") for p in path.glob("*.npy")}


def products() -> list[str]:
    r"""Names of the prepared observation products.

    Returns:
        products : Names, sorted.
    """

    return sorted(p.name for p in (PATH_OBSERVATIONS / "prepared").iterdir() if p.is_dir())


def compare_observations(
    f: dict[str, np.ndarray], c: dict[str, np.ndarray], date: str, product: str
) -> np.ndarray:
    r"""Ingredients of a system against the observations of a product on one day, per subset and variant.

    Arguments:
        f       : Fields of the evaluated system, per variable.
        c       : Fields of the climatology of the day, per variable.
        date    : Date string 'YYYY-MM-DD'.
        product : Name of the product.

    Returns:
        sums : Ingredients (subset, weighting, transform, statistic), zeros if no observation that day.
    """

    obs = load_product(product)
    name, subsets = obs["variable"].item(), obs["subsets"]
    sums = np.zeros((len(subsets), len(WEIGHTINGS), len(TRANSFORMS), len(STATISTICS)))

    day = (pd.Timestamp(date) - pd.Timestamp(obs["origin"].item())).days
    start, end = np.searchsorted(obs["unit_day"], [day, day + 1])
    if start == end:
        return sums

    # Operator of the day, from the flat field of the model to the units (a NaN in its support gives a NaN)
    lo, hi = np.searchsorted(obs["op_unit"], [start, end])
    unit, cell, weight = (
        obs["op_unit"][lo:hi] - start,
        obs["op_cell"][lo:hi],
        obs["op_weight"][lo:hi],
    )
    operator = lambda x: np.bincount(unit, weights=weight * x.ravel()[cell], minlength=end - start)
    groups = obs["unit_subsets"][start:end].astype(np.float64)
    area, passes = obs["unit_area"][start:end], obs["unit_pass"][start:end]

    for t, kind in enumerate(TRANSFORMS):
        ff, cc = (operator(transform(name, x[name], kind)) for x in (f, c))
        oo = transform(name, obs["unit_value"][start:end], kind)
        if not np.isfinite(ff).any():
            continue
        if bool(obs["anomaly"]):
            ff, oo, cc = (
                remove_pass_mean(x, area, passes, np.isfinite(ff) & np.isfinite(oo))
                for x in (ff, oo, cc)
            )
        sums[:, 0, t] = ingredients(ff, oo, cc, area, groups)
        sums[:, 1, t] = ingredients(ff, oo, cc, np.ones_like(area), groups)

    return sums


def remove_pass_mean(
    x: np.ndarray, w: np.ndarray, passes: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    r"""Remove the weighted mean of each pass (a day of a gridded product, or a satellite track).

    Arguments:
        x      : Values of the units (N,).
        w      : Weights of the units (N,).
        passes : Pass of each unit, from 0 (N,).
        valid  : Units used for the means (N,).

    Returns:
        x : Values minus the mean of their pass.
    """

    w = np.where(valid & np.isfinite(x), w, 0.0)
    total = np.bincount(passes, weights=w)
    mean = np.bincount(passes, weights=w * np.nan_to_num(x)) / np.where(total > 0, total, 1.0)

    return x - mean[passes]
