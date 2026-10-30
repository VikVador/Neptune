r"""Script to build the land/sea mask of Neptune, where the water columns often outside their physical bounds are land."""

import argparse
import dask
import numpy as np
import xarray as xr

from dawgz import after, job, schedule

from neptune.config import PATH_MASK, PATH_MASK_NEMO, PATH_NEP_SCRATCH
from neptune.data import (
    DATASET_DATES_EXCLUDED,
    DATASET_DATES_TRAINING,
    DATASET_REGION,
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    VARIABLES_BOUNDS,
)
from neptune.data.dataset import get_paths_by_date, load_states
from neptune.tools import load_configuration

# Counts of the days outside the bounds, one file per year
PATH_TMP = PATH_NEP_SCRATCH / "tmp" / "mask"

# Share of the days above which a water column with values outside their bounds becomes land
FREQUENCY = 0.01


# fmt: off
#
def outside_bounds(x: np.ndarray, variables: list[str]) -> np.ndarray:
    r"""Find the points where at least one variable is outside its physical bounds.

    Arguments:
        x         : States of a group of variables (C, [Z,] Y, X), NaN on land.
        variables : Names of the variables, in the order of the channels.

    Returns:
        outside : Whether a variable is outside its bounds ([Z,] Y, X).
    """

    lower = np.array([VARIABLES_BOUNDS[var][0] for var in variables]).reshape(-1, *[1] * (x.ndim - 1))
    upper = np.array([VARIABLES_BOUNDS[var][1] for var in variables]).reshape(-1, *[1] * (x.ndim - 1))

    return ((x < lower) | (x > upper)).any(axis=0)


def count_outside_bounds(year: int) -> None:
    r"""Count, for every voxel, the training days of a year with a value outside its physical bounds.

    Arguments:
        year : Year of the training period to cover.
    """

    dask.config.set(scheduler="synchronous")

    paths  = get_paths_by_date()
    dates  = [d for d in sorted(paths) if d[:4] == str(year) and d not in DATASET_DATES_EXCLUDED]
    counts = 0

    for date in dates:
        x_s, x_o = (x.numpy() for x in load_states(paths[date]))
        outside  = outside_bounds(x_o, DATASET_VARIABLES_OCEAN)
        outside[0] |= outside_bounds(x_s, DATASET_VARIABLES_SURFACE)
        counts   = counts + outside.astype(np.int32)

    np.savez(PATH_TMP / f"{year}.npz", counts=counts, days=len(dates))


def build_mask() -> None:
    r"""Merge the counts of every year and save the mask of NEMO, with the water columns often outside their bounds as land."""

    files  = sorted(PATH_TMP.glob("*.npz"))
    counts = sum(np.load(f)["counts"] for f in files)
    days   = sum(int(np.load(f)["days"]) for f in files)

    # A water column becomes land when one of its voxels is outside its bounds on more than FREQUENCY of the days
    frequency = (counts / days).max(axis=0)
    columns   = frequency > FREQUENCY

    # The encoding of the original store (zarr v2 compressor) is not carried over
    with xr.open_zarr(PATH_MASK_NEMO) as ds:
        mask = ds.load().drop_encoding()

    region = {"latitude": DATASET_REGION["y"], "longitude": DATASET_REGION["x"]}
    mask["mask"][dict(region)] = mask["mask"][dict(region)].where(~columns, 0)

    # The share of the days outside the bounds is kept, to show where the mask changed
    mask["frequency"] = xr.zeros_like(mask["mask"].isel(level=0), dtype=np.float32)
    mask["frequency"][dict(region)] = frequency.astype(np.float32)
    mask.attrs["description"] = f"Mask of NEMO, the water columns outside their physical bounds on more than {FREQUENCY:.0%} of the {days} training days being land."

    mask.to_zarr(PATH_MASK, mode="w")
    print(f"{int(columns.sum())} water columns set to land, from {len(files)} years and {days} days.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build the land/sea mask of Neptune, from the days outside the physical bounds.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the mask .yml configuration file.",
    )

    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster-based scheduling and 'async' for local execution.",
    )

    args   = parser.parse_args()
    config = load_configuration(args.config)[0]
    years  = list(range(int(DATASET_DATES_TRAINING[0][:4]), int(DATASET_DATES_TRAINING[1][:4]) + 1))

    PATH_TMP.mkdir(parents=True, exist_ok=True)

    @job(array=len(years), **config["Compute"])
    def count(i: int) -> None:
        r"""Count the days outside the bounds of the i-th year."""
        count_outside_bounds(years[i])

    @after(count)
    @job(**config["Aggregate"])
    def aggregate() -> None:
        r"""Build the mask from the counts of every year."""
        build_mask()

    schedule(
        aggregate,
        name="NEPT-MASK",
        backend=args.backend,
        export="ALL",
    )
