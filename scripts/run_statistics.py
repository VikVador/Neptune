r"""Script to compute the mean and std of the cleaned states, and of their daily increments, over the training period."""

import argparse
import dask
import numpy as np
import xarray as xr

from datetime import date, timedelta
from dawgz import after, job, schedule

from neptune.config import PATH_MASK_NEMO, PATH_NEP_SCRATCH, PATH_STATS, PATH_STATS_INCREMENTS
from neptune.data import (
    DATASET_DATES_EXCLUDED,
    DATASET_DATES_TRAINING,
    DATASET_REGION,
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
)
from neptune.data.dataset import NeptuneDataset
from neptune.tools import load_configuration

# Partial sums, one file per year
PATH_TMP = PATH_NEP_SCRATCH / "tmp" / "statistics"


# fmt: off
#
def moments(x: np.ndarray) -> np.ndarray:
    r"""Count, sum and sum of squares of each channel and level, over the points that are not NaN.

    Arguments:
        x : States of a group of variables (C, [Z,] Y, X), NaN on land and outside the physical bounds.

    Returns:
        moments : Count, sum and sum of squares (3, C, [Z]).
    """

    x = x.astype(np.float64)
    return np.stack([np.isfinite(x).sum((-2, -1)), np.nansum(x, (-2, -1)), np.nansum(x**2, (-2, -1))])


def compute_sums(year: int) -> None:
    r"""Sum the moments of the cleaned states of a year, and of their increments from the previous day.

    Arguments:
        year : Year of the training period to cover.
    """

    dask.config.set(scheduler="synchronous")

    # The states are cleaned and transformed like in training, but neither standardized nor filled
    dataset  = NeptuneDataset(*DATASET_DATES_TRAINING, standardized=False, fill_with_nans=True)
    excluded = set(DATASET_DATES_EXCLUDED)
    dates    = [d for d in dataset.dates if d[:4] == str(year) and d not in excluded]
    sums     = {"states_surface": 0, "states_ocean": 0, "increments_surface": 0, "increments_ocean": 0}
    previous = {}

    # The increment of the first day of the year needs the last day of the previous year
    first = (date.fromisoformat(dates[0]) - timedelta(days=1)).isoformat()
    if first in dataset.dates and first not in excluded:
        previous[first] = [x.numpy() for x in dataset.preprocess(first)]

    for d in dates:
        x_s, x_o = (x.numpy() for x in dataset.preprocess(d))
        sums["states_surface"] = sums["states_surface"] + moments(x_s)
        sums["states_ocean"]   = sums["states_ocean"] + moments(x_o)

        # An increment needs both days, the excluded ones breaking the sequence
        before = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
        if before in previous:
            p_s, p_o = previous[before]
            sums["increments_surface"] = sums["increments_surface"] + moments(x_s - p_s)
            sums["increments_ocean"]   = sums["increments_ocean"] + moments(x_o - p_o)

        previous = {d: [x_s, x_o]}

    np.savez(PATH_TMP / f"{year}.npz", **sums)


def mean_std(sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the mean and std from the count, sum and sum of squares.

    Arguments:
        sums : Count, sum and sum of squares (3, C, [Z]).

    Returns:
        mean : Mean of each channel and level (C, [Z]).
        std  : Standard deviation of each channel and level (C, [Z]), at least 1e-8.
    """

    count, total, squares = sums
    mean = total / count

    return mean, np.maximum(np.sqrt(np.maximum(squares / count - mean**2, 0)), 1e-8)


def aggregate_sums() -> None:
    r"""Merge the sums of every year and save the statistics of the states and of their increments."""

    files = sorted(PATH_TMP.glob("*.npz"))
    sums  = {key: sum(np.load(f)[key] for f in files) for key in np.load(files[0]).files}
    depth = xr.open_zarr(PATH_MASK_NEMO).level.isel(level=DATASET_REGION["z"]).values

    for kind, path in (("states", PATH_STATS), ("increments", PATH_STATS_INCREMENTS)):
        mean_s, std_s = mean_std(sums[f"{kind}_surface"])
        mean_o, std_o = mean_std(sums[f"{kind}_ocean"])

        # One mean and std per surface variable, and per level of each ocean variable
        data_vars = {
            var: xr.DataArray([mean_s[i], std_s[i]], dims=["statistic"], coords={"statistic": ["mean", "std"]})
            for i, var in enumerate(DATASET_VARIABLES_SURFACE)
        }
        data_vars |= {
            var: xr.DataArray(
                np.stack([mean_o[i], std_o[i]]),
                dims=["statistic", "depth"],
                coords={"statistic": ["mean", "std"], "depth": depth},
            )
            for i, var in enumerate(DATASET_VARIABLES_OCEAN)
        }

        xr.Dataset(data_vars).to_zarr(path, mode="w")

    print(f"Statistics of the states and increments saved, from {len(files)} years.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute the statistics of the cleaned states and of their daily increments.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the statistics .yml configuration file.",
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

    # Removing the partial sums of a previous run
    PATH_TMP.mkdir(parents=True, exist_ok=True)
    for stale in PATH_TMP.glob("*.npz"):
        stale.unlink()

    @job(array=len(years), **config["Compute"])
    def compute(i: int) -> None:
        r"""Sum the moments of the i-th year."""
        compute_sums(years[i])

    @after(compute)
    @job(**config["Aggregate"])
    def aggregate() -> None:
        r"""Merge the sums of every year into the statistics."""
        aggregate_sums()

    schedule(
        aggregate,
        name="NEPT-STATS",
        backend=args.backend,
        export="ALL",
    )
