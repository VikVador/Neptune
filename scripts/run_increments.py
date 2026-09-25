r"""Script to compute the mean and standard deviation of the daily increments of every variable."""

import argparse
import dask
import torch
import xarray as xr

from dawgz import job, schedule
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from neptune.config import PATH_MASK, PATH_STATS_INCREMENTS
from neptune.data import (
    DATASET_DATES_TRAINING,
    DATASET_REGION,
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    Z,
)
from neptune.data.dataset import NeptuneDataset
from neptune.data.statistics import OnlineStats


# fmt: off
#
def compute_increments_statistics(path_output: Path, samples: int, num_workers: int) -> None:
    r"""Compute online the mean and std of the daily increments x_t+1 - x_t, per variable and level.

    Arguments:
        path_output : Path to the output .zarr file.
        samples     : Number of random pairs of consecutive days.
        num_workers : Number of processes loading the pairs.
    """

    dask.config.set(scheduler="synchronous")

    # Random pairs (x_t, x_t+1), land set to NaN to be ignored by the statistics
    dataset    = NeptuneDataset(*DATASET_DATES_TRAINING, standardized=False, fill_with_nans=True)
    indices    = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[:samples]
    dataloader = DataLoader(Subset(dataset, indices.tolist()), batch_size=None, num_workers=num_workers)

    stats_surface = [OnlineStats() for _ in DATASET_VARIABLES_SURFACE]
    stats_ocean   = [[OnlineStats() for _ in range(Z)] for _ in DATASET_VARIABLES_OCEAN]

    for x_inp_s, x_inp_o, x_out_s, x_out_o, _ in tqdm(dataloader, desc="Increments", mininterval=10):
        dx_s = (x_out_s - x_inp_s)[0].numpy()
        dx_o = (x_out_o - x_inp_o)[0].numpy()

        for i, stats in enumerate(stats_surface):
            stats.update(dx_s[i])
        for i, stats_levels in enumerate(stats_ocean):
            for z, stats in enumerate(stats_levels):
                stats.update(dx_o[i, z])

    # Same layout as the statistics of the states
    depth = xr.open_zarr(PATH_MASK).level.isel(level=DATASET_REGION["z"]).values

    data_vars = {
        var: xr.DataArray(
            [stats.mean, stats.std],
            dims=["statistic"],
            coords={"statistic": ["mean", "std"]},
        )
        for var, stats in zip(DATASET_VARIABLES_SURFACE, stats_surface, strict=True)
    }

    data_vars |= {
        var: xr.DataArray(
            [[stats.mean for stats in stats_levels], [stats.std for stats in stats_levels]],
            dims=["statistic", "depth"],
            coords={"statistic": ["mean", "std"], "depth": depth},
        )
        for var, stats_levels in zip(DATASET_VARIABLES_OCEAN, stats_ocean, strict=True)
    }

    xr.Dataset(data_vars).to_zarr(path_output, mode="w")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute the statistics of the daily increments of every variable.")
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster-based scheduling and 'async' for local execution.",
    )

    args        = parser.parse_args()
    samples     = 1024
    num_workers = 7

    # Visualization partition | Maximum resources of a single job (1 GPU is mandatory)
    @job(
        cpus      = 8,
        gpus      = 1,
        ram       = "60GB",
        time      = "04:00:00",
        account   = "bsmfc",
        partition = "visu",
    )
    def compute() -> None:
        r"""Compute the statistics of the daily increments over random pairs of consecutive days."""
        compute_increments_statistics(PATH_STATS_INCREMENTS, samples=samples, num_workers=num_workers)

    schedule(
        compute,
        name="NEPT-INCREMENTS",
        backend=args.backend,
        export="ALL",
    )
