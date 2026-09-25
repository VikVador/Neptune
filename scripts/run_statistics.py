r"""Script to compute global mean and standard deviation of all dataset variables."""

import argparse
import dask
import multiprocessing
import numpy as np
import pickle
import xarray as xr

from dawgz import after, job, schedule
from pathlib import Path

from neptune.config import PATH_MASK, PATH_NEP_SCRATCH, PATH_STATS
from neptune.data import DATASET_DATES_TRAINING
from neptune.data.statistics import OnlineStats, clean
from neptune.data.tools import generate_paths
from neptune.tools import load_configuration

# Partial statistics, one file per variable and period
PATH_TMP = PATH_NEP_SCRATCH / "tmp" / "statistics"

# Number of years per task, small tasks balancing the load between processes
YEARS_PER_TASK = 2


# fmt: off
#
def list_dataset_variables() -> list[tuple[str, str | None, list[tuple[int, float]]]]:
    r"""Return one entry per physical variable in the dataset.

    Returns:
        entries : One entry per variable, as (var, depth_dim, levels).
    """

    paths = generate_paths()
    first_paths = next(iter(paths.values()))

    ds = xr.open_mfdataset(
        first_paths,
        combine="by_coords",
        compat="override",
        coords="minimal",
        data_vars="minimal",
    ).drop_vars(["nav_lat", "nav_lon"], errors="ignore")

    result = []
    for var in ds.data_vars:
        dims = ds[var].dims
        if "y" not in dims or "x" not in dims:
            continue
        depth_dim = next((d for d in dims if d.startswith("depth")), None)
        if depth_dim is None:
            result.append((var, None, []))
        else:
            levels = [(i, float(v)) for i, v in enumerate(ds[depth_dim].values)]
            result.append((var, depth_dim, levels))

    ds.close()

    return result


def list_periods() -> list[tuple[int, int]]:
    r"""Split the training years into periods of YEARS_PER_TASK years.

    Returns:
        periods : First and last year of each period.
    """

    first, last = (int(date[:4]) for date in DATASET_DATES_TRAINING)

    return [(year, min(year + YEARS_PER_TASK - 1, last)) for year in range(first, last + 1, YEARS_PER_TASK)]


def compute_stats(
    var: str,
    depth_dim: str | None,
    levels: list[tuple[int, float]],
    period: tuple[int, int],
) -> None:
    r"""Compute online mean and std for one variable (all levels) over the sea points of a period.

    Arguments:
        var       : Name of the dataset variable.
        depth_dim : Name of the depth dimension, or None for 2D variables.
        levels    : (level_index, depth_value) pairs, empty for 2D variables.
        period    : First and last year of the training period to cover.
    """

    dask.config.set(scheduler="synchronous")
    date_start, date_end = DATASET_DATES_TRAINING

    # Land is either NaN or 0 in the raw files, the mask sets it to NaN everywhere
    mask = xr.open_zarr(PATH_MASK).mask.values

    level_depth_pairs = levels if levels else [(None, None)]
    stats_map = {lvl: OnlineStats() for lvl, _ in level_depth_pairs}

    for month, month_paths in sorted(generate_paths().items()):
        if not (date_start[:7] <= month <= date_end[:7] and period[0] <= int(month[:4]) <= period[1]):
            continue

        ds = xr.open_mfdataset(
            month_paths,
            combine="by_coords",
            compat="override",
            coords="minimal",
            data_vars="minimal",
        )

        if var not in ds:
            ds.close()
            continue

        da = ds[var].load()
        ds.close()

        for lvl, _ in level_depth_pairs:
            data = da.isel({depth_dim: lvl}).values if lvl is not None else da.values
            data = np.where(mask[lvl if lvl is not None else 0] == 1, data, np.nan)
            stats_map[lvl].update(clean(data.astype(np.float32), var))

    with open(PATH_TMP / f"{var}_{period[0]}.pkl", "wb") as f:
        pickle.dump(
            {
                "var": var,
                "depth_dim": depth_dim,
                "levels": {lvl: {"depth_val": dv, "stats": stats_map[lvl]} for lvl, dv in level_depth_pairs},
            },
            f,
        )


def compute_stats_serial(tasks: list[tuple[tuple[str, str | None, list], tuple[int, int]]]) -> None:
    r"""Compute the statistics of several (variable, period) pairs, one after the other.

    Arguments:
        tasks : ((var, depth_dim, levels), period) pairs.
    """

    for (var, depth_dim, levels), period in tasks:
        compute_stats(var, depth_dim, levels, period)


def compute_stats_parallel(
    tasks: list[tuple[tuple[str, str | None, list], tuple[int, int]]],
    processes: int,
) -> None:
    r"""Compute the statistics of several (variable, period) pairs, shared between processes.

    Forked processes inherit the functions of the script, which therefore need not be pickled.

    Arguments:
        tasks     : ((var, depth_dim, levels), period) pairs.
        processes : Number of parallel processes.
    """

    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=compute_stats_serial, args=(tasks[k::processes],)) for k in range(processes)]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    # Security
    failed = [worker.exitcode for worker in workers if worker.exitcode != 0]
    if failed:
        raise RuntimeError(f"ERROR - {len(failed)} processes failed with exit codes {failed}.")


def aggregate_stats(path_output: Path) -> None:
    r"""Merge the partial statistics of every period and assemble the final statistics dataset.

    Arguments:
        path_output : Path to the output .zarr file.
    """

    entries = {}
    for pkl_file in sorted(PATH_TMP.glob("*.pkl")):
        with open(pkl_file, "rb") as f:
            entry = pickle.load(f)

        if entry["var"] not in entries:
            entries[entry["var"]] = entry
        else:
            for lvl, level in entry["levels"].items():
                entries[entry["var"]]["levels"][lvl]["stats"].merge(level["stats"])

    data_vars = {}
    for var, entry in entries.items():
        depth_dim, levels = entry["depth_dim"], entry["levels"]

        if depth_dim is None:
            stats = levels[None]["stats"]
            data_vars[var] = xr.DataArray(
                [stats.mean, stats.std],
                dims=["statistic"],
                coords={"statistic": ["mean", "std"]},
            )
        else:
            sorted_keys = sorted(levels)
            data_vars[var] = xr.DataArray(
                np.array([
                    [levels[k]["stats"].mean for k in sorted_keys],
                    [levels[k]["stats"].std for k in sorted_keys],
                ]),
                dims=["statistic", depth_dim],
                coords={
                    "statistic": ["mean", "std"],
                    depth_dim: [levels[k]["depth_val"] for k in sorted_keys],
                },
            )

    xr.Dataset(data_vars).to_zarr(path_output, mode="w")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute the statistics of every dataset variable, over sea points.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the statistics .yml configuration file.",
    )

    parser.add_argument(
        "--path_output",
        "-o",
        type=Path,
        default=PATH_STATS,
        help="Path to the output .zarr file (default: PATH_STATS).",
    )

    parser.add_argument(
        "--aggregate-only",
        "-a",
        action="store_true",
        help="Only aggregate the partial statistics already computed.",
    )

    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster-based scheduling and 'async' for local execution.",
    )

    args             = parser.parse_args()
    config           = load_configuration(args.config)[0]
    config_compute   = config["Compute"]
    config_aggregate = config["Aggregate"]
    jobs             = config_compute.pop("jobs")

    # Aggregation only
    if args.aggregate_only:
        aggregate_stats(args.path_output)

    # Computation and aggregation
    else:
        tasks = [(entry, period) for entry in list_dataset_variables() for period in list_periods()]

        # Removing the partial statistics of a previous run
        PATH_TMP.mkdir(parents=True, exist_ok=True)
        for stale in PATH_TMP.glob("*.pkl"):
            stale.unlink()

        @job(array=jobs, **config_compute)
        def compute(i: int) -> None:
            r"""Compute the statistics of the i-th share of (variable, period) pairs, one process per CPU."""
            compute_stats_parallel(tasks[i::jobs], processes=config_compute["cpus"])

        @after(compute)
        @job(**config_aggregate)
        def aggregate() -> None:
            r"""Merge the statistics of every period into the final dataset."""
            aggregate_stats(args.path_output)

        schedule(
            aggregate,
            name="NEPT-STATS",
            backend=args.backend,
            export="ALL",
        )
