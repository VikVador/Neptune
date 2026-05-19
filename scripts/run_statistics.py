r"""Script to compute global mean and standard deviation of all dataset variables."""

import argparse
import dawgz
import numpy as np
import pickle
import sys
import wandb
import xarray as xr
import yaml

from pathlib import Path

from neptune.data import DATASET_DATES_TRAINING
from neptune.data.statistics import OnlineStats, clean
from neptune.data.tools import generate_paths


# fmt: off
#
def list_dataset_variables() -> list[tuple[str, str | None, list[tuple[int, float]]]]:
    r"""Return one entry per physical variable in the dataset.

    Returns:
        entries: one entry per variable, as (var, depth_dim, levels).
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


def compute_stats(
    var: str,
    depth_dim: str | None,
    levels: list[tuple[int, float]],
) -> None:
    r"""Compute online mean and std for one variable (all levels) over all training months.

    Arguments:
        var       : name of the dataset variable.
        depth_dim : name of the depth dimension, or None for 2D variables.
        levels    : list of (level_index, depth_value) pairs; empty for 2D variables.
    """
    tmp_dir = Path.cwd() / "tmp"
    date_start, date_end = DATASET_DATES_TRAINING

    wandb.init(entity="neptuneAI", project="neptune - statistics", name=var, mode="online")

    level_depth_pairs = levels if levels else [(None, None)]
    stats_map = {lvl: OnlineStats() for lvl, _ in level_depth_pairs}

    tmp_dir.mkdir(parents=True, exist_ok=True)

    for month, month_paths in sorted(generate_paths().items()):
        if not (date_start[:7] <= month <= date_end[:7]):
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
            stats_map[lvl].update(clean(data.astype(np.float32), var))

    with open(tmp_dir / f"{var}.pkl", "wb") as f:
        pickle.dump(
            {
                "var": var,
                "depth_dim": depth_dim,
                "levels": {
                    lvl: {
                        "depth_val": dv,
                        "mean": stats_map[lvl].mean,
                        "std": stats_map[lvl].std,
                    }
                    for lvl, dv in level_depth_pairs
                },
            },
            f,
        )

    wandb.log({
        "statistics": wandb.Table(
            columns=["depth_val", "mean", "std"],
            data=[[dv, stats_map[lvl].mean, stats_map[lvl].std] for lvl, dv in level_depth_pairs],
        )
    })
    wandb.finish()


def aggregate_stats(path_output: str) -> None:
    r"""Load all temp files and assemble the final xarray statistics dataset.

    Arguments:
        path_output: path to the output .zarr file.
    """
    tmp_dir = Path.cwd() / "tmp"
    data_vars = {}

    for pkl_file in tmp_dir.glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            entry = pickle.load(f)

        var, depth_dim, levels = entry["var"], entry["depth_dim"], entry["levels"]

        if depth_dim is None:
            stats = levels[None]
            data_vars[var] = xr.DataArray(
                [stats["mean"], stats["std"]],
                dims=["statistic"],
                coords={"statistic": ["mean", "std"]},
            )
        else:
            sorted_keys = sorted(k for k in levels)
            data_vars[var] = xr.DataArray(
                np.array([
                    [levels[k]["mean"] for k in sorted_keys],
                    [levels[k]["std"] for k in sorted_keys],
                ]),
                dims=["statistic", depth_dim],
                coords={
                    "statistic": ["mean", "std"],
                    depth_dim: [levels[k]["depth_val"] for k in sorted_keys],
                },
            )

    xr.Dataset(data_vars).to_zarr(path_output, mode="w")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute global dataset statistics.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML config file (relative to scripts/).",
    )
    parser.add_argument(
        "--path_output",
        "-o",
        type=str,
        required=True,
        help="Output .zarr path.",
    )
    parser.add_argument(
        "--aggregate-only",
        "-a",
        action="store_true",
        help="Skip compute jobs and run only the aggregation step.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend.",
    )

    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_stats(args.path_output)
        sys.exit(0)

    cfg        = yaml.safe_load((Path(__file__).parent / args.config).read_text())
    job_config = cfg["compute_stats"]
    jobs       = list_dataset_variables()

    tmp_dir = Path.cwd() / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for stale in tmp_dir.glob("*.pkl"):
        stale.unlink()

    @dawgz.job(array=len(jobs), **job_config)
    def compute(i: int) -> None:
        var, depth_dim, levels = jobs[i]
        compute_stats(var, depth_dim, levels)

    dawgz.schedule(compute, name="NEPTUNE-STATS", backend=args.backend, export="ALL")
