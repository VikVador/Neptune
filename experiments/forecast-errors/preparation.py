r"""Prepare once everything that does not need the forecast: grid, climatology, observations, inventory and baselines."""

import argparse
import comparisons
import dask.array as da
import errors
import json
import numpy as np
import observations
import pandas as pd
import subprocess
import xarray as xr

from comparisons import CLIMATO, SSH_OPTIONS, compute_day, compute_persistence, run_dates
from dawgz import after, job, schedule
from errors import GRIDS, LEVELS, PATH_PREPARATION, build_structure, index_reanalysis, read_state
from observations import (
    DAYS,
    PRODUCTS_GRIDDED,
    prepare_alongtrack,
    prepare_gridded,
    prepare_insitu,
)

from neptune.data import (
    DATASET_DATES_EXCLUDED,
    DATASET_DATES_TRAINING,
    DATASET_VARIABLES,
    DATASET_VARIABLES_SURFACE,
)
from neptune.data.dataset import get_paths_by_date
from neptune.tools import freeze_modules, load_configuration

# fmt: off
#
# ----- Forecasts on Climato, and the runs compared (J-1 to J+9 covered by the interim reanalysis)
#
FORECASTS = ["/mast_tmp10/OUTPUTS/CMEMS/NRT_V2024", "/mast_tmp10/OUTPUTS/CMEMS/NRT_V2025"]
RUNS      = ("2025-01-02", "2026-09-15")
# fmt: on


def build_inventory() -> None:
    r"""List the useful forecast files on Climato with a single connection, and keep those of each compared run."""

    kinds = " -o ".join(f"-name '*_{k}_*'" for k in GRIDS)
    command = f"find {' '.join(FORECASTS)} -mindepth 2 -maxdepth 2 -name 'BS_1d_*' \\( {kinds} \\) -printf '%p %s\\n'"
    listing = subprocess.run(
        [*SSH_OPTIONS.split(), CLIMATO, command], capture_output=True, text=True, check=True
    ).stdout
    (PATH_PREPARATION / "climato.txt").write_text(listing)

    found = {}
    for line in filter(None, listing.split("\n")):
        path, size = line.split()
        version, folder, name = path.split("/")[-3:]
        run = f"{folder[:4]}-{folder[4:6]}-{folder[6:]}"
        day = name.split("_")[-1][:8]
        day = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        if (
            RUNS[0] <= run <= RUNS[1]
            and name.split("_")[2] == run_dates(run)[0].replace("-", "")
            and day in run_dates(run)
        ):
            found.setdefault(run, {}).setdefault(version, []).append((path, int(size)))

    # The newest version wins for the dates in both folders
    runs = {
        run: {"version": max(v), "files": sorted(v[max(v)])} for run, v in sorted(found.items())
    }

    with open(PATH_PREPARATION / "runs.json", "w") as file:
        json.dump(runs, file, indent=1)


def climatology_template() -> None:
    r"""Create the empty store of the daily climatology on the full grid, filled day by day."""

    structure = errors.load_structure()
    days = [d.strftime("%m-%d") for d in pd.date_range("2000-01-01", "2000-12-31")]
    shapes = {
        name: (366, 258, 578) if name in DATASET_VARIABLES_SURFACE else (366, LEVELS, 258, 578)
        for name in DATASET_VARIABLES
    }
    dims = {3: ("day", "y", "x"), 4: ("day", "level", "y", "x")}

    ds = xr.Dataset(
        {
            name: (dims[len(s)], da.full(s, np.nan, dtype=np.float32, chunks=(1, *s[1:])))
            for name, s in shapes.items()
        },
        coords={
            "day": days,
            "longitude": ("x", structure["lon"]),
            "latitude": ("y", structure["lat"]),
            "level": structure["depth"],
        },
        attrs={
            "description": f"Daily climatology of HR001 on the full grid, {DATASET_DATES_TRAINING[0]} to {DATASET_DATES_TRAINING[1]}, with the corrections of the data audit."
        },
    )
    ds.to_zarr(PATH_PREPARATION / "climatology.zarr", mode="w", compute=False)


def compute_climatology(days: list[str]) -> None:
    r"""Average the corrected states of HR001 over the training years, for some days of the year.

    Arguments:
        days : Days of the year 'MM-DD'.
    """

    paths = get_paths_by_date()
    years = range(int(DATASET_DATES_TRAINING[0][:4]), int(DATASET_DATES_TRAINING[1][:4]) + 1)
    store = xr.open_zarr(PATH_PREPARATION / "climatology.zarr")
    index = list(store["day"].values)

    for day in days:
        sums, counts = {}, {}
        for date in (f"{year}-{day}" for year in years):
            if date not in paths or date in DATASET_DATES_EXCLUDED:
                continue
            fields, _ = read_state({
                k: next(p for p in paths[date] if f"_{k}_" in p) for k in GRIDS
            })
            for name, x in fields.items():
                valid = np.isfinite(x)
                sums[name] = sums.get(name, 0.0) + np.where(valid, x, 0.0)
                counts[name] = counts.get(name, 0) + valid

        dims = {2: ("day", "y", "x"), 3: ("day", "level", "y", "x")}
        mean = {
            name: (
                dims[x.ndim],
                (np.where(counts[name] > 0, x / np.maximum(counts[name], 1), np.nan)[None]).astype(
                    np.float32
                ),
            )
            for name, x in sums.items()
        }
        i = index.index(day)
        xr.Dataset(mean).to_zarr(
            PATH_PREPARATION / "climatology.zarr", region={"day": slice(i, i + 1)}
        )


# fmt: off
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare the references of the forecast errors.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to the .yml configuration file.")
    parser.add_argument("--backend", "-b", type=str, default="slurm", choices=["slurm", "async"], help="Computation backend.")
    parser.add_argument("--setup", action="store_true", help="Only build the grid, the index of the reanalysis and the inventory.")
    parser.add_argument("--stage", type=str, default="references", choices=["references", "baselines"], help="References (observations, climatology), then baselines.")
    args    = parser.parse_args()

    # Light steps, run once on the login node (the inventory opens a single connection to Climato)
    if args.setup:
        PATH_PREPARATION.mkdir(parents=True, exist_ok=True)
        build_structure(), index_reanalysis(), build_inventory()
        raise SystemExit
    config  = load_configuration(args.config)[0]
    cluster = config["Cluster"]
    chunk   = config["Preparation"]["days_per_job"]

    # The runs are listed by the inventory, done before submitting the baselines
    with open(PATH_PREPARATION / "runs.json") as file:
        runs = list(json.load(file))

    products  = [*PRODUCTS_GRIDDED, "alongtrack", "insitu"]
    clim_days = [d.strftime("%m-%d") for d in pd.date_range("2000-01-01", "2000-12-31")]
    clim_days = [clim_days[i : i + chunk] for i in range(0, len(clim_days), chunk)]
    days      = [str(d.date()) for d in DAYS]
    days      = [days[i : i + chunk] for i in range(0, len(days), chunk)]
    runs      = [runs[i : i + chunk] for i in range(0, len(runs), chunk)]

    # The jobs run the code as it is at submission
    freeze_modules([errors, comparisons, observations])

    @job(array=len(products), **cluster["observations"])
    def prepare_observations(i: int) -> None:
        r"""Prepare one observation product on the grid of the model.

        Arguments:
            i : Index of the product.
        """

        {"alongtrack": prepare_alongtrack, "insitu": prepare_insitu}.get(products[i], lambda: prepare_gridded(products[i]))()

    @job(**cluster["light"])
    def template() -> None:
        r"""Create the empty store of the climatology."""
        climatology_template()

    @after(template)
    @job(array=len(clim_days), **cluster["climatology"])
    def climatology(i: int) -> None:
        r"""Compute the climatology of a chunk of days of the year.

        Arguments:
            i : Index of the chunk.
        """

        compute_climatology(clim_days[i])

    @job(array=len(days), **cluster["baselines"])
    def baselines_days(i: int) -> None:
        r"""Compare the climatology and the reanalysis of a chunk of days.

        Arguments:
            i : Index of the chunk.
        """

        for date in days[i]:
            compute_day(date)

    @job(array=len(runs), **cluster["baselines"])
    def baselines_runs(i: int) -> None:
        r"""Compare the persistence of a chunk of runs.

        Arguments:
            i : Index of the chunk.
        """

        for run in runs[i]:
            compute_persistence(run)

    # The baselines need the references: they are submitted once the references are complete
    jobs = [climatology, prepare_observations] if args.stage == "references" else [baselines_days, baselines_runs]
    schedule(*jobs, name=f"FERR-{args.stage.upper()}", backend=args.backend, export="ALL")
