r"""Comparisons of the forecast errors: one run of the forecast or of the persistence, or one day of the baselines."""

import numpy as np
import os
import pandas as pd
import shutil
import subprocess
import time
import xarray as xr

from concurrent.futures import ThreadPoolExecutor
from errors import (
    CORRECTIONS,
    HYPOXIA_BINS,
    LEVELS,
    PATH_DAYS,
    PATH_RUNS,
    PATH_TRANSIT,
    STATISTICS,
    TRANSFORMS,
    WEIGHTINGS,
    ZONES,
    compare_fields,
    compare_hypoxia,
    compare_observations,
    load_product,
    products,
    read_climatology,
    read_reanalysis,
    read_state,
)
from pathlib import Path

from neptune.data import DATASET_VARIABLES

# fmt: off
#
# ----- Remote forecasts on Climato, and the runs whose 3D error maps are kept (a Tuesday and a Thursday per season)
#
CLIMATO      = "climato"
SSH_OPTIONS  = "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=60"
RUNS_MAPS_3D = ["2025-01-14", "2025-01-16", "2025-04-15", "2025-04-17", "2025-07-15", "2025-07-17", "2025-10-14", "2025-10-16"]
STATUS       = ["ok", "missing", "copied"]
# fmt: on


def compare_all(
    f: dict, o: dict, c: dict, date: str, observations: bool = True
) -> dict[str, np.ndarray]:
    r"""Compare the fields of a system on one day to the reanalysis, its hypoxia and every observation product.

    Arguments:
        f            : Fields of the evaluated system, per variable.
        o            : Fields of the reanalysis, per variable, or None to skip the comparisons to it.
        c            : Fields of the climatology of the day, per variable.
        date         : Date string 'YYYY-MM-DD'.
        observations : Whether to compare to the observations.

    Returns:
        sums : Ingredients per group ('vs_reanalysis', 'contingency', 'extent', 'histogram', 'vs_<product>').
    """

    sums = {}
    if o is not None:
        sums["vs_reanalysis"] = compare_fields(f, o, c)
        sums.update(compare_hypoxia(f["DOX"], o["DOX"]))
    if observations:
        for product in products():
            sums[f"vs_{product}"] = compare_observations(f, c, date, product)

    return sums


def empty_like(sums: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r"""Zeros shaped as the ingredients of a day, for a missing or copied day.

    Arguments:
        sums : Ingredients of a valid day.

    Returns:
        zeros : Same groups, filled with zeros.
    """

    return {name: np.zeros_like(x) for name, x in sums.items()}


def to_datasets(
    days: list[dict[str, np.ndarray]], dim: str, labels: list
) -> dict[str, xr.Dataset]:
    r"""Stack the ingredients of several days into one dataset per group of the results file.

    Arguments:
        days   : Ingredients of each day, as returned by compare_all.
        dim    : Name of the stacking dimension, 'lead' or 'day'.
        labels : Coordinate of the stacking dimension.

    Returns:
        datasets : Datasets per group ('vs_reanalysis', 'hypoxia', 'vs_<product>').
    """

    stack = {name: np.stack([d[name] for d in days]) for name in days[0]}
    datasets = {}
    if "vs_reanalysis" in stack:
        datasets["vs_reanalysis"] = xr.Dataset(
            {"ingredients": ((dim, "variable", "level", "zone", "weighting", "transform", "statistic"), stack["vs_reanalysis"])},
            coords={
                dim: labels, "variable": DATASET_VARIABLES, "level": [*range(1, LEVELS + 1), "volume"], "zone": ZONES,
                "weighting": WEIGHTINGS, "transform": TRANSFORMS, "statistic": STATISTICS,
            },
        )  # fmt: skip
        datasets["hypoxia"] = xr.Dataset(
            {
                "contingency" : ((dim, "cell"), stack["contingency"]),
                "extent"      : ((dim, "source", "measure"), stack["extent"]),
                "histogram"   : ((dim, "truth", "bin"), stack["histogram"]),
            },
            coords={
                dim: labels, "cell": ["hit", "false_alarm", "miss", "correct_negative"], "source": ["system", "reanalysis"],
                "measure": ["area", "volume"], "truth": ["hypoxic", "not_hypoxic"], "bin": HYPOXIA_BINS[:-1],
            },
        )  # fmt: skip
    for name in (n for n in stack if n.startswith("vs_") and n != "vs_reanalysis"):
        datasets[name] = xr.Dataset(
            {"ingredients": ((dim, "subset", "weighting", "transform", "statistic"), stack[name])},
            coords={
                dim: labels, "subset": load_product(name[3:])["subsets"], "weighting": WEIGHTINGS,
                "transform": TRANSFORMS, "statistic": STATISTICS,
            },
        )  # fmt: skip

    return datasets


def write(path: Path, groups: dict[str, xr.Dataset]) -> None:
    r"""Write the groups of a results file, under a temporary name renamed once complete.

    Arguments:
        path   : Final path of the netCDF file.
        groups : Datasets per group, e.g. 'forecast/vs_reanalysis'.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    for group, ds in groups.items():
        encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
        ds.to_netcdf(
            temporary, group=group, mode="a" if temporary.exists() else "w", encoding=encoding
        )
    os.replace(temporary, path)


def run_dates(run: str) -> list[str]:
    r"""Dates of the leads 0 to 10 of a run, from J-1 to J+9.

    Arguments:
        run : Date J of the run 'YYYY-MM-DD'.

    Returns:
        dates : Date of each lead 'YYYY-MM-DD'.
    """

    return [
        str(d.date()) for d in pd.date_range(pd.Timestamp(run) - pd.Timedelta(days=1), periods=11)
    ]


def transfer(run: str, files: list[tuple[str, int]]) -> subprocess.Popen:
    r"""Start the transfer of the files of a run from Climato, with a single connection.

    Arguments:
        run   : Date J of the run 'YYYY-MM-DD'.
        files : Remote path and size of each file of the run.

    Returns:
        process : The running rsync.
    """

    folder = PATH_TRANSIT / run
    folder.mkdir(parents=True, exist_ok=True)
    remote = str(Path(files[0][0]).parent)
    listing = folder / ".files"
    listing.write_text("\n".join(Path(p).name for p, _ in files) + "\n")

    return subprocess.Popen(
        ["rsync", "-t", "--no-p", "--no-g", "--chmod=ug=rwX", "--partial", "--timeout=300", "-e", SSH_OPTIONS,
         f"--files-from={listing}", f"{CLIMATO}:{remote}/", f"{folder}/"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )  # fmt: skip


def wait_for(paths: list[Path], sizes: list[int], process: subprocess.Popen) -> bool:
    r"""Wait until files are fully transferred, or until the transfer stops.

    Arguments:
        paths   : Local paths of the files.
        sizes   : Expected sizes [bytes].
        process : The running rsync.

    Returns:
        complete : Whether every file arrived with its expected size.
    """

    while True:
        if all(p.exists() and p.stat().st_size == s for p, s in zip(paths, sizes, strict=False)):
            return True
        if process.poll() is not None:
            return all(
                p.exists() and p.stat().st_size == s for p, s in zip(paths, sizes, strict=False)
            )
        time.sleep(5)


def compute_run(run: str, files: list[tuple[str, int]]) -> None:
    r"""Transfer one forecast run and compare it, day by day, deleting each day once compared.

    Arguments:
        run   : Date J of the run 'YYYY-MM-DD'.
        files : Remote path and size of each file of the run (the 4 kinds of the 11 days found on Climato).
    """

    start, log = time.time(), {"run": run}
    process = transfer(run, files)
    folder = PATH_TRANSIT / run
    by_date = {}
    for path, size in files:
        date = Path(path).name.split("_")[-1].split("-")[0]
        kind = next(k for k in ("grid_T", "grid_U", "grid_V", "ptrc_T") if f"_{k}_" in path)
        by_date.setdefault(f"{date[:4]}-{date[4:6]}-{date[6:]}", {})[kind] = (
            folder / Path(path).name,
            size,
        )

    forecast, persistence, status, corrections, surface, volume = [], [], [], [], [], []
    initial, previous = None, None
    pool = ThreadPoolExecutor(2)
    for date in run_dates(run):
        entries = by_date.get(date, {})
        ok = len(entries) == 4 and wait_for(
            [p for p, _ in entries.values()], [s for _, s in entries.values()], process
        )
        if not ok:
            status.append(1)
            (
                forecast.append(None),
                persistence.append(None),
                corrections.append(np.zeros((len(DATASET_VARIABLES), 2))),
            )
            surface.append(None), volume.append(None)
            continue

        f, counts = read_state({k: p for k, (p, _) in entries.items()})
        copied = previous is not None and np.array_equal(
            f["votemper"], previous["votemper"], equal_nan=True
        )
        previous, initial = f, initial or f
        o, c = read_reanalysis(date), read_climatology(date)

        status.append(2 if copied else 0)
        corrections.append(counts)
        # The forecast and its persistence are compared at the same time, numpy releasing the GIL
        days = (
            [None, None]
            if copied
            else [
                x.result() for x in [pool.submit(compare_all, g, o, c, date) for g in (f, initial)]
            ]
        )
        forecast.append(days[0]), persistence.append(days[1])
        surface.append(
            np.stack([
                (f[n] - o[n])[..., 0, :, :] if f[n].ndim == 3 else f[n] - o[n]
                for n in DATASET_VARIABLES
            ]).astype(np.float32)
        )
        if run in RUNS_MAPS_3D:
            volume.append(
                np.stack([
                    f[n] - o[n]
                    if f[n].ndim == 3
                    else np.pad(
                        (f[n] - o[n])[None],
                        ((0, LEVELS - 1), (0, 0), (0, 0)),
                        constant_values=np.nan,
                    )
                    for n in DATASET_VARIABLES
                ]).astype(np.float32)
            )
        for p, _ in entries.values():
            p.unlink()

    stderr = process.communicate()[1].decode()
    reference = next((x for x in forecast if x is not None), None)
    if reference is None:
        raise RuntimeError(f"ERROR - No day of the run {run} could be compared: {stderr[-500:]}")

    fill = lambda days: [x if x is not None else empty_like(reference) for x in days]
    leads = list(range(11))
    groups = {f"forecast/{k}": v for k, v in to_datasets(fill(forecast), "lead", leads).items()}
    groups |= {
        f"persistence_forecast/{k}": v
        for k, v in to_datasets(fill(persistence), "lead", leads).items()
    }

    nan = np.full(next(x for x in surface if x is not None).shape, np.nan, dtype=np.float32)
    groups["forecast/vs_reanalysis"] = (
        groups["forecast/vs_reanalysis"]
        .assign(
            corrections=(("lead", "variable", "correction"), np.stack(corrections)),
            status=(("lead",), np.array(status, dtype=np.int8)),
        )
        .assign_coords(correction=CORRECTIONS)
    )
    groups["forecast/maps"] = xr.Dataset(
        {
            "surface_error": (
                ("lead", "variable", "y", "x"),
                np.stack([x if x is not None else nan for x in surface]),
            )
        },
        coords={"lead": leads, "variable": DATASET_VARIABLES},
    )
    if run in RUNS_MAPS_3D:
        nan3 = np.full(next(x for x in volume if x is not None).shape, np.nan, dtype=np.float32)
        groups["forecast/maps"]["volume_error"] = (
            ("lead", "variable", "z", "y", "x"),
            np.stack([x if x is not None else nan3 for x in volume]),
        )

    write(PATH_RUNS / f"{run}.nc", groups)
    shutil.rmtree(folder, ignore_errors=True)

    log |= {
        "bytes": sum(s for _, s in files), "seconds": round(time.time() - start), "missing": status.count(1),
        "copied": status.count(2), "rsync": process.returncode, "node": os.uname().nodename,
    }  # fmt: skip
    print(log, flush=True)


def compute_persistence(run: str) -> None:
    r"""Compare the persistence of a run (the reanalysis of J-1 held constant) on its 11 days.

    Arguments:
        run : Date J of the run 'YYYY-MM-DD'.
    """

    dates = run_dates(run)
    initial = read_reanalysis(dates[0])
    days = [compare_all(initial, read_reanalysis(d), read_climatology(d), d) for d in dates]
    groups = {f"persistence/{k}": v for k, v in to_datasets(days, "lead", list(range(11))).items()}
    write(PATH_RUNS.parent / "persistence" / f"{run}.nc", groups)


def compute_day(date: str) -> None:
    r"""Compare the climatology and the reanalysis of one day to the reanalysis and the observations.

    Arguments:
        date : Date string 'YYYY-MM-DD'.
    """

    o, c = read_reanalysis(date), read_climatology(date)
    groups = {
        f"climatology/{k}": v
        for k, v in to_datasets([compare_all(c, o, c, date)], "day", [date]).items()
    }
    groups |= {
        f"reanalysis/{k}": v
        for k, v in to_datasets([compare_all(o, None, c, date)], "day", [date]).items()
    }
    write(PATH_DAYS / f"{date}.nc", groups)
