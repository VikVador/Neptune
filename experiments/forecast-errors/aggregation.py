r"""Gather the ingredients of every run and day, then compute the metrics of each group of runs."""

import argparse
import errors
import json
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from comparisons import run_dates
from dawgz import after, job, schedule
from errors import PATH_DAYS, PATH_PREPARATION, PATH_RESULTS, PATH_RUNS
from pathlib import Path

from neptune.tools import freeze_modules, load_configuration

# fmt: off
#
# ----- Systems, with the folder of their files and how they are stacked
#
SYSTEMS = {
    "forecast"             : (PATH_RUNS, "lead"),
    "persistence_forecast" : (PATH_RUNS, "lead"),
    "persistence"          : (PATH_RUNS.parent / "persistence", "lead"),
    "climatology"          : (PATH_DAYS, "day"),
    "reanalysis"           : (PATH_DAYS, "day"),
}

# Metrics computed per run then averaged (correlations, ratio of variances), the others from the pooled sums
METRICS_PER_RUN = ["std_ratio", "spatial_correlation", "anomaly_correlation", "anomaly_correlation_uncentred"]
METRICS_POOLED  = ["count", "bias", "mse", "rmse", "unbiased_rmse", "mae", "percent_bias", "nash_sutcliffe",
                   "multiplicative_bias", "multiplicative_error"]
SEASONS         = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}
# fmt: on


def groups_of_runs(runs: list[str], versions: dict[str, str]) -> dict[str, np.ndarray]:
    r"""Groups of runs used to aggregate the ingredients: all, by weekday, by season and by version.

    Arguments:
        runs     : Dates J of the runs.
        versions : Version of the system of each run.

    Returns:
        groups : Membership of each run to each group.
    """

    dates = pd.DatetimeIndex(runs)
    groups = {
        "all": np.ones(len(runs), bool),
        "tuesday": dates.dayofweek == 1,
        "other": dates.dayofweek != 1,
    }
    groups |= {name: dates.month.isin(months) for name, months in SEASONS.items()}
    groups |= {
        v.lower().replace("nrt_", ""): np.array([versions[r] == v for r in runs])
        for v in sorted(set(versions.values()))
    }

    return groups


def read(path: Path, group: str) -> dict[str, np.ndarray]:
    r"""Read the variables of a group of a results file, opening it once with netCDF4.

    Arguments:
        path  : Path of the netCDF file.
        group : Group to read, e.g. '/forecast/vs_reanalysis'.

    Returns:
        arrays : Values of each data variable of the group.
    """

    with netCDF4.Dataset(path) as ds:
        node = ds[group]
        return {
            name: np.asarray(v[:])
            for name, v in node.variables.items()
            if name not in node.dimensions
        }


def gather() -> None:
    r"""Gather the ingredients of the runs and days into one store, every system stacked on (run, lead)."""

    with open(PATH_PREPARATION / "runs.json") as file:
        inventory = json.load(file)
    runs = sorted(
        r
        for r in inventory
        if all((SYSTEMS[s][0] / f"{r}.nc").exists() for s in ("forecast", "persistence"))
    )
    store = PATH_RESULTS / "ingredients.zarr"

    for system, (folder, dim) in SYSTEMS.items():
        with xr.open_datatree(next(folder.glob("*.nc")), engine="netcdf4") as tree:
            templates = {
                g: tree[g].to_dataset().load()
                for g in tree.groups
                if g.startswith(f"/{system}/") and not g.endswith("/maps")
            }

        for group, template in templates.items():
            # A day of the baselines serves up to 11 runs: each is read once
            if dim == "day":
                days = {
                    d: read(folder / f"{d}.nc", group)
                    for d in sorted({d for r in runs for d in run_dates(r)})
                }
            arrays = {name: [] for name in template.data_vars}
            for run in runs:
                values = (
                    read(folder / f"{run}.nc", group)
                    if dim == "lead"
                    else {n: np.concatenate([days[d][n] for d in run_dates(run)]) for n in arrays}
                )
                for name in arrays:
                    arrays[name].append(values[name])

            template = template.rename(day="lead") if dim == "day" else template
            ds = xr.Dataset(
                {name: (("run", *template[name].dims), np.stack(x)) for name, x in arrays.items()},
                coords={
                    **{k: v for k, v in template.coords.items() if k != "lead"},
                    "run": runs,
                    "lead": range(11),
                },
                attrs={
                    "description": f"Ingredients of the {system} against {group.split('/')[-1]}, per run and lead.",
                    "versions": json.dumps({r: inventory[r]["version"] for r in runs}),
                },
            )
            ds.to_zarr(store, group=group.strip("/"), mode="w")


def metrics(sums: xr.DataArray) -> xr.Dataset:
    r"""Compute the metrics from the ingredients.

    Arguments:
        sums : Ingredients, with a dimension 'statistic'.

    Returns:
        metrics : One variable per metric, NaN where there is no point.
    """

    s = {name: sums.sel(statistic=name, drop=True) for name in sums["statistic"].values}
    w = s["sum_weights"].where(s["sum_weights"] > 0)
    mean_f, mean_o = s["sum_forecast"] / w, s["sum_reference"] / w
    var_f, var_o = (
        s["sum_squared_forecast"] / w - mean_f**2,
        s["sum_squared_reference"] / w - mean_o**2,
    )
    cov = s["sum_product"] / w - mean_f * mean_o
    mean_fa, mean_oa = s["sum_forecast_anomaly"] / w, s["sum_reference_anomaly"] / w
    var_fa, var_oa = (
        s["sum_squared_forecast_anomaly"] / w - mean_fa**2,
        s["sum_squared_reference_anomaly"] / w - mean_oa**2,
    )
    cov_a = s["sum_product_anomaly"] / w - mean_fa * mean_oa

    bias, mse, mae = s["sum_error"] / w, s["sum_squared_error"] / w, s["sum_absolute_error"] / w
    log10 = sums["transform"] == "log10" if "transform" in sums.dims else False

    return xr.Dataset({
        "count"                         : s["count"],
        "bias"                          : bias,
        "mse"                           : mse,
        "rmse"                          : np.sqrt(mse),
        "unbiased_rmse"                 : np.sqrt((mse - bias**2).clip(min=0)),
        "mae"                           : mae,
        "std_ratio"                     : np.sqrt(var_f / var_o),
        "spatial_correlation"           : cov / np.sqrt(var_f * var_o),
        "anomaly_correlation"           : cov_a / np.sqrt(var_fa * var_oa),
        "anomaly_correlation_uncentred" : s["sum_product_anomaly"] / np.sqrt(s["sum_squared_forecast_anomaly"] * s["sum_squared_reference_anomaly"]),
        "percent_bias"                  : 100 * s["sum_error"] / s["sum_reference"],
        "nash_sutcliffe"                : 1 - s["sum_squared_error"] / (s["sum_squared_reference"] - s["sum_reference"] ** 2 / w),
        "multiplicative_bias"           : (10**bias).where(log10),
        "multiplicative_error"          : (10**mae).where(log10),
    })  # fmt: skip


def hypoxia_metrics(ds: xr.Dataset) -> xr.Dataset:
    r"""Compute the scores of the hypoxia from the pooled contingency table, extents and histograms.

    Arguments:
        ds : Sums of the contingency table, extents and histograms of a group of runs.

    Returns:
        scores : Scores per lead, and the hit and false alarm rates of each threshold (ROC curve).
    """

    a, b, c, d = (
        ds["contingency"].sel(cell=x, drop=True)
        for x in ["hit", "false_alarm", "miss", "correct_negative"]
    )
    n = a + b + c + d
    hit_rate, false_rate = a / (a + c), b / (b + d)
    random = (a + b) * (a + c) / n
    log = lambda x: np.log(x.where((x > 0) & (x < 1)))
    cumulative = ds["histogram"].cumsum("bin")

    return xr.Dataset({
        "accuracy"         : (a + d) / n,
        "frequency_bias"   : (a + b) / (a + c),
        "recall"           : hit_rate,
        "precision"        : a / (a + b),
        "f1"               : 2 * a / (2 * a + b + c),
        "csi"              : a / (a + b + c),
        "ets"              : (a - random) / (a + b + c - random),
        "sedi"             : (log(false_rate) - log(hit_rate) + log(1 - hit_rate) - log(1 - false_rate))
                             / (log(false_rate) + log(hit_rate) + log(1 - hit_rate) + log(1 - false_rate)),
        "extent_error"     : ds["extent"].sel(source="system", drop=True) - ds["extent"].sel(source="reanalysis", drop=True),
        "roc_hit_rate"     : cumulative.sel(truth="hypoxic", drop=True) / ds["histogram"].sel(truth="hypoxic", drop=True).sum("bin"),
        "roc_false_rate"   : cumulative.sel(truth="not_hypoxic", drop=True) / ds["histogram"].sel(truth="not_hypoxic", drop=True).sum("bin"),
    })  # fmt: skip


def group_metrics(ds: xr.Dataset, groups: dict[str, np.ndarray]) -> xr.Dataset:
    r"""Compute the metrics of each group of runs: pooled sums for the errors, per run then averaged for the others.

    Arguments:
        ds     : Ingredients of every run.
        groups : Membership of each run to each group.

    Returns:
        metrics : Metrics with a dimension 'group'.
    """

    ingredients = ds["ingredients"].load()
    parts = []
    for member in groups.values():
        sel = ingredients.isel(run=np.flatnonzero(member))
        pooled = metrics(sel.sum("run"))[METRICS_POOLED]
        per_run = metrics(sel)[METRICS_PER_RUN].mean("run", skipna=True)
        parts.append(xr.merge([pooled, per_run]))

    return xr.concat(parts, "group").assign_coords(group=np.array(list(groups)))


def compute_metrics() -> None:
    r"""Compute the metrics of every system, reference and group of runs, pooled or averaged per run."""

    tree = xr.open_datatree(PATH_RESULTS / "ingredients.zarr", engine="zarr")
    for node in tree.subtree:
        if node.is_empty or node.parent is None or node.parent.is_root:
            continue
        ds = node.to_dataset()
        groups = groups_of_runs(list(ds["run"].values), json.loads(ds.attrs["versions"]))
        if node.name == "hypoxia":
            ds = ds.load()
            parts = [
                hypoxia_metrics(ds.isel(run=np.flatnonzero(m)).sum("run")) for m in groups.values()
            ]
            result = xr.concat(parts, "group").assign_coords(group=np.array(list(groups)))
        elif "variable" in ds.dims:
            # One variable at a time, the per-run metrics of every run not fitting in memory at once
            result = xr.concat(
                [group_metrics(ds.sel(variable=[n]), groups) for n in ds["variable"].values],
                "variable",
            )
        else:
            result = group_metrics(ds, groups)
        result.attrs = {
            "description": f"Metrics of {node.path}: pooled sums for the errors, per run then averaged for {', '.join(METRICS_PER_RUN)}."
        }
        result.to_zarr(PATH_RESULTS / "metrics.zarr", group=node.path.strip("/"), mode="w")

    # Skill of the forecast against the persistence and the climatology, on the same reference
    store = PATH_RESULTS / "metrics.zarr"
    for system in ("forecast", "persistence_forecast"):
        for reference in tree[system].children:
            if reference == "hypoxia":
                continue
            ds = xr.open_zarr(store, group=f"{system}/{reference}")
            skill = {
                f"skill_vs_{base}": 1
                - ds["mse"] / xr.open_zarr(store, group=f"{base}/{reference}")["mse"]
                for base in ("persistence", "climatology")
            }
            xr.Dataset(skill).to_zarr(store, group=f"{system}/{reference}", mode="a")


# fmt: off
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Aggregate the forecast errors.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to the .yml configuration file.")
    parser.add_argument("--backend", "-b", type=str, default="slurm", choices=["slurm", "async"], help="Computation backend.")
    args    = parser.parse_args()
    config  = load_configuration(args.config)[0]
    cluster = config["Cluster"]

    # The jobs run the code as it is at submission
    freeze_modules([errors])

    @job(**cluster["aggregation"])
    def ingredients() -> None:
        r"""Gather the ingredients of every run and day."""
        gather()

    @after(ingredients)
    @job(**cluster["aggregation"])
    def scores() -> None:
        r"""Compute the metrics of every group of runs."""
        compute_metrics()

    schedule(scores, name="FERR-AGG", backend=args.backend, export="ALL")
