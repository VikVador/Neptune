r"""Ensemble forecasts of the FGN and their metrics, against the persistence and the daily climatology."""

__all__ = [
    "load_forecaster",
    "forecast_ensemble",
    "compute_metrics",
    "compute_animations",
    "aggregate_metrics",
    "aggregate_maps",
    "clean",
]

import numpy as np
import shutil
import torch
import xarray as xr

from albus.metrics import apply_mask, mean_square_error, root_mean_square_error, skill
from collections.abc import Iterator
from shaggy.tools import load
from torch import Tensor

from neptune.config import PATH_CLIMATOLOGY, PATH_DIAGNOSTICS, PATH_MODELS
from neptune.data import DATASET_VARIABLES_OCEAN, DATASET_VARIABLES_SURFACE, X, Y
from neptune.data.dataset import NeptuneDataset
from neptune.data.weights import get_weights_conditioning, get_weights_mask
from neptune.diagnostics import LAYERS, SPLITS, TRANSLATIONS_METRICS
from neptune.diagnostics.geography import load_depth
from neptune.model import FGN

# Forecast of the FGN and its baselines
BASELINES = ["forecast", "persistence", "climatology"]


def load_forecaster(checkpoint_name: str, split: str, days: int) -> tuple[FGN, NeptuneDataset]:
    r"""Load a FGN and the windows of a split, its N previous states followed by `days` states.

    Arguments:
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
        days            : Number of forecasted days K.

    Returns:
        fgn     : FGN in evaluation mode, on the GPU if available.
        dataset : Standardized windows of the split.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fgn = load(PATH_MODELS / checkpoint_name, FGN, device=device).eval()
    dataset = NeptuneDataset(*SPLITS[split], input_states=fgn.input_states, output_states=days)

    return fgn, dataset


@torch.no_grad()
def forecast_ensemble(
    fgn: FGN,
    x_inp_s: Tensor,
    x_inp_o: Tensor,
    c_inp: Tensor,
    dates: list[str],
    members: int,
    chunk: int,
    noise: list[float],
) -> Iterator[tuple[int, Tensor, Tensor]]:
    r"""Forecast autoregressively an ensemble by chunks of members, yielding each forecasted day of each chunk.

    Arguments:
        fgn     : FGN in evaluation mode.
        x_inp_s : Previous surface states (N, C_s, Y, X).
        x_inp_o : Previous ocean states (N, C_o, Z, Y, X).
        c_inp   : ERA5 conditioning of the last input state of each step (K, C_c, Y, X).
        dates   : Date strings 'YYYY-MM-DD' of the last input state of each step (K,).
        members : Number of ensemble members E.
        chunk   : Number of members forecasted at once.
        noise   : Standard deviation of the noise added to the ERA5 conditioning of each step (K,).

    Returns:
        k   : Index of the forecasted day.
        x_s : Forecasted surface states of the members of the chunk, standardized (chunk, C_s, Y, X).
        x_o : Forecasted ocean states of the members of the chunk, standardized (chunk, C_o, Z, Y, X).
    """

    device = fgn.mask_surface.device

    # A single noisy ERA5 per step is shared by every member, the members being diversified by the noise of the FGN
    conditionings = [
        get_weights_conditioning([date], c_inp[k : k + 1], noise=noise[k], device=device)
        for k, date in enumerate(dates)
    ]

    for e in range(0, members, chunk):
        size = min(chunk, members - e)
        x_s, x_o = (
            x_inp_s.to(device).expand(size, *x_inp_s.shape),
            x_inp_o.to(device).expand(size, *x_inp_o.shape),
        )

        for k, (cond_s, cond_o) in enumerate(conditionings):
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                y_s, y_o = fgn(
                    x_s,
                    x_o,
                    cond_s.expand(size, *cond_s.shape[1:]),
                    cond_o.expand(size, *cond_o.shape[1:]),
                )

            yield k, y_s[:, 0], y_o[:, 0]

            # The window slides by one day: the oldest state goes out, the forecast comes in
            x_s, x_o = torch.cat([x_s[:, 1:], y_s], dim=1), torch.cat([x_o[:, 1:], y_o], dim=1)


def _climatology(date: str, device: torch.device) -> tuple[Tensor, Tensor]:
    r"""Load the daily climatology of the day of the year of a date.

    Arguments:
        date   : Date string 'YYYY-MM-DD'.
        device : Target device.

    Returns:
        clim_s : Surface climatology, in physical units (C_s, Y, X).
        clim_o : Ocean climatology, in physical units (C_o, Z, Y, X).
    """

    with xr.open_zarr(PATH_CLIMATOLOGY) as ds:
        ds = ds.sel(day=date[5:]).load()

    return (
        torch.as_tensor(ds[DATASET_VARIABLES_SURFACE].to_array().values, device=device),
        torch.as_tensor(ds[DATASET_VARIABLES_OCEAN].to_array().values, device=device),
    )


@torch.no_grad()
def compute_metrics(
    checkpoint_name: str,
    split: str,
    indices: list[int],
    days: int,
    members: int,
    chunk: int,
    noise: list[float],
    job: int,
) -> None:
    r"""Compute the mean squared errors and variances of the ensemble forecasts of some windows of a split.

    Arguments:
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
        indices         : Indices of the windows of the split to forecast.
        days            : Number of forecasted days K.
        members         : Number of ensemble members E.
        chunk           : Number of members forecasted at once.
        noise           : Standard deviation of the noise added to the ERA5 conditioning of each step (K,).
        job             : Index of the job, naming the files of the maps.
    """

    fgn, dataset = load_forecaster(checkpoint_name, split, days)
    device, n = fgn.mask_surface.device, fgn.input_states
    mask_s, mask_o = fgn.mask_surface, fgn.mask_ocean
    path = PATH_DIAGNOSTICS / checkpoint_name / split
    maps = {}

    for index in indices:
        x_inp_s, x_inp_o, x_out_s, x_out_o, c_inp, dates = dataset[index]

        # Ground truth and persistence of the last input state, in physical units
        truth_s, truth_o = dataset.unstandardize(x_out_s.to(device), x_out_o.to(device))
        persistence_s, persistence_o = dataset.unstandardize(
            x_inp_s[-1].to(device), x_inp_o[-1].to(device)
        )

        # The errors of each day and their squares are summed over the members, on the CPU, by chunks of members
        sums = [None] * days
        for k, x_s, x_o in forecast_ensemble(
            fgn, x_inp_s, x_inp_o, c_inp, dates[n - 1 : -1], members, chunk, noise
        ):
            x_s, x_o = dataset.unstandardize(x_s.float(), x_o.float())
            errors = (x_s - truth_s[k], x_o - truth_o[k])
            terms = [
                t.cpu()
                for t in [d.sum(dim=0) for d in errors] + [d.pow(2).sum(dim=0) for d in errors]
            ]
            sums[k] = (
                terms if sums[k] is None else [a + b for a, b in zip(sums[k], terms, strict=True)]
            )

        metrics = {}
        for k in range(days):
            clim_s, clim_o = _climatology(dates[n + k], device)
            e_s, e_o, q_s, q_o = (t.to(device) for t in sums[k])

            # Squared errors and variance of the ensemble at each point, NaN on land
            fields = {
                ("forecast", "mse"): (
                    apply_mask((e_s / members) ** 2, mask_s),
                    apply_mask((e_o / members) ** 2, mask_o),
                ),
                ("forecast", "var"): (
                    apply_mask((q_s - e_s**2 / members) / (members - 1), mask_s),
                    apply_mask((q_o - e_o**2 / members) / (members - 1), mask_o),
                ),
            }
            fields |= {
                (baseline, "mse"): (
                    mean_square_error(truth_s[k], y_s, "C Y X", "", mask_s),
                    mean_square_error(truth_o[k], y_o, "C Z Y X", "", mask_o),
                )
                for baseline, (y_s, y_o) in (
                    ("persistence", (persistence_s, persistence_o)),
                    ("climatology", (clim_s, clim_o)),
                )
            }

            # The metrics are averaged over the sea, per variable and level
            for key, (s, o) in fields.items():
                metrics.setdefault(key, ([], []))
                metrics[key][0].append(s.nanmean(dim=(-2, -1)))
                metrics[key][1].append(o.nanmean(dim=(-2, -1)))

            # The maps are kept at the surface and averaged along each axis of the ocean, then summed over the windows
            if k in (0, days - 1):
                (mse_s, mse_o), (var_s, var_o) = (
                    fields[("forecast", "mse")],
                    fields[("forecast", "var")],
                )
                terms = {"surface": torch.stack([mse_s, var_s])}
                terms |= {
                    name: torch.stack([mse_o.nanmean(dim=axis), var_o.nanmean(dim=axis)])
                    for name, axis in (("Z", 1), ("X", 3), ("Y", 2))
                }
                summed = maps.setdefault(k + 1, {"sums": dict.fromkeys(terms, 0), "count": 0})
                summed["sums"] = {name: summed["sums"][name] + terms[name] for name in terms}
                summed["count"] += 1

        (path / "metrics").mkdir(parents=True, exist_ok=True)
        torch.save(
            {key: (torch.stack(s).cpu(), torch.stack(o).cpu()) for key, (s, o) in metrics.items()},
            path / "metrics" / f"{dates[n - 1]}.pt",
        )

    for day, summed in maps.items():
        folder = path / "maps" / f"day_{day:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sums": {name: t.cpu() for name, t in summed["sums"].items()},
                "count": summed["count"],
            },
            folder / f"{job}.pt",
        )


def _slice(x: Tensor, plane: str, position: int) -> Tensor:
    r"""Slice ocean fields along a plane, at a level (x-y), a longitude (y-z) or a latitude (z-x).

    Arguments:
        x        : Ocean fields (..., Z, Y, X).
        plane    : Plane of the slice, 'x-y', 'y-z' or 'z-x'.
        position : Index of the level, of the longitude or of the latitude of the slice.

    Returns:
        x : Slice of the fields, (..., Y, X), (..., Z, Y) or (..., Z, X).
    """

    if plane == "x-y":
        return x[..., position, :, :]
    if plane == "y-z":
        return x[..., :, :, position]

    return x[..., :, position, :]


def _cut(x_s: Tensor, x_o: Tensor, c: int | None, plane: str, p: int | None) -> Tensor:
    r"""Cut a slice of a surface variable (position None) or of an ocean variable along a plane.

    Arguments:
        x_s   : Surface fields (..., C_s, Y, X), or a single one if `c` is None.
        x_o   : Ocean fields (..., C_o, Z, Y, X), or a single one if `c` is None.
        c     : Index of the variable in its group, or None for single fields.
        plane : Plane of the slice, 'x-y', 'y-z' or 'z-x'.
        p     : Index of the level, of the longitude or of the latitude of the slice, None at the surface.

    Returns:
        x : Slice of the fields.
    """

    if p is None:
        return x_s if c is None else x_s[..., c, :, :]

    return _slice(x_o if c is None else x_o[..., c, :, :, :], plane, p)


def _sections(count: int) -> tuple[list[int], list[int]]:
    r"""Find evenly spaced vertical slices across the domain, its edges excluded.

    Arguments:
        count : Number of slices along each horizontal axis.

    Returns:
        columns : Indices of the longitudes of the y-z slices.
        rows    : Indices of the latitudes of the z-x slices.
    """

    columns, rows = (
        np.linspace(0, n - 1, count + 2)[1:-1].round().astype(int).tolist() for n in (X, Y)
    )

    return columns, rows


@torch.no_grad()
def compute_animations(
    checkpoint_name: str,
    split: str,
    index: int,
    days: int,
    members: int,
    chunk: int,
    noise: list[float],
    levels: list[int],
    sections: int,
) -> None:
    r"""Forecast an ensemble from a window of a split and save the fields of its animations.

    Arguments:
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
        index           : Index of the window of the split to forecast.
        days            : Number of forecasted days K.
        members         : Number of ensemble members E.
        chunk           : Number of members forecasted at once.
        noise           : Standard deviation of the noise added to the ERA5 conditioning of each step (K,).
        levels          : Indices of the levels of the horizontal slices of the ocean variables.
        sections        : Number of vertical slices along each horizontal axis (y-z and z-x).
    """

    fgn, dataset = load_forecaster(checkpoint_name, split, days)
    device, n = fgn.mask_surface.device, fgn.input_states

    x_inp_s, x_inp_o, x_out_s, x_out_o, c_inp, dates = dataset[index]
    truth_s, truth_o = dataset.unstandardize(x_out_s.to(device), x_out_o.to(device))

    # Slices of the surface variables (position None), and of the ocean variables at each level, longitude and latitude
    columns, rows = _sections(sections)
    slices = [(var, c, "x-y", None) for c, var in enumerate(DATASET_VARIABLES_SURFACE)]
    for plane, positions in (("x-y", levels), ("y-z", columns), ("z-x", rows)):
        slices += [
            (var, c, plane, p) for c, var in enumerate(DATASET_VARIABLES_OCEAN) for p in positions
        ]

    # The slices of every member and day are gathered on the CPU by chunks of members
    gathered = {(var, plane, p): [[] for _ in range(days)] for var, _, plane, p in slices}
    for k, x_s, x_o in forecast_ensemble(
        fgn, x_inp_s, x_inp_o, c_inp, dates[n - 1 : -1], members, chunk, noise
    ):
        x_s, x_o = dataset.unstandardize(x_s.float(), x_o.float())
        for var, c, plane, p in slices:
            gathered[(var, plane, p)][k].append(_cut(x_s, x_o, c, plane, p).cpu())

    # Each frame holds the ground truth, the ensemble mean and median, 8 members and the skills of the day, NaN on land
    nan = float("nan")
    frames = {}
    for var, c, plane, p in slices:
        mask = _cut(fgn.mask_surface[0], fgn.mask_ocean[0], None, plane, p).cpu()
        steps = []
        for k in range(days):
            x, truth = (
                torch.cat(gathered[(var, plane, p)][k]),
                _cut(truth_s[k], truth_o[k], c, plane, p).cpu(),
            )
            median = x.quantile(0.5, dim=0)
            steps.append({
                "truth": truth.masked_fill(mask == 0, nan),
                "mean": x.mean(dim=0).masked_fill(mask == 0, nan),
                "median": median.masked_fill(mask == 0, nan),
                "members": x[:8].masked_fill(mask == 0, nan),
                "skill": skill(truth, x, "E A B", "E", "A B", mask),
                "skill_median": root_mean_square_error(truth, median, "A B", "A B", mask),
            })
        frames[(var, plane, p)] = {
            key: torch.stack([step[key] for step in steps]) for key in steps[0]
        }

    # One file per variable, the frames of each slice being stacked along the days
    folder = PATH_DIAGNOSTICS / checkpoint_name / split / "animations" / dates[n - 1]
    folder.mkdir(parents=True, exist_ok=True)
    for var in DATASET_VARIABLES_SURFACE + DATASET_VARIABLES_OCEAN:
        torch.save(
            {(plane, p): f for (name, plane, p), f in frames.items() if name == var},
            folder / f"{var}.pt",
        )


def _scores(mse: Tensor, var: Tensor, members: int) -> dict[str, Tensor]:
    r"""Compute the skill, the spread and their ratio, corrected for the ensemble size, from mean squared values.

    Arguments:
        mse     : Mean squared error of the ensemble mean.
        var     : Mean variance of the ensemble.
        members : Number of ensemble members E.

    Returns:
        scores : Skill, spread and ratio, with the shape of `mse`.
    """

    skill, spread = mse.sqrt(), var.sqrt()

    return {
        "skill": skill,
        "spread": spread,
        "ratio": ((members + 1) / members) ** 0.5 * spread / skill,
    }


def aggregate_metrics(
    checkpoint_name: str, split: str, members: int
) -> dict[tuple[str, str], tuple[Tensor, Tensor, Tensor]]:
    r"""Aggregate the metrics of every forecast of a split, per variable and level, and per layer.

    Arguments:
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
        members         : Number of ensemble members E.

    Returns:
        metrics : Skill, spread and ratio of the forecast and skill of the baselines, for the surface (K, C_s)
                  and the ocean variables, per level (K, C_o, Z) and per layer (K, C_o, L).
    """

    paths = sorted((PATH_DIAGNOSTICS / checkpoint_name / split / "metrics").glob("*.pt"))
    files = [torch.load(path, weights_only=True) for path in paths]

    # Each layer weights its levels by their number of sea points (Z, L)
    _, mask_o = get_weights_mask()
    layers = torch.as_tensor(np.digitize(load_depth(), LAYERS))
    weights = mask_o[0].sum(dim=(-2, -1))[:, None] * torch.nn.functional.one_hot(
        layers, len(LAYERS) + 1
    )
    weights = weights / weights.sum(dim=0)

    means = {}
    for key in files[0]:
        s, o = (torch.stack([f[key][i] for f in files]).mean(dim=0) for i in range(2))
        means[key] = (s, o, o @ weights)

    scores = [
        _scores(mse, var, members)
        for mse, var in zip(means[("forecast", "mse")], means[("forecast", "var")], strict=True)
    ]
    metrics = {
        ("forecast", metric): tuple(score[metric] for score in scores)
        for metric in TRANSLATIONS_METRICS
    }
    metrics |= {
        (baseline, "skill"): tuple(mse.sqrt() for mse in means[(baseline, "mse")])
        for baseline in BASELINES[1:]
    }

    return metrics


def aggregate_maps(
    checkpoint_name: str, split: str, day: int, members: int
) -> dict[str, dict[str, Tensor]]:
    r"""Aggregate the maps of the metrics of a forecasted day over the forecasts of a split.

    Arguments:
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
        day             : Forecasted day, from 1 to K.
        members         : Number of ensemble members E.

    Returns:
        maps : Skill, spread and ratio, NaN on land, at the surface (C_s, Y, X) and along each axis, 'Z'
               (C_o, Y, X), 'X' (C_o, Z, Y) and 'Y' (C_o, Z, X).
    """

    sums, count = {}, 0
    for path in (PATH_DIAGNOSTICS / checkpoint_name / split / "maps" / f"day_{day:02d}").glob(
        "*.pt"
    ):
        file = torch.load(path, weights_only=True)
        sums = {name: sums.get(name, 0) + t for name, t in file["sums"].items()}
        count += file["count"]

    scores = {name: _scores(*t / count, members) for name, t in sums.items()}

    return {
        metric: {name: score[metric] for name, score in scores.items()}
        for metric in TRANSLATIONS_METRICS
    }


def clean(checkpoint_name: str, split: str) -> None:
    r"""Delete the intermediate results of the diagnostics of a split, once aggregated.

    Arguments:
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
    """
    shutil.rmtree(PATH_DIAGNOSTICS / checkpoint_name / split, ignore_errors=True)
