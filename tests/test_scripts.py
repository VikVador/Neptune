r"""Tests for the scripts building the mask and the statistics."""

import importlib.util
import numpy as np

from pathlib import Path


def load_script(name: str):  # noqa: ANN201
    r"""Load a script of the scripts folder, which is not a package."""

    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_moments_typical() -> None:
    r"""Determines if the moments give the mean and std of each channel and level, ignoring NaNs."""

    run_statistics = load_script("run_statistics")
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, size=(3, 4, 16, 16)).astype(np.float32)
    x[0, 1, :4] = np.nan

    sums = run_statistics.moments(x[:, :, :8]) + run_statistics.moments(x[:, :, 8:])
    mean, std = run_statistics.mean_std(sums)

    assert mean.shape == std.shape == (3, 4)
    assert np.allclose(mean, np.nanmean(x, (-2, -1)), atol=1e-5)
    assert np.allclose(std, np.nanstd(x, (-2, -1)), atol=1e-5)


def test_mean_std_constant() -> None:
    r"""Determines if a constant channel keeps a positive std."""

    run_statistics = load_script("run_statistics")
    _, std = run_statistics.mean_std(run_statistics.moments(np.ones((2, 8, 8))))

    assert (std == 1e-8).all()


def test_outside_bounds_typical() -> None:
    r"""Determines if a point is flagged when one of its variables is outside its bounds, NaN being ignored."""

    run_mask = load_script("run_mask")
    x = np.zeros((2, 3, 4, 4), dtype=np.float32)
    x[0, 0, 0, 0] = -5.0
    x[1, 2, 3, 3] = 50.0
    x[1, 1, 1, 1] = np.nan

    outside = run_mask.outside_bounds(x, ["votemper", "vosaline"])

    assert outside.shape == (3, 4, 4)
    assert outside[0, 0, 0] and outside[2, 3, 3]
    assert outside.sum() == 2
