r"""Tests for the forecast errors experiment."""

import numpy as np
import sys
import xarray as xr

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "forecast-errors"))

from aggregation import hypoxia_metrics, metrics
from errors import STATISTICS, ingredients, kopelevich_regions, remove_pass_mean, transform
from observations import pressure_to_depth


def _metrics(f: np.ndarray, o: np.ndarray, c: np.ndarray, w: np.ndarray) -> xr.Dataset:
    r"""Metrics of one group computed from the ingredients of the points."""

    sums = ingredients(f, o, c, w, np.ones((f.size, 1)))[0]
    return metrics(xr.DataArray(sums, dims=["statistic"], coords={"statistic": STATISTICS}))


def test_ingredients_typical() -> None:
    r"""Determines if the metrics from the ingredients match their direct computation."""

    rng = np.random.default_rng(0)
    o = rng.normal(18.0, 1.0, 500)
    f = o + rng.normal(0.1, 0.2, 500)
    c = np.full(500, 18.0)
    w = rng.uniform(1.0, 2.0, 500)
    m = _metrics(f, o, c, w)

    d = f - o
    assert np.isclose(m["bias"], np.average(d, weights=w))
    assert np.isclose(m["rmse"], np.sqrt(np.average(d**2, weights=w)))
    assert np.isclose(m["mae"], np.average(np.abs(d), weights=w))
    fa, oa = f - np.average(f, weights=w), o - np.average(o, weights=w)
    assert np.isclose(
        m["spatial_correlation"],
        np.sum(w * fa * oa) / np.sqrt(np.sum(w * fa**2) * np.sum(w * oa**2)),
    )
    assert np.isclose(
        m["nash_sutcliffe"], 1 - np.sum(w * d**2) / np.sum(w * (o - np.average(o, weights=w)) ** 2)
    )
    assert m["count"] == 500


def test_ingredients_empty() -> None:
    r"""Determines if a group without valid point gives zero ingredients, and NaN metrics."""

    f = np.full(10, np.nan)
    sums = ingredients(f, np.ones(10), np.ones(10), np.ones(10), np.ones((10, 2)))

    assert sums.shape == (2, len(STATISTICS))
    assert np.all(sums == 0)
    assert np.isnan(_metrics(f, np.ones(10), np.ones(10), np.ones(10))["rmse"])


def test_ingredients_groups() -> None:
    r"""Determines if the ingredients of two disjoint groups add up to those of their union."""

    rng = np.random.default_rng(1)
    f, o, c, w = rng.normal(size=(4, 100))
    w = np.abs(w)
    first = rng.uniform(size=100) < 0.5
    groups = np.stack([np.ones(100), first, ~first], axis=1)
    sums = ingredients(f, o, c, w, groups)

    assert np.allclose(sums[0], sums[1] + sums[2])


def test_remove_pass_mean_typical() -> None:
    r"""Determines if the weighted mean of each pass is removed, ignoring the invalid units."""

    x = np.array([1.0, 3.0, 10.0, 20.0, 100.0])
    passes = np.array([0, 0, 1, 1, 1])
    valid = np.array([True, True, True, True, False])
    y = remove_pass_mean(x, np.ones(5), passes, valid)

    assert np.allclose(y[:4], [-1.0, 1.0, -5.0, 5.0])


def test_transform_log10() -> None:
    r"""Determines if only the log-normal variables are transformed, with the epsilon of Neptune."""

    x = np.array([0.0, 1.0])

    assert np.allclose(transform("CHL", x, "log10"), np.log10(x + 1e-5))
    assert np.all(np.isnan(transform("votemper", x, "log10")))
    assert np.array_equal(transform("votemper", x, "raw"), x)


def test_hypoxia_metrics_perfect() -> None:
    r"""Determines if a perfect detection scores one, and a ROC curve reaching its corner."""

    ds = xr.Dataset({
        "contingency" : (("cell",), [2.0, 0.0, 0.0, 8.0]),
        "extent"      : (("source", "measure"), [[2.0, 5.0], [2.0, 5.0]]),
        "histogram"   : (("truth", "bin"), [[2.0, 0.0, 0.0], [0.0, 3.0, 5.0]]),
    }, coords={
        "cell"    : ["hit", "false_alarm", "miss", "correct_negative"],
        "source"  : ["system", "reanalysis"],
        "measure" : ["area", "volume"],
        "truth"   : ["hypoxic", "not_hypoxic"],
    })  # fmt: skip
    scores = hypoxia_metrics(ds)

    for name in ("accuracy", "recall", "precision", "f1", "csi", "ets"):
        assert np.isclose(scores[name], 1.0)
    assert np.allclose(scores["extent_error"], 0.0)
    assert np.allclose(scores["roc_hit_rate"], 1.0)
    assert np.isclose(scores["roc_false_rate"][0], 0.0)


def test_kopelevich_regions_typical() -> None:
    r"""Determines if the open sea is split into its western, central and eastern regions."""

    lon, lat = np.array([31.0, 36.0, 40.5]), np.array([43.0, 43.0, 42.0])
    region = kopelevich_regions(lon, lat, np.full(3, 2000.0))

    assert list(region) == [1, 2, 3]


def test_pressure_to_depth_typical() -> None:
    r"""Determines if the depth is slightly smaller than the pressure in decibars."""

    depth = pressure_to_depth(np.array([0.0, 1000.0]), np.array([43.0, 43.0]))

    assert depth[0] == 0.0
    assert 985.0 < depth[1] < 995.0
