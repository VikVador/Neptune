r"""Tests for neptune.data.statistics."""

import numpy as np

from neptune.data.statistics import OnlineStats, clean


def test_online_stats_typical() -> None:
    r"""Determines if the online statistics match numpy over several batches, ignoring NaNs."""

    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, size=n) for n in (100, 250, 50)]
    batches[1][:10] = np.nan

    stats = OnlineStats()
    for batch in batches:
        stats.update(batch)

    values = np.concatenate(batches)
    assert np.isclose(stats.mean, np.nanmean(values))
    assert np.isclose(stats.std, np.nanstd(values))


def test_online_stats_merge() -> None:
    r"""Determines if merging partial statistics equals computing them in a single pass."""

    rng = np.random.default_rng(1)
    first, second = rng.normal(0.0, 1.0, size=300), rng.normal(5.0, 3.0, size=700)

    single, partial_1, partial_2 = OnlineStats(), OnlineStats(), OnlineStats()
    single.update(first)
    single.update(second)
    partial_1.update(first)
    partial_2.update(second)
    partial_1.merge(partial_2)

    assert np.isclose(partial_1.mean, single.mean)
    assert np.isclose(partial_1.std, single.std)
    assert partial_1.count == single.count == 1000


def test_online_stats_precision() -> None:
    r"""Determines if a tiny variance around a large mean survives float32 inputs (deep salinity)."""

    rng = np.random.default_rng(2)
    values = (22.0 + 0.005 * rng.standard_normal(100_000)).astype(np.float32)

    stats = OnlineStats()
    stats.update(values)

    assert np.isclose(stats.std, 0.005, rtol=0.05)


def test_clean_typical() -> None:
    r"""Determines if bounded variables are clipped, e.g. anoxic oxygen is a constant 0."""

    data = np.array([-5.0] * 50 + [np.nan] + [1.0] * 50)
    assert (clean(data, "DOX")[:50] == 0.0).all()
    assert np.nanmin(clean(data, "uo")) == -5.0

    stats = OnlineStats()
    stats.update(clean(np.full(100, -3.0), "DOX"))
    assert stats.count == 100
    assert stats.mean == 0.0
    assert stats.std == 1e-8
