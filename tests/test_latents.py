r"""Tests for experiments/forecast-latent-states/latents.py."""

import importlib.util
import numpy as np
import pytest
import xarray as xr

from pathlib import Path
from shaggy.tools import save

from neptune.data import X, Y, Z
from neptune.data.statistics import OnlineStats
from neptune.model.encoders import Encoder

# The script lives in an experiment folder, which is not a package
SPEC = importlib.util.spec_from_file_location(
    "latents",
    Path(__file__).parents[1] / "experiments" / "forecast-latent-states" / "latents.py",
)
latents = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latents)

CONFIG = {
    "lat_channels": 8,
    "hid_channels": [4, 8],
    "hid_blocks": [1, 1],
    "patch_size": 8,
    "stride": 2,
}


def test_statistics_typical(tmp_path: Path) -> None:
    r"""Determines if the saved statistics are the mean and std of each channel over time and space."""

    rng = np.random.default_rng(0)
    batches = [rng.normal(5.0, 2.0, size=(n, 3, 2, 4, 6)).astype(np.float32) for n in (4, 4, 1)]

    stats = [OnlineStats() for _ in range(3)]
    for z in batches:
        latents.update_statistics(stats, z)
    latents.save_statistics(stats, tmp_path / "statistics.zarr")

    z = np.concatenate(batches)
    saved = xr.open_zarr(tmp_path / "statistics.zarr").latents
    assert saved.dims == ("statistic", "channel") and saved.shape == (2, 3)
    assert np.allclose(saved.sel(statistic="mean"), z.mean(axis=(0, 2, 3, 4)), rtol=1e-5)
    assert np.allclose(saved.sel(statistic="std"), z.std(axis=(0, 2, 3, 4)), rtol=1e-5)


def test_statistics_constant_channel(tmp_path: Path) -> None:
    r"""Determines if a constant channel of a single day keeps its value as mean and a positive std."""

    z = np.ones((1, 2, 4, 6), dtype=np.float32)
    z[:, 1] = np.arange(24).reshape(4, 6)

    stats = [OnlineStats() for _ in range(2)]
    latents.update_statistics(stats, z)
    latents.save_statistics(stats, tmp_path / "statistics.zarr")

    saved = xr.open_zarr(tmp_path / "statistics.zarr").latents.values
    assert saved[0, 0] == 1.0 and 0.0 < saved[1, 0] <= 1e-8
    assert np.isclose(saved[0, 1], 11.5) and np.isclose(saved[1, 1], np.arange(24).std())

    with pytest.raises(ValueError):
        latents.update_statistics(stats, np.ones((1, 3, 4, 6), dtype=np.float32))


@pytest.mark.integration
@pytest.mark.parametrize("group", ["surface", "ocean"])
def test_encode_split_typical(
    group: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Determines if every day of a split is encoded in order, one day per chunk, with the statistics of its latents."""

    save(Encoder(group, **CONFIG), {"group": group, **CONFIG}, tmp_path / "models" / "ENC_test")
    monkeypatch.setattr(latents, "PATH_MODELS", tmp_path / "models")
    monkeypatch.setattr(latents, "PATH_LATENTS", tmp_path / "latents")
    monkeypatch.setattr(latents, "SPLITS", {"training": ("2000-01-01", "2000-01-05")})

    # Five days in batches of two, the last batch holding a single day
    latents.encode_split("ENC_test", "training", batch_size=2, num_workers=0, prefetch_factor=1)

    shape = (Y // 16, X // 16) if group == "surface" else (Z // 16, Y // 16, X // 16)
    ds = xr.open_zarr(tmp_path / "latents" / "ENC_test" / "training.zarr")
    dates = np.array([f"2000-01-0{d}" for d in range(1, 6)], dtype="datetime64[ns]")
    assert ds.latents.shape == (5, 8, *shape) and ds.latents.dtype == np.float32
    assert ds.latents.encoding["chunks"] == (1, 8, *shape)
    assert np.array_equal(ds.time.values, dates)

    z = ds.latents.values
    axes = (0, *range(2, z.ndim))
    stats = xr.open_zarr(tmp_path / "latents" / "ENC_test" / "statistics.zarr").latents
    assert np.isfinite(z).all() and stats.shape == (2, 8)
    assert np.allclose(stats.sel(statistic="mean"), z.mean(axis=axes), rtol=1e-4, atol=1e-6)
    assert np.allclose(stats.sel(statistic="std"), z.std(axis=axes), rtol=1e-4, atol=1e-6)
