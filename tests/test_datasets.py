r"""Tests for neptune.data.dataset."""

import pytest
import torch

from neptune.data import (
    DATASET_DATES_TRAINING,
    DATASET_REGION,
    DATASET_VARIABLES,
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    VARIABLES_CLIPPING,
)
from neptune.data.dataset import NeptuneDataset, get_datasets
from neptune.data.weights import get_weights_state_mask

Z = DATASET_REGION["depthu"].stop
Y = DATASET_REGION["y"].stop
X = DATASET_REGION["x"].stop
C = len(DATASET_VARIABLES_SURFACE) + len(DATASET_VARIABLES_OCEAN) * Z


@pytest.fixture()
def tiny_ds(monkeypatch: pytest.MonkeyPatch) -> NeptuneDataset:
    monkeypatch.setattr(
        "neptune.data.dataset.get_weights_state_mask",
        lambda: torch.ones(3, 8, 8),
    )
    monkeypatch.setattr(
        "neptune.data.dataset.get_weights_stats",
        lambda: (torch.zeros(3, 1, 1), torch.ones(3, 1, 1)),
    )
    monkeypatch.setattr("neptune.data.dataset.generate_paths", lambda: {})
    return NeptuneDataset("2000-01-01", "2000-12-31")


def test_init_invalid_date() -> None:
    r"""Determines if an invalid date format raises a ValueError."""
    with pytest.raises(ValueError):
        NeptuneDataset("01/01/1998", "01/01/2000")


def test_standardize_3d(tiny_ds: NeptuneDataset) -> None:
    r"""Determines if standardize returns the correct shape for a (C, Y, X) tensor."""
    x = torch.randn(3, 8, 8)
    assert tiny_ds.standardize(x).shape == (3, 8, 8)


def test_standardize_4d(tiny_ds: NeptuneDataset) -> None:
    r"""Determines if standardize returns the correct shape for a (B, C, Y, X) tensor."""
    x = torch.randn(4, 3, 8, 8)
    assert tiny_ds.standardize(x).shape == (4, 3, 8, 8)


def test_unstandardize_roundtrip(tiny_ds: NeptuneDataset) -> None:
    r"""Determines if standardize followed by unstandardize is the identity."""
    x = torch.randn(3, 8, 8)
    assert torch.allclose(tiny_ds.unstandardize(tiny_ds.standardize(x)), x, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
def test_init_valid() -> None:
    r"""Determines if the dataset loads the correct number of dates."""
    ds = NeptuneDataset(*DATASET_DATES_TRAINING)
    expected = sorted(
        d for d in ds.date_to_paths if DATASET_DATES_TRAINING[0] <= d <= DATASET_DATES_TRAINING[1]
    )
    assert len(ds) > 0
    assert len(ds) == len(expected)


@pytest.mark.integration
def test_preprocess_shape() -> None:
    r"""Determines if a preprocessed sample has the expected shape (C, Y, X)."""
    ds = NeptuneDataset(*DATASET_DATES_TRAINING)
    sample, _ = ds[3]
    assert sample.shape == (C, Y, X)


@pytest.mark.integration
def test_preprocess_dtype() -> None:
    r"""Determines if a preprocessed sample has dtype float32."""
    ds = NeptuneDataset(*DATASET_DATES_TRAINING)
    sample, _ = ds[3]
    assert sample.dtype == torch.float32


@pytest.mark.integration
def test_preprocess_no_nan() -> None:
    r"""Determines if a preprocessed sample (fill_with_nans=False) contains no NaN."""
    ds = NeptuneDataset(*DATASET_DATES_TRAINING)
    sample, _ = ds[3]
    assert not sample.isnan().any()


@pytest.mark.integration
def test_preprocess_land_zero() -> None:
    r"""Determines if land pixels are zero when fill_with_nans is False."""
    ds = NeptuneDataset(*DATASET_DATES_TRAINING)
    sample, _ = ds[3]
    mask = get_weights_state_mask()
    assert (sample[mask == 0] == 0).all()


@pytest.mark.integration
def test_preprocess_clipping() -> None:
    r"""Determines if physically clipped variables have no negative values."""
    ds_raw = NeptuneDataset(*DATASET_DATES_TRAINING, standardized=False)
    sample_raw, _ = ds_raw[3]
    clipped_channels = []
    ch = 0
    for var in DATASET_VARIABLES:
        n = 1 if var in DATASET_VARIABLES_SURFACE else Z
        if var in VARIABLES_CLIPPING:
            clipped_channels.extend(range(ch, ch + n))
        ch += n
    assert sample_raw[clipped_channels].min() >= 0


@pytest.mark.integration
def test_fill_with_nans() -> None:
    r"""Determines if land pixels are NaN when fill_with_nans is True."""
    ds_nan = NeptuneDataset(*DATASET_DATES_TRAINING, fill_with_nans=True)
    sample, _ = ds_nan[3]
    mask = get_weights_state_mask()
    assert sample[mask == 0].isnan().all()


@pytest.mark.integration
def test_get_datasets_lengths() -> None:
    r"""Determines if train, validation, and test datasets all contain at least one sample."""
    train, val, test = get_datasets()
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
