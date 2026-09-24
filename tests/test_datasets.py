r"""Tests for neptune.data.dataset."""

import pytest
import torch

from neptune.data import (
    DATASET_DATES_TRAINING,
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    VARIABLES_CLIPPING,
    X,
    Y,
    Z,
)
from neptune.data.dataset import NeptuneDataset, get_datasets
from neptune.data.weights import get_weights_mask

C_S = len(DATASET_VARIABLES_SURFACE)
C_O = len(DATASET_VARIABLES_OCEAN)


@pytest.fixture()
def tiny_ds(monkeypatch: pytest.MonkeyPatch) -> NeptuneDataset:
    r"""Dataset over 10 fake consecutive days, with 2 surface and 3 ocean variables on 4 levels."""
    monkeypatch.setattr(
        "neptune.data.dataset.get_weights_mask",
        lambda: (torch.ones(1, 8, 8), torch.ones(1, 4, 8, 8)),
    )
    monkeypatch.setattr(
        "neptune.data.dataset.get_weights_stats",
        lambda: (
            torch.zeros(2, 1, 1),
            torch.ones(2, 1, 1),
            torch.zeros(3, 4, 1, 1),
            torch.ones(3, 4, 1, 1),
        ),
    )
    monkeypatch.setattr(
        "neptune.data.dataset.generate_paths",
        lambda: {"2000-01": [f"BS_1d_200001{d:02d}_grid_T.nc" for d in range(1, 11)]},
    )
    return NeptuneDataset("2000-01-01", "2000-01-10", input_states=2, output_states=3)


def test_init_invalid_date() -> None:
    r"""Determines if an invalid date format raises a ValueError."""
    with pytest.raises(ValueError):
        NeptuneDataset("01/01/1998", "01/01/2000")


def test_init_window_too_large(tiny_ds: NeptuneDataset) -> None:
    r"""Determines if a window longer than the date range raises a ValueError."""
    with pytest.raises(ValueError):
        NeptuneDataset("2000-01-01", "2000-01-10", input_states=6, output_states=5)


def test_len_windows(tiny_ds: NeptuneDataset) -> None:
    r"""Determines if the number of windows is the number of dates minus the window length plus one."""
    assert len(tiny_ds) == 10 - (2 + 3) + 1


def test_getitem_window(tiny_ds: NeptuneDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if a window returns N previous and K future consecutive states, with their dates."""

    def _fake_preprocess(date: str) -> tuple[torch.Tensor, torch.Tensor]:
        day = float(date[-2:])
        return torch.full((2, 8, 8), day), torch.full((3, 4, 8, 8), day)

    monkeypatch.setattr(tiny_ds, "preprocess", _fake_preprocess)
    x_inp_s, x_inp_o, x_out_s, x_out_o, dates = tiny_ds[4]

    assert x_inp_s.shape == (2, 2, 8, 8)
    assert x_inp_o.shape == (2, 3, 4, 8, 8)
    assert x_out_s.shape == (3, 2, 8, 8)
    assert x_out_o.shape == (3, 3, 4, 8, 8)
    assert dates == [f"2000-01-{d:02d}" for d in range(5, 10)]
    assert x_inp_o[:, 0, 0, 0, 0].tolist() == [5.0, 6.0]
    assert x_out_s[:, 0, 0, 0].tolist() == [7.0, 8.0, 9.0]


def test_standardize_roundtrip(tiny_ds: NeptuneDataset) -> None:
    r"""Determines if standardize followed by unstandardize is the identity, with a time dimension."""
    x_s, x_o = torch.randn(5, 2, 8, 8), torch.randn(5, 3, 4, 8, 8)
    y_s, y_o = tiny_ds.unstandardize(*tiny_ds.standardize(x_s, x_o))
    assert torch.allclose(y_s, x_s)
    assert torch.allclose(y_o, x_o)


def test_replace_outliers(tiny_ds: NeptuneDataset) -> None:
    r"""Determines if values beyond n_std are replaced by the mean and inliers remain unchanged."""
    x_s, x_o = torch.zeros(2, 8, 8), torch.zeros(3, 4, 8, 8)
    x_s[0, 0, 0] = 10.0
    x_s[1, 0, 0] = 2.0
    x_o[2, 3, 0, 0] = -10.0

    y_s, y_o = tiny_ds.replace_outliers(x_s, x_o, n_std=5.0)

    assert y_s[0, 0, 0] == 0.0
    assert y_s[1, 0, 0] == 2.0
    assert y_o[2, 3, 0, 0] == 0.0


@pytest.mark.integration
def test_getitem_shapes() -> None:
    r"""Determines if a real window has the expected shapes, dtype and dates."""
    ds = NeptuneDataset(*DATASET_DATES_TRAINING, input_states=2, output_states=1)
    x_inp_s, x_inp_o, x_out_s, x_out_o, dates = ds[0]
    assert x_inp_s.shape == (2, C_S, Y, X)
    assert x_inp_o.shape == (2, C_O, Z, Y, X)
    assert x_out_s.shape == (1, C_S, Y, X)
    assert x_out_o.shape == (1, C_O, Z, Y, X)
    assert x_inp_o.dtype == torch.float32
    assert dates == ["1998-01-01", "1998-01-02", "1998-01-03"]


@pytest.mark.integration
def test_preprocess_land_zero() -> None:
    r"""Determines if land pixels are zero and no NaN remains when fill_with_nans is False."""
    ds = NeptuneDataset(*DATASET_DATES_TRAINING)
    x_s, x_o = ds.preprocess(ds.dates[3])
    mask_surface, mask_ocean = get_weights_mask()
    assert not x_s.isnan().any() and not x_o.isnan().any()
    assert (x_s * (1 - mask_surface) == 0).all()
    assert (x_o * (1 - mask_ocean) == 0).all()


@pytest.mark.integration
def test_preprocess_fill_with_nans() -> None:
    r"""Determines if land pixels are NaN when fill_with_nans is True."""
    ds = NeptuneDataset(*DATASET_DATES_TRAINING, fill_with_nans=True)
    x_s, x_o = ds.preprocess(ds.dates[3])
    mask_surface, mask_ocean = get_weights_mask()
    assert x_s[:, mask_surface[0] == 0].isnan().all()
    assert x_o[:, mask_ocean[0] == 0].isnan().all()


@pytest.mark.integration
def test_preprocess_clipping() -> None:
    r"""Determines if physically clipped variables have no negative values."""
    ds = NeptuneDataset(*DATASET_DATES_TRAINING, standardized=False)
    x_s, x_o = ds.preprocess(ds.dates[3])
    for i, var in enumerate(DATASET_VARIABLES_SURFACE):
        if var in VARIABLES_CLIPPING:
            assert x_s[i].min() >= 0
    for i, var in enumerate(DATASET_VARIABLES_OCEAN):
        if var in VARIABLES_CLIPPING:
            assert x_o[i].min() >= 0


@pytest.mark.integration
def test_get_datasets_lengths() -> None:
    r"""Determines if train, validation and test datasets all contain at least one window."""
    for ds in get_datasets(input_states=2, output_states=4):
        assert len(ds) > 0
