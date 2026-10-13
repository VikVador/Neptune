r"""Tests for neptune.diagnostics."""

import pytest
import torch

from neptune.data import (
    DATASET_CONDITIONING,
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    X,
    Y,
    Z,
)
from neptune.diagnostics.metrics import _scores, _slice, forecast_ensemble
from neptune.model import FGN
from tests.test_fgn import CONFIG

C_S = len(DATASET_VARIABLES_SURFACE)
C_O = len(DATASET_VARIABLES_OCEAN)
C_C = len(DATASET_CONDITIONING)


def test_scores_typical() -> None:
    r"""Determines if the scores are the square roots of the mean squares, with a unit ratio for a calibrated ensemble."""

    mse, members = torch.tensor([4.0, 9.0]), 4
    scores = _scores(mse, mse * members / (members + 1), members)

    assert torch.allclose(scores["skill"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(scores["ratio"], torch.ones(2))


def test_slice_typical() -> None:
    r"""Determines if the ocean fields are sliced at a level (x-y), a longitude (y-z) or a latitude (z-x)."""

    x = torch.randn(2, 4, 5, 6)  # (M, Z, Y, X)

    assert torch.equal(_slice(x, "x-y", 1), x[:, 1])
    assert torch.equal(_slice(x, "y-z", 2), x[:, :, :, 2])
    assert torch.equal(_slice(x, "z-x", 3), x[:, :, 3])


@pytest.mark.integration
def test_forecast_ensemble_typical() -> None:
    r"""Determines if the ensemble yields every day of each chunk of members, differing between members."""

    torch.manual_seed(0)
    fgn = FGN(**{**CONFIG, "cond_channels_surface": 2 + C_C, "cond_channels_ocean": 2}).eval()
    dates = ["2018-01-01", "2018-01-02", "2018-01-03"]

    days = list(
        forecast_ensemble(
            fgn,
            torch.randn(2, C_S, Y, X),
            torch.randn(2, C_O, Z, Y, X),
            torch.randn(3, C_C, Y, X),
            dates,
            members=3,
            chunk=2,
            noise=[0.0, 0.1, 0.1],
        )
    )
    _, x_s, x_o = days[-1]

    assert [k for k, _, _ in days] == [0, 1, 2, 0, 1, 2]
    assert x_s.shape == (1, C_S, Y, X)
    assert x_o.shape == (1, C_O, Z, Y, X)
    assert not torch.equal(days[2][2][0], days[2][2][1])
