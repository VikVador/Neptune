r"""Online statistics utilities for dataset preprocessing."""

__all__ = [
    "OnlineStats",
    "clean",
]

import numpy as np

from neptune.data import VARIABLES_CLIPPING

_EPS = 1e-8


class OnlineStats:
    r"""Incremental mean and variance using weighted batch averaging."""

    def __init__(self) -> None:
        self.mu = None
        self.mu_sq = None
        self.count = 0

    def update(self, data: np.ndarray) -> None:
        r"""Update statistics with a new batch of values (NaNs are ignored).

        Arguments:
            data: flat or multi-dimensional array of raw values.
        """
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            return

        b_mu = float(valid.mean())
        b_mu_sq = float((valid**2).mean())
        b_count = len(valid)

        if self.count == 0:
            self.mu = b_mu
            self.mu_sq = b_mu_sq
        else:
            w1 = self.count / (self.count + b_count)
            w2 = b_count / (self.count + b_count)
            self.mu = w1 * self.mu + w2 * b_mu
            self.mu_sq = w1 * self.mu_sq + w2 * b_mu_sq

        self.count += b_count

    @property
    def mean(self) -> float:
        return self.mu if self.mu is not None else 0.0

    @property
    def std(self) -> float:
        if self.mu is None:
            return 1.0
        return max(float(np.sqrt(max(self.mu_sq - self.mu**2, 0.0))), _EPS)


def clean(
    data: np.ndarray,
    var: str,
) -> np.ndarray:
    r"""Apply physical clipping and quantile filtering to raw data.

    Arguments:
        data: raw numpy array (may contain NaNs).
        var:  variable name, used to look up physical bounds.

    Returns:
        data: cleaned array with out-of-range values set to NaN.
    """
    lo, hi = VARIABLES_CLIPPING.get(var, (None, None))
    if lo is not None:
        data = np.where(data < lo, np.nan, data)
    if hi is not None:
        data = np.where(data > hi, np.nan, data)

    valid = data[~np.isnan(data)]
    if len(valid) > 0:
        q_lo, q_hi = np.quantile(valid, [0.02, 0.98])
        data = np.where((data < q_lo) | (data > q_hi), np.nan, data)

    return data
