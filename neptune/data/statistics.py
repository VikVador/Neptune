r"""Online statistics utilities for dataset preprocessing."""

__all__ = [
    "OnlineStats",
    "ChunkedStats",
    "clean",
]

import numpy as np

from neptune.data import VARIABLES_CLIPPING


class OnlineStats:
    r"""Incremental mean and variance using weighted batch averaging."""

    def __init__(self) -> None:
        self.mu = None
        self.mu_sq = None
        self.count = 0

    def update(self, data: np.ndarray) -> None:
        r"""Update statistics with a new batch of values (NaNs are ignored).

        Arguments:
            data : Flat or multi-dimensional array of raw values.
        """

        # Double precision, since the variance is the difference of two close moments
        valid = data[~np.isnan(data)].astype(np.float64)
        if len(valid) == 0:
            return

        self._combine(float(valid.mean()), float((valid**2).mean()), len(valid))

    def merge(self, other: "OnlineStats") -> None:
        r"""Merge the statistics of another instance, as if its values had been seen by this one.

        Arguments:
            other : Statistics computed on another batch of values.
        """

        if other.count > 0:
            self._combine(other.mu, other.mu_sq, other.count)

    def _combine(self, b_mu: float, b_mu_sq: float, b_count: int) -> None:
        r"""Combine the current moments with the moments of a batch, weighted by their counts.

        Arguments:
            b_mu    : Mean of the batch.
            b_mu_sq : Mean of the squared values of the batch.
            b_count : Number of values in the batch.
        """

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
        r"""Mean of every value seen so far, or 0 when no value has been seen."""
        return self.mu if self.mu is not None else 0.0

    @property
    def std(self) -> float:
        r"""Standard deviation of every value seen so far, or 1 when no value has been seen."""

        if self.mu is None:
            return 1.0

        return max(float(np.sqrt(max(self.mu_sq - self.mu**2, 0.0))), 1e-8)


class ChunkedStats:
    r"""Average, over chunks of values (e.g. days or months), of the mean and std of each chunk.

    With daily chunks, the standard deviation reflects the variability within a day rather than
    the one across the whole period, which is dominated by the seasonal cycle for some variables.
    """

    def __init__(self) -> None:
        self.means = OnlineStats()
        self.stds = OnlineStats()

    def update(self, data: np.ndarray) -> None:
        r"""Update statistics with a new chunk of values (NaNs are ignored).

        Arguments:
            data : Flat or multi-dimensional array of the raw values of one chunk.
        """

        chunk = OnlineStats()
        chunk.update(data)

        if chunk.count > 0:
            self.means.update(np.array([chunk.mean]))
            self.stds.update(np.array([chunk.std]))

    def merge(self, other: "ChunkedStats") -> None:
        r"""Merge the statistics of another instance, as if its chunks had been seen by this one.

        Arguments:
            other : Statistics computed on other chunks of values.
        """

        self.means.merge(other.means)
        self.stds.merge(other.stds)

    @property
    def mean(self) -> float:
        r"""Average mean of every chunk seen so far, or 0 when no chunk has been seen."""
        return self.means.mean

    @property
    def std(self) -> float:
        r"""Average standard deviation of every chunk seen so far, or 1 when none has been seen."""
        return self.stds.mean if self.stds.count > 0 else 1.0


def clean(
    data: np.ndarray,
    var: str,
) -> np.ndarray:
    r"""Apply physical clipping and quantile filtering to raw data.

    Arguments:
        data : Raw array of values, possibly containing NaNs.
        var  : Variable name, used to look up its physical bounds.

    Returns:
        data : Cleaned array, clipped to the physical bounds, with extreme quantiles set to NaN.
    """

    # Clipping as in NeptuneDataset (e.g. negative oxygen of the anoxic zone is 0)
    lo, hi = VARIABLES_CLIPPING.get(var, (None, None))
    if lo is not None or hi is not None:
        data = np.clip(data, lo, hi)

    valid = data[~np.isnan(data)]
    if len(valid) > 0:
        q_lo, q_hi = np.quantile(valid, [0.02, 0.98])
        data = np.where((data < q_lo) | (data > q_hi), np.nan, data)

    return data
