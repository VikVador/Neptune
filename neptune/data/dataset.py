r"""Dataset."""

__all__ = [
    "NeptuneDataset",
    "get_datasets",
]

import re
import torch
import xarray as xr

from collections import defaultdict
from torch import Tensor
from torch.utils.data import Dataset

from neptune.data import (
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    DATASET_REGION,
    DATASET_VARIABLES,
    VARIABLES_CLIPPING,
)
from neptune.data.tools import (
    assert_date_format,
    generate_paths,
)
from neptune.data.weights import (
    get_weights_state_mask,
    get_weights_stats,
)


class NeptuneDataset(Dataset):
    r"""Creates a Neptune dataset.

    Arguments:
        date_start     : Start date of the split (format: 'YYYY-MM-DD').
        date_end       : End date of the split (format: 'YYYY-MM-DD').
        standardized   : If True, standardize each channel using precomputed statistics.
        fill_with_nans : If True, land pixels are set to NaN instead of 0.
    """

    def __init__(
        self,
        date_start: str,
        date_end: str,
        standardized: bool = True,
        fill_with_nans: bool = False,
    ) -> None:
        super().__init__()

        assert_date_format(date_start)
        assert_date_format(date_end)

        self.standardized = standardized
        self.fill_with_nans = fill_with_nans
        self.mask_tensor = get_weights_state_mask()
        self.mean_tensor, self.std_tensor = get_weights_stats()

        date_to_paths: dict[str, list[str]] = defaultdict(list)
        for paths in generate_paths().values():
            for path in paths:
                match = re.search(r"BS_1d_(\d{8})_", path)
                if match:
                    d = match.group(1)
                    date_to_paths[f"{d[:4]}-{d[4:6]}-{d[6:]}"].append(path)

        self.dates = sorted(d for d in date_to_paths if date_start <= d <= date_end)
        self.date_to_paths = dict(date_to_paths)

    def __len__(self) -> int:
        r"""Return the number of valid dates in the split."""
        return len(self.dates)

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        r"""Return a preprocessed sample and its associated date.

        Arguments:
            idx: Index into the dates list.

        Returns:
            sample: Tensor of shape (C, Y, X).
            date:   Date string 'YYYY-MM-DD'.
        """
        date = self.dates[idx]
        return self.preprocess(date), date

    def standardize(self, data: Tensor) -> Tensor:
        r"""Standardize a sample channel-wise using precomputed statistics.

        Arguments:
            data: Tensor of shape (C, Y, X) or (B, C, Y, X).

        Returns:
            data: Standardized tensor of the same shape.
        """
        return (data - self.mean_tensor) / self.std_tensor

    def unstandardize(self, data: Tensor) -> Tensor:
        r"""Reverse the channel-wise standardization.

        Arguments:
            data: Standardized tensor of shape (C, Y, X) or (B, C, Y, X).

        Returns:
            data: Tensor in original physical units, same shape.
        """
        return data * self.std_tensor + self.mean_tensor

    def preprocess(self, date: str) -> Tensor:
        r"""Load and stack a single day into a (C, Y, X) tensor.

        Arguments:
            date: Date string 'YYYY-MM-DD'.

        Returns:
            sample: Tensor of shape (C, Y, X).
        """
        ds = xr.open_mfdataset(
            self.date_to_paths[date],
            combine="by_coords",
            compat="override",
            coords="minimal",
            data_vars="minimal",
        )

        # Drop redundant 2D spatial coordinates
        ds = ds.drop_vars(["nav_lat", "nav_lon"], errors="ignore")

        # Select variables present in files
        ds = ds[[v for v in DATASET_VARIABLES if v in ds]]

        # Apply spatial and depth region
        ds = ds.isel(**DATASET_REGION)

        # Select the single time step
        ds = ds.isel(time_counter=0)

        # Load into memory
        ds = ds.load()

        # Unify depth dimension (depthu, depthv → deptht)
        for var, old_dim in [("uo", "depthu"), ("vo", "depthv")]:
            if var in ds and old_dim in ds[var].dims:
                ds[var] = xr.DataArray(
                    ds[var].values, dims=["deptht", "y", "x"], attrs=ds[var].attrs
                )
        ds = ds.drop_vars(["depthu", "depthv"], errors="ignore")

        # Clip physical bounds (invalid values become NaN)
        for var, (lo, hi) in VARIABLES_CLIPPING.items():
            if var in ds:
                if lo is not None:
                    ds[var] = ds[var].where(ds[var] >= lo)
                if hi is not None:
                    ds[var] = ds[var].where(ds[var] <= hi)

        # Stack all variables into channels → (C, Y, X)
        channels = []
        for var in DATASET_VARIABLES:
            if var not in ds:
                continue
            data = torch.as_tensor(ds[var].values.copy(), dtype=torch.float32)
            if data.ndim == 3:
                channels.extend(data.unbind(0))
            else:
                channels.append(data)
        sample = torch.stack(channels, dim=0)

        # Standardize channel-wise
        if self.standardized:
            sample = self.standardize(sample)

        # Mask land
        if self.fill_with_nans:
            return sample.masked_fill(self.mask_tensor == 0, float("nan"))
        return sample.nan_to_num(0.0) * self.mask_tensor


def get_datasets(
    standardized: bool = True,
    fill_with_nans: bool = False,
) -> tuple[NeptuneDataset, NeptuneDataset, NeptuneDataset]:
    r"""Create train, validation and test datasets using the predefined date splits.

    Arguments:
        standardized   : Forwarded to each NeptuneDataset.
        fill_with_nans : Forwarded to each NeptuneDataset.

    Returns:
        train : Training dataset
        val   : Validation dataset
        test  : Test dataset
    """
    kwargs: dict = {"standardized": standardized, "fill_with_nans": fill_with_nans}
    return (
        NeptuneDataset(*DATASET_DATES_TRAINING, **kwargs),
        NeptuneDataset(*DATASET_DATES_VALIDATION, **kwargs),
        NeptuneDataset(*DATASET_DATES_TEST, **kwargs),
    )
