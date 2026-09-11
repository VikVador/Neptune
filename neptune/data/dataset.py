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
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    VARIABLES_CLIPPING,
    Z,
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
        date_start     : Start date of the date range (format: 'YYYY-MM-DD').
        date_end       : End date of the date range (format: 'YYYY-MM-DD').
        standardized   : If True, standardize each channel using statistics.
        fill_with_nans : If True, land pixels are set to NaN instead of 0.
        split          : If True, split the output into surface and ocean variables.
    """

    def __init__(
        self,
        date_start: str,
        date_end: str,
        standardized: bool = True,
        fill_with_nans: bool = False,
        split: bool = True,
    ) -> None:
        super().__init__()

        assert_date_format(date_start)
        assert_date_format(date_end)

        self.standardized = standardized
        self.fill_with_nans = fill_with_nans
        self.split_output = split
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

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str] | tuple[Tensor, str]:
        r"""Return a preprocessed sample and its date, split by default into surface/ocean variables.

        Arguments:
            idx: Index into the dates list.

        Returns:
            x_s : Surface variables (C_s, Y, X) (Only if split is True).
            x_o : Ocean 3D variables (C_o, Z, Y, X) (Only if split is True).
            x   : Stacked variables (C, Y, X) (Only if split is False).
            d   : Date string 'YYYY-MM-DD'.
        """
        date = self.dates[idx]
        x = self.preprocess(date)
        if self.split_output:
            x_s, x_o = self.split(x)
            return x_s, x_o, date
        return x, date

    def split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Split a stacked sample into surface and ocean 3D variables.

        Arguments:
            x : Input tensor (..., C, Y, X).

        Returns:
            x_s : Surface variables (..., C_s, Y, X).
            x_o : Ocean 3D variables (..., C_o, Z, Y, X).
        """
        n_surface = len(DATASET_VARIABLES_SURFACE)
        n_ocean = len(DATASET_VARIABLES_OCEAN)
        x_s = x[..., :n_surface, :, :]
        x_o = x[..., n_surface:, :, :].reshape(*x.shape[:-3], n_ocean, Z, *x.shape[-2:])
        return x_s, x_o

    def unsplit(self, x_s: Tensor, x_o: Tensor) -> Tensor:
        r"""Merge surface and ocean 3D variables back into a stacked sample.

        Arguments:
            x_s : Surface variables (..., C_s, Y, X).
            x_o : Ocean 3D variables (..., C_o, Z, Y, X).

        Returns:
            x : Output Tensor (..., C, Y, X).
        """
        n_ocean = len(DATASET_VARIABLES_OCEAN)
        x_o_flat = x_o.reshape(*x_o.shape[:-4], n_ocean * Z, *x_o.shape[-2:])
        return torch.cat([x_s, x_o_flat], dim=-3)

    def standardize(self, data: Tensor) -> Tensor:
        r"""Standardize a sample channel-wise using precomputed statistics.

        Arguments:
            data: Tensor of shape (C, Y, X) or (B, C, Y, X).

        Returns:
            data: Standardized tensor of the same shape.
        """
        mean = self.mean_tensor.to(data.device, dtype=data.dtype)
        std = self.std_tensor.to(data.device, dtype=data.dtype)
        return (data - mean) / std

    def unstandardize(self, data: Tensor) -> Tensor:
        r"""Reverse the channel-wise standardization.

        Arguments:
            data: Standardized tensor of shape (C, Y, X) or (B, C, Y, X).

        Returns:
            data: Tensor in original physical units, same shape.
        """
        mean = self.mean_tensor.to(data.device, dtype=data.dtype)
        std = self.std_tensor.to(data.device, dtype=data.dtype)
        return data * std + mean

    def replace_outliers(self, data: Tensor, n_std: float = 10.0) -> Tensor:
        r"""Replace channel-wise statistical outliers with the channel mean.

        Arguments:
            data  : Tensor of shape (C, Y, X) or (B, C, Y, X).
            n_std : Number of standard deviations used as the outlier threshold.

        Returns:
            data: Tensor of the same shape, with outliers replaced by the channel mean.
        """
        mean = self.mean_tensor.to(data.device, dtype=data.dtype)
        std = self.std_tensor.to(data.device, dtype=data.dtype)
        outliers = (data > mean + n_std * std) | (data < mean - n_std * std)
        return torch.where(outliers, mean, data)

    def preprocess(self, date: str) -> Tensor:
        r"""Load and stack a single day into a (C, Y, X) tensor.

        Arguments:
            date: Date string 'YYYY-MM-DD'.

        Returns:
            sample: Tensor of shape (C, Y, X).
        """
        with xr.open_mfdataset(
            self.date_to_paths[date],
            combine="by_coords",
            compat="override",
            coords="minimal",
            data_vars="minimal",
        ) as raw:
            ds = raw.drop_vars(["nav_lat", "nav_lon"], errors="ignore")
            missing = [v for v in DATASET_VARIABLES if v not in ds]
            if missing:
                raise KeyError(f"ERROR - Missing required variables: {missing}")

            # Extracting partial domain
            ds = (
                ds[DATASET_VARIABLES]
                .isel(x=DATASET_REGION["x"], y=DATASET_REGION["y"], time_counter=0)
                .load()
            )

        # Extracting levels
        z_slice = DATASET_REGION["z"]
        channels = []
        for var in DATASET_VARIABLES:
            da = ds[var]
            if var in VARIABLES_CLIPPING:
                lo, hi = VARIABLES_CLIPPING[var]
                da = da.clip(min=lo, max=hi)
            if var in DATASET_VARIABLES_OCEAN:
                depth_dim = next((d for d in da.dims if d.startswith("depth")), None)
                if depth_dim:
                    da = da.isel({depth_dim: z_slice})
            data = torch.as_tensor(da.values.copy(), dtype=torch.float32)
            if data.ndim == 3:
                channels.extend(data.unbind(0))
            else:
                channels.append(data)

        # Preprocessing
        sample = torch.stack(channels, dim=0)
        sample = self.replace_outliers(sample)
        if self.standardized:
            sample = self.standardize(sample)
        if self.fill_with_nans:
            return sample.masked_fill(self.mask_tensor == 0, float("nan"))
        return sample.nan_to_num(0.0) * self.mask_tensor


def get_datasets(
    standardized: bool = True,
    fill_with_nans: bool = False,
) -> tuple[NeptuneDataset, NeptuneDataset, NeptuneDataset]:
    r"""Create train, validation and test datasets using the predefined date splits.

    Arguments:
        standardized   : If True, standardize each channel using precomputed statistics.
        fill_with_nans : If True, land pixels are set to NaN instead of 0.

    Returns:
        train : Training dataset
        val   : Validation dataset
        test  : Test dataset
    """
    kwargs: dict = {
        "standardized": standardized,
        "fill_with_nans": fill_with_nans,
    }
    return (
        NeptuneDataset(*DATASET_DATES_TRAINING, **kwargs),
        NeptuneDataset(*DATASET_DATES_VALIDATION, **kwargs),
        NeptuneDataset(*DATASET_DATES_TEST, **kwargs),
    )
