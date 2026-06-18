r"""Dataset."""

__all__ = [
    "NeptuneDataset",
    "NeptuneForecastLatentDataset",
    "NeptuneBlanketLatentDataset",
    "get_datasets",
    "get_forecast_latent_datasets",
    "get_blanket_latent_datasets",
]

import re
import torch
import xarray as xr

from collections import defaultdict
from torch import Tensor
from torch.utils.data import Dataset

from neptune.config import PATH_EXP_AE_LATENTS
from neptune.data import (
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    DATASET_REGION,
    DATASET_VARIABLES,
    DATASET_VARIABLES_OCEAN,
    VARIABLES_CLIPPING,
)
from neptune.data.tools import assert_date_format, generate_paths
from neptune.data.weights import get_weights_state_mask, get_weights_stats


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

            # Extracting partial region and variables
            ds = ds[DATASET_VARIABLES].isel(
                x=DATASET_REGION["x"],
                y=DATASET_REGION["y"],
                time_counter=0,
            )

            # Extracting depth for ocean variables
            z_slice = DATASET_REGION["z"]
            for var in DATASET_VARIABLES_OCEAN:
                if var in ds:
                    depth_dim = next((d for d in ds[var].dims if d.startswith("depth")), None)
                    if depth_dim:
                        ds[var] = ds[var].isel({depth_dim: z_slice})

            # Loading into memory
            ds.load()

        # Clip physical bounds
        for var, (lo, hi) in VARIABLES_CLIPPING.items():
            if var in ds:
                ds[var] = ds[var].clip(min=lo, max=hi)

        # Stack all variables into channels
        channels = []
        for var in DATASET_VARIABLES:
            if var not in ds:
                raise KeyError(f"ERROR - Missing required variable: {var}")
            data = torch.as_tensor(ds[var].values.copy(), dtype=torch.float32)
            if data.ndim == 3:
                channels.extend(data.unbind(0))
            else:
                channels.append(data)
        sample = torch.stack(channels, dim=0)

        # Replace statistical outliers channel-wise with the channel mean
        sample = self.replace_outliers(sample)

        # Standardize channel-wise
        if self.standardized:
            sample = self.standardize(sample)

        # Mask land
        if self.fill_with_nans:
            return sample.masked_fill(self.mask_tensor == 0, float("nan"))
        return sample.nan_to_num(0.0) * self.mask_tensor


def _load_latent_split(checkpoint_name: str, split: str) -> tuple[Tensor, list[str]]:
    r"""Helper tool to load latents and dates tensors for a given split.

    Arguments:
        checkpoint_name : Autoencoder name (directory under PATH_EXP_AE_LATENTS).
        split           : Dataset split ("train", "validation" or "test").

    Returns:
        latents : Tensor of shape (N, C, H, W).
        dates   : List of N date strings 'YYYY-MM-DD'.
    """

    # Security
    if split not in ("train", "validation", "test"):
        raise ValueError(
            f"ERROR - split must be one of ('train', 'validation', 'test'), got {split!r}"
        )

    # Load latents and dates tensors
    latent_dir = PATH_EXP_AE_LATENTS / checkpoint_name
    latent_file = next(latent_dir.glob(f"{split}_[0-9]*.pt"))
    dates_file = next(latent_dir.glob(f"{split}_dates_*.pt"))
    return (torch.load(latent_file, weights_only=True), torch.load(dates_file, weights_only=False))


class NeptuneForecastLatentDataset(Dataset):
    r"""Dataset returning (past, future) latent pairs for forecasting training.

    Arguments:
        checkpoint_name : Autoencoder name (directory under PATH_EXP_AE_LATENTS).
        split           : Dataset split ("train", "validation" or "test").
        input_size      : Number of past timesteps in the input window  (T_input).
        output_size     : Number of future timesteps in the output window (T_output).
    """

    def __init__(
        self,
        checkpoint_name: str,
        split: str,
        input_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.latents, self.dates = _load_latent_split(checkpoint_name, split)
        self.input_size = input_size
        self.output_size = output_size

        n_total = len(self.latents)
        self._n = n_total - input_size - output_size + 1

        if self._n <= 0:
            raise ValueError(
                f"Not enough timesteps ({n_total}) for "
                f"input_size={input_size} + output_size={output_size}."
            )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, list[str], list[str]]:
        r"""Return a (past, future) latent pair with their dates.

        Arguments:
            idx : Sample index.

        Returns:
            z_in      : Input latents,  shape (T_input,  C, H, W).
            z_out     : Output latents, shape (T_output, C, H, W).
            dates_in  : Date strings for each input  timestep, length T_input.
            dates_out : Date strings for each output timestep, length T_output.
        """
        t = idx + self.input_size - 1
        z_in = self.latents[t - self.input_size + 1 : t + 1]
        z_out = self.latents[t + 1 : t + self.output_size + 1]
        dates_in = self.dates[t - self.input_size + 1 : t + 1]
        dates_out = self.dates[t + 1 : t + self.output_size + 1]
        return z_in, z_out, dates_in, dates_out


class NeptuneBlanketLatentDataset(Dataset):
    r"""Dataset returning fixed-length consecutive latent sequences.

    Arguments:
        checkpoint_name : Autoencoder name (directory under PATH_EXP_AE_LATENTS).
        split           : Dataset split ("train", "validation" or "test").
        blanket_size    : Number of consecutive timesteps per sample.
    """

    def __init__(
        self,
        checkpoint_name: str,
        split: str,
        blanket_size: int,
    ) -> None:
        super().__init__()

        self.latents, self.dates = _load_latent_split(checkpoint_name, split)
        self.blanket_size = blanket_size

        self._n = len(self.latents) - blanket_size + 1

        if self._n <= 0:
            raise ValueError(
                f"Not enough timesteps ({len(self.latents)}) for blanket_size={blanket_size}."
            )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Tensor, list[str]]:
        r"""Return a consecutive latent sequence with its dates.

        Arguments:
            idx : Sample index.

        Returns:
            z     : Latent sequence, shape (blanket_size, C, H, W).
            dates : Date strings for each timestep, length blanket_size.
        """
        z = self.latents[idx : idx + self.blanket_size]
        dates = self.dates[idx : idx + self.blanket_size]
        return z, dates


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


def get_forecast_latent_datasets(
    checkpoint_name: str,
    input_size: int,
    output_size: int,
) -> tuple[
    NeptuneForecastLatentDataset, NeptuneForecastLatentDataset, NeptuneForecastLatentDataset
]:
    r"""Create train, validation and test forecast latent datasets.

    Arguments:
        checkpoint_name : Autoencoder name (directory under PATH_EXP_AE_LATENTS).
        input_size      : Number of past timesteps in the input window.
        output_size     : Number of future timesteps in the output window.

    Returns:
        train : Training dataset.
        val   : Validation dataset.
        test  : Test dataset.
    """
    kwargs: dict = {
        "checkpoint_name": checkpoint_name,
        "input_size": input_size,
        "output_size": output_size,
    }
    return (
        NeptuneForecastLatentDataset(split="train", **kwargs),
        NeptuneForecastLatentDataset(split="validation", **kwargs),
        NeptuneForecastLatentDataset(split="test", **kwargs),
    )


def get_blanket_latent_datasets(
    checkpoint_name: str,
    blanket_size: int,
) -> tuple[NeptuneBlanketLatentDataset, NeptuneBlanketLatentDataset, NeptuneBlanketLatentDataset]:
    r"""Create train, validation and test blanket latent datasets.

    Arguments:
        checkpoint_name : Autoencoder name (directory under PATH_EXP_AE_LATENTS).
        blanket_size    : Number of consecutive timesteps per sample.

    Returns:
        train : Training dataset.
        val   : Validation dataset.
        test  : Test dataset.
    """
    kwargs: dict = {
        "checkpoint_name": checkpoint_name,
        "blanket_size": blanket_size,
    }
    return (
        NeptuneBlanketLatentDataset(split="train", **kwargs),
        NeptuneBlanketLatentDataset(split="validation", **kwargs),
        NeptuneBlanketLatentDataset(split="test", **kwargs),
    )
