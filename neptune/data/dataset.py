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
        standardized   : If True, standardize each channel using statistics.
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
            x : Tensor of shape (C, Y, X).
            d : Date string 'YYYY-MM-DD'.
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


def _get_latent_stats(checkpoint_name: str) -> tuple[Tensor, Tensor]:
    r"""Compute per-channel mean and std from the training latents.

    Arguments:
        checkpoint_name : Autoencoder name (directory under PATH_EXP_AE_LATENTS).

    Returns:
        mean : Per-channel mean (C, 1, 1).
        std  : Per-channel std (C, 1, 1).
    """
    latents, _ = _load_latent_split(checkpoint_name, "train")
    mean = latents.mean(dim=(0, 2, 3), keepdim=True).squeeze(0)
    std = latents.std(dim=(0, 2, 3), keepdim=True).squeeze(0)
    return mean, std


class NeptuneForecastLatentDataset(Dataset):
    r"""Dataset returning (past, future) latent pairs for forecasting training.

    Arguments:
        checkpoint_name : Autoencoder name (directory under PATH_EXP_AE_LATENTS).
        split           : Dataset split ("train", "validation" or "test").
        input_size      : Number of past timesteps in the input window  (T_in).
        output_size     : Number of future timesteps in the output window (T_out).
        standardized    : If True, standardize latents channel-wise using statistics.
    """

    def __init__(
        self,
        checkpoint_name: str,
        split: str,
        input_size: int,
        output_size: int,
        standardized: bool = True,
    ) -> None:
        super().__init__()

        self.latents, self.dates = _load_latent_split(checkpoint_name, split)
        self.input_size = input_size
        self.output_size = output_size

        self.mean, self.std = _get_latent_stats(checkpoint_name)
        if standardized:
            self.latents = (self.latents - self.mean) / self.std

        n_total = len(self.latents)
        self._n = n_total - input_size - output_size + 1

        if self._n <= 0:
            raise ValueError(
                f"Not enough timesteps ({n_total}) for "
                f"input_size={input_size} + output_size={output_size}."
            )

    def standardize(self, z: Tensor) -> Tensor:
        r"""Standardize latents channel-wise using statistics.

        Arguments:
            z : Latent tensor of shape (*, C, H, W).

        Returns:
            z : Standardized tensor of the same shape.
        """
        mean = self.mean.to(z.device, dtype=z.dtype)
        std = self.std.to(z.device, dtype=z.dtype)
        return (z - mean) / std

    def unstandardize(self, z: Tensor) -> Tensor:
        r"""Reverse the channel-wise standardization.

        Arguments:
            z : Standardized latent tensor of shape (*, C, H, W).

        Returns:
            z : Tensor in original latent units, same shape.
        """
        mean = self.mean.to(z.device, dtype=z.dtype)
        std = self.std.to(z.device, dtype=z.dtype)
        return z * std + mean

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, list[str], list[str]]:
        r"""Return a (past, future) latent pair with their dates.

        Arguments:
            idx : Sample index.

        Returns:
            z_in  : Input latents (T_in,  C, H, W).
            z_out : Output latents (T_out, C, H, W).
            d_in  : Date strings for each input  timestep (D_in).
            d_out : Date strings for each output timestep (D_out).
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
        standardized    : If True, standardize latents channel-wise using statistics.
    """

    def __init__(
        self,
        checkpoint_name: str,
        split: str,
        blanket_size: int,
        standardized: bool = True,
    ) -> None:
        super().__init__()

        self.latents, self.dates = _load_latent_split(checkpoint_name, split)
        self.blanket_size = blanket_size

        self.mean, self.std = _get_latent_stats(checkpoint_name)
        if standardized:
            self.latents = (self.latents - self.mean) / self.std

        self._n = len(self.latents) - blanket_size + 1

        if self._n <= 0:
            raise ValueError(
                f"Not enough timesteps ({len(self.latents)}) for blanket_size={blanket_size}."
            )

    def standardize(self, z: Tensor) -> Tensor:
        r"""Standardize latents channel-wise using statistics.

        Arguments:
            z : Latent tensor of shape (*, C, H, W).

        Returns:
            z : Standardized tensor of the same shape.
        """
        mean = self.mean.to(z.device, dtype=z.dtype)
        std = self.std.to(z.device, dtype=z.dtype)
        return (z - mean) / std

    def unstandardize(self, z: Tensor) -> Tensor:
        r"""Reverse the channel-wise standardization.

        Arguments:
            z : Standardized latent tensor of shape (*, C, H, W).

        Returns:
            z : Tensor in original latent units, same shape.
        """
        mean = self.mean.to(z.device, dtype=z.dtype)
        std = self.std.to(z.device, dtype=z.dtype)
        return z * std + mean

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Tensor, list[str]]:
        r"""Return a consecutive latent sequence with its dates.

        Arguments:
            idx : Sample index.

        Returns:
            z     : Latent sequence (T_blanket, C, H, W).
            dates : Date strings for each timesteps (D).
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
    standardized: bool = True,
) -> tuple[
    NeptuneForecastLatentDataset, NeptuneForecastLatentDataset, NeptuneForecastLatentDataset
]:
    r"""Create train, validation and test forecast latent datasets.

    Arguments:
        checkpoint_name : Autoencoder name (directory under PATH_EXP_AE_LATENTS).
        input_size      : Number of past timesteps in the input window.
        output_size     : Number of future timesteps in the output window.
        standardized    : If True, standardize latents channel-wise using statistics.

    Returns:
        train : Training dataset.
        val   : Validation dataset.
        test  : Test dataset.
    """
    kwargs: dict = {
        "checkpoint_name": checkpoint_name,
        "input_size": input_size,
        "output_size": output_size,
        "standardized": standardized,
    }
    return (
        NeptuneForecastLatentDataset(split="train", **kwargs),
        NeptuneForecastLatentDataset(split="validation", **kwargs),
        NeptuneForecastLatentDataset(split="test", **kwargs),
    )


def get_blanket_latent_datasets(
    checkpoint_name: str,
    blanket_size: int,
    standardized: bool = True,
) -> tuple[NeptuneBlanketLatentDataset, NeptuneBlanketLatentDataset, NeptuneBlanketLatentDataset]:
    r"""Create train, validation and test blanket latent datasets.

    Arguments:
        checkpoint_name : Autoencoder name (directory under PATH_EXP_AE_LATENTS).
        blanket_size    : Number of consecutive timesteps per sample.
        standardized    : If True, standardize latents channel-wise using statistics.

    Returns:
        train : Training dataset.
        val   : Validation dataset.
        test  : Test dataset.
    """
    kwargs: dict = {
        "checkpoint_name": checkpoint_name,
        "blanket_size": blanket_size,
        "standardized": standardized,
    }
    return (
        NeptuneBlanketLatentDataset(split="train", **kwargs),
        NeptuneBlanketLatentDataset(split="validation", **kwargs),
        NeptuneBlanketLatentDataset(split="test", **kwargs),
    )
