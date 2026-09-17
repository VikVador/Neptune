r"""Dataset."""

__all__ = [
    "NeptuneDataset",
    "NeptuneForecastDataset",
    "get_datasets",
    "get_forecast_datasets",
]

import re
import torch
import xarray as xr

from collections import defaultdict
from pathlib import Path
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


def _find_split(date_start: str, date_end: str) -> str:
    r"""Determine which official split a date range belongs to.

    Arguments:
        date_start : Start date of the range (format: 'YYYY-MM-DD').
        date_end   : End date of the range (format: 'YYYY-MM-DD').

    Returns:
        split : "train", "validation" or "test".
    """
    splits = {
        "train": DATASET_DATES_TRAINING,
        "validation": DATASET_DATES_VALIDATION,
        "test": DATASET_DATES_TEST,
    }
    for split, (split_start, split_end) in splits.items():
        if split_start <= date_start and date_end <= split_end:
            return split

    raise ValueError(
        f"ERROR - Date range {date_start}..{date_end} does not fit within a single split."
    )


def _load_latents(latents_dir: Path, split: str) -> dict:
    r"""Load the saved latents of one split.

    Arguments:
        latents_dir : Directory produced by encode.py (PATH_LATENTS / "latent_XXXX_YYYY").
        split       : Dataset split ("train", "validation" or "test").

    Returns:
        data : Dict with "z_surface" (N, C_s, Y', X'), "z_ocean" (N, C_o, Z', Y', X') and "dates" (N).
    """
    if split not in ("train", "validation", "test"):
        raise ValueError(
            f"ERROR - split must be one of ('train', 'validation', 'test'), got {split!r}"
        )

    return torch.load(latents_dir / f"{split}.pt", weights_only=False)


def _latent_stats(latents_dir: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Compute per-channel mean and std of the training latents, for standardization.

    Arguments:
        latents_dir : Directory produced by encode.py (PATH_LATENTS / "latent_XXXX_YYYY").

    Returns:
        mean_surface, std_surface : Per-channel statistics of the surface latents.
        mean_ocean, std_ocean     : Per-channel statistics of the ocean latents.
    """

    def _stats(z: Tensor) -> tuple[Tensor, Tensor]:
        dims = tuple(d for d in range(z.dim()) if d != 1)
        return z.mean(dim=dims, keepdim=True), z.std(dim=dims, keepdim=True)

    train = _load_latents(latents_dir, "train")
    mean_surface, std_surface = _stats(train["z_surface"])
    mean_ocean, std_ocean = _stats(train["z_ocean"])
    return mean_surface, std_surface, mean_ocean, std_ocean


class NeptuneForecastDataset(Dataset):
    r"""Dataset returning latent inputs and ambient outputs for forecast training.

    Arguments:
        latents_dir   : Directory produced by encode.py (PATH_LATENTS / "latent_XXXX_YYYY").
        date_start    : Start date of the date range (format: 'YYYY-MM-DD').
        date_end      : End date of the date range (format: 'YYYY-MM-DD').
        input_states  : Number of past latent states given as input (T_in).
        output_states : Number of future ambient states to predict (T_out).
    """

    def __init__(
        self,
        latents_dir: Path,
        date_start: str,
        date_end: str,
        input_states: int,
        output_states: int,
    ) -> None:
        super().__init__()

        assert_date_format(date_start)
        assert_date_format(date_end)

        split = _find_split(date_start, date_end)
        data = _load_latents(latents_dir, split)

        self.mean_surface, self.std_surface, self.mean_ocean, self.std_ocean = _latent_stats(
            latents_dir
        )

        keep = [i for i, d in enumerate(data["dates"]) if date_start <= d <= date_end]
        self.z_surface = self.standardize(data["z_surface"][keep], "surface")
        self.z_ocean = self.standardize(data["z_ocean"][keep], "ocean")
        self.dates = [data["dates"][i] for i in keep]

        self.input_states = input_states
        self.output_states = output_states
        self.ambient = NeptuneDataset(date_start, date_end, standardized=True, fill_with_nans=True)

        self._n = len(self.dates) - input_states - output_states + 1
        if self._n <= 0:
            raise ValueError(
                f"Not enough timesteps ({len(self.dates)}) for "
                f"input_states={input_states} + output_states={output_states}."
            )

    def standardize(self, z: Tensor, role: str) -> Tensor:
        r"""Standardize a latent tensor channel-wise using training statistics.

        Arguments:
            z    : Latent tensor of shape (*, C, ...).
            role : "surface" or "ocean".

        Returns:
            z : Standardized tensor of the same shape.
        """
        mean, std = (
            (self.mean_surface, self.std_surface)
            if role == "surface"
            else (self.mean_ocean, self.std_ocean)
        )
        return (z - mean.to(z.device, dtype=z.dtype)) / std.to(z.device, dtype=z.dtype)

    def unstandardize(self, z: Tensor, role: str) -> Tensor:
        r"""Reverse the channel-wise standardization of a latent tensor.

        Arguments:
            z    : Standardized latent tensor of shape (*, C, ...).
            role : "surface" or "ocean".

        Returns:
            z : Tensor in original latent units, same shape.
        """
        mean, std = (
            (self.mean_surface, self.std_surface)
            if role == "surface"
            else (self.mean_ocean, self.std_ocean)
        )
        return z * std.to(z.device, dtype=z.dtype) + mean.to(z.device, dtype=z.dtype)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, list[str], list[str]]:
        r"""Return a (latent input, ambient output) pair with their dates.

        Arguments:
            idx : Sample index.

        Returns:
            z_s_in    : Input surface latents (T_in, C_s, Y', X').
            z_o_in    : Input ocean latents   (T_in, C_o, Z', Y', X').
            x_s_out   : Output ambient surface variables (T_out, C_s, Y, X).
            x_o_out   : Output ambient ocean 3D variables (T_out, C_o, Z, Y, X).
            dates_in  : Date strings for each input timestep (T_in).
            dates_out : Date strings for each output timestep (T_out).
        """
        t = idx + self.input_states - 1
        dates_in = self.dates[t - self.input_states + 1 : t + 1]
        dates_out = self.dates[t + 1 : t + self.output_states + 1]

        z_s_in = self.z_surface[t - self.input_states + 1 : t + 1]
        z_o_in = self.z_ocean[t - self.input_states + 1 : t + 1]

        outputs = [self.ambient.split(self.ambient.preprocess(d)) for d in dates_out]
        x_s_out = torch.stack([x_s for x_s, _ in outputs])
        x_o_out = torch.stack([x_o for _, x_o in outputs])

        return z_s_in, z_o_in, x_s_out, x_o_out, dates_in, dates_out


def get_forecast_datasets(
    latents_dir: Path,
    input_states: int,
    output_states: int,
) -> tuple[NeptuneForecastDataset, NeptuneForecastDataset, NeptuneForecastDataset]:
    r"""Create train, validation and test forecast datasets using the predefined date splits.

    Arguments:
        latents_dir   : Directory produced by encode.py (PATH_LATENTS / "latent_XXXX_YYYY").
        input_states  : Number of past latent states given as input.
        output_states : Number of future ambient states to predict.

    Returns:
        train : Training dataset.
        val   : Validation dataset.
        test  : Test dataset.
    """
    kwargs: dict = {
        "latents_dir": latents_dir,
        "input_states": input_states,
        "output_states": output_states,
    }
    return (
        NeptuneForecastDataset(
            date_start=DATASET_DATES_TRAINING[0], date_end=DATASET_DATES_TRAINING[1], **kwargs
        ),
        NeptuneForecastDataset(
            date_start=DATASET_DATES_VALIDATION[0], date_end=DATASET_DATES_VALIDATION[1], **kwargs
        ),
        NeptuneForecastDataset(
            date_start=DATASET_DATES_TEST[0], date_end=DATASET_DATES_TEST[1], **kwargs
        ),
    )
