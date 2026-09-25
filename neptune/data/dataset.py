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
)
from neptune.data.tools import (
    assert_date_format,
    generate_paths,
)
from neptune.data.weights import (
    get_weights_mask,
    get_weights_stats,
)


class NeptuneDataset(Dataset):
    r"""Creates a Neptune dataset of consecutive (surface, ocean) states.

    Arguments:
        date_start     : Start date of the date range (format: 'YYYY-MM-DD').
        date_end       : End date of the date range (format: 'YYYY-MM-DD').
        input_states   : Number of previous states N given as input.
        output_states  : Number of future states K to predict.
        standardized   : If True, standardize each variable using statistics.
        fill_with_nans : If True, land pixels are set to NaN instead of 0.
    """

    def __init__(
        self,
        date_start: str,
        date_end: str,
        input_states: int = 1,
        output_states: int = 1,
        standardized: bool = True,
        fill_with_nans: bool = False,
    ) -> None:
        super().__init__()

        # Security
        assert_date_format(date_start)
        assert_date_format(date_end)

        self.input_states = input_states
        self.output_states = output_states
        self.standardized = standardized
        self.fill_with_nans = fill_with_nans
        self.mask_surface, self.mask_ocean = get_weights_mask()
        self.mean_surface, self.std_surface, self.mean_ocean, self.std_ocean = get_weights_stats()

        date_to_paths: dict[str, list[str]] = defaultdict(list)
        for paths in generate_paths().values():
            for path in paths:
                match = re.search(r"BS_1d_(\d{8})_", path)
                if match:
                    d = match.group(1)
                    date_to_paths[f"{d[:4]}-{d[4:6]}-{d[6:]}"].append(path)

        self.dates = sorted(d for d in date_to_paths if date_start <= d <= date_end)
        self.date_to_paths = dict(date_to_paths)

        # Security
        if len(self) <= 0:
            raise ValueError(
                f"ERROR - Not enough dates ({len(self.dates)}) for "
                f"input_states={input_states} + output_states={output_states}."
            )

    def __len__(self) -> int:
        r"""Return the number of windows of consecutive states in the date range."""
        return len(self.dates) - self.input_states - self.output_states + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, list[str]]:
        r"""Return a window of N previous and K future states, with their dates.

        Arguments:
            idx : Index of the window.

        Returns:
            x_inp_s : Previous surface states (N, C_s, Y, X).
            x_inp_o : Previous ocean states (N, C_o, Z, Y, X).
            x_out_s : Future surface states (K, C_s, Y, X).
            x_out_o : Future ocean states (K, C_o, Z, Y, X).
            dates   : Date strings 'YYYY-MM-DD' of the window (N + K).
        """

        dates = self.dates[idx : idx + self.input_states + self.output_states]
        states = [self.preprocess(date) for date in dates]

        x_s = torch.stack([x_s for x_s, _ in states])
        x_o = torch.stack([x_o for _, x_o in states])

        n = self.input_states

        return x_s[:n], x_o[:n], x_s[n:], x_o[n:], dates

    def _statistics(self, x_s: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Return the standardization statistics on the device and dtype of the states.

        Arguments:
            x_s : Surface states (..., C_s, Y, X).

        Returns:
            mean_s, std_s : Surface statistics (C_s, 1, 1).
            mean_o, std_o : Ocean statistics (C_o, Z, 1, 1).
        """

        return tuple(
            t.to(x_s.device, dtype=x_s.dtype)
            for t in (self.mean_surface, self.std_surface, self.mean_ocean, self.std_ocean)
        )

    def standardize(self, x_s: Tensor, x_o: Tensor) -> tuple[Tensor, Tensor]:
        r"""Standardize surface and ocean states using precomputed statistics.

        Arguments:
            x_s : Surface states (..., C_s, Y, X).
            x_o : Ocean states (..., C_o, Z, Y, X).

        Returns:
            x_s : Standardized surface states, same shape.
            x_o : Standardized ocean states, same shape.
        """

        mean_s, std_s, mean_o, std_o = self._statistics(x_s)

        return (x_s - mean_s) / std_s, (x_o - mean_o) / std_o

    def unstandardize(self, x_s: Tensor, x_o: Tensor) -> tuple[Tensor, Tensor]:
        r"""Reverse the standardization of surface and ocean states.

        Arguments:
            x_s : Standardized surface states (..., C_s, Y, X).
            x_o : Standardized ocean states (..., C_o, Z, Y, X).

        Returns:
            x_s : Surface states in physical units, same shape.
            x_o : Ocean states in physical units, same shape.
        """

        mean_s, std_s, mean_o, std_o = self._statistics(x_s)

        return x_s * std_s + mean_s, x_o * std_o + mean_o

    def replace_outliers(
        self,
        x_s: Tensor,
        x_o: Tensor,
        n_std: float = 10.0,
    ) -> tuple[Tensor, Tensor]:
        r"""Replace statistical outliers of surface and ocean states with the mean.

        Arguments:
            x_s   : Surface states (..., C_s, Y, X).
            x_o   : Ocean states (..., C_o, Z, Y, X).
            n_std : Number of standard deviations used as the outlier threshold.

        Returns:
            x_s : Surface states, with outliers replaced by the mean.
            x_o : Ocean states, with outliers replaced by the mean.
        """

        mean_s, std_s, mean_o, std_o = self._statistics(x_s)
        outliers_s = (x_s - mean_s).abs() > n_std * std_s
        outliers_o = (x_o - mean_o).abs() > n_std * std_o

        return torch.where(outliers_s, mean_s, x_s), torch.where(outliers_o, mean_o, x_o)

    def _fill_land(self, x: Tensor, mask: Tensor) -> Tensor:
        r"""Set land pixels to 0, or to NaN if fill_with_nans is True.

        Arguments:
            x    : Surface (..., C_s, Y, X) or ocean (..., C_o, Z, Y, X) states.
            mask : Matching land/sea mask, (1, Y, X) or (1, Z, Y, X).

        Returns:
            x : States with land pixels filled, same shape.
        """

        if self.fill_with_nans:
            return x.masked_fill(mask == 0, float("nan"))

        return x.nan_to_num(0.0) * mask

    def preprocess(self, date: str) -> tuple[Tensor, Tensor]:
        r"""Load and preprocess the surface and ocean states of a single day.

        Arguments:
            date : Date string 'YYYY-MM-DD'.

        Returns:
            x_s : Surface state (C_s, Y, X).
            x_o : Ocean state (C_o, Z, Y, X).
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

        # Clipping and extracting levels
        variables = {}
        for var in DATASET_VARIABLES:
            da = ds[var]
            if var in VARIABLES_CLIPPING:
                lo, hi = VARIABLES_CLIPPING[var]
                da = da.clip(min=lo, max=hi)
            if var in DATASET_VARIABLES_OCEAN:
                depth_dim = next(d for d in da.dims if d.startswith("depth"))
                da = da.isel({depth_dim: DATASET_REGION["z"]})
            variables[var] = torch.as_tensor(da.values.copy(), dtype=torch.float32)

        x_s = torch.stack([variables[var] for var in DATASET_VARIABLES_SURFACE])
        x_o = torch.stack([variables[var] for var in DATASET_VARIABLES_OCEAN])

        # Preprocessing
        x_s, x_o = self.replace_outliers(x_s, x_o)
        if self.standardized:
            x_s, x_o = self.standardize(x_s, x_o)

        return self._fill_land(x_s, self.mask_surface), self._fill_land(x_o, self.mask_ocean)


def get_datasets(
    input_states: int = 1,
    output_states: int = 1,
    standardized: bool = True,
    fill_with_nans: bool = False,
) -> tuple[NeptuneDataset, NeptuneDataset, NeptuneDataset]:
    r"""Create train, validation and test datasets using predefined date splits.

    Arguments:
        input_states   : Number of previous states N given as input.
        output_states  : Number of future states K to predict.
        standardized   : If True, standardize each variable using precomputed statistics.
        fill_with_nans : If True, land pixels are set to NaN instead of 0.

    Returns:
        train : Training dataset
        val   : Validation dataset
        test  : Test dataset
    """

    kwargs: dict = {
        "input_states": input_states,
        "output_states": output_states,
        "standardized": standardized,
        "fill_with_nans": fill_with_nans,
    }

    return (
        NeptuneDataset(*DATASET_DATES_TRAINING, **kwargs),
        NeptuneDataset(*DATASET_DATES_VALIDATION, **kwargs),
        NeptuneDataset(*DATASET_DATES_TEST, **kwargs),
    )
