r"""Dataset."""

__all__ = [
    "NeptuneDataset",
    "get_datasets",
    "get_paths_by_date",
    "load_states",
]

import re
import torch
import xarray as xr

from collections import defaultdict
from torch import Tensor
from torch.utils.data import Dataset

from neptune.config import PATH_CONDITIONING
from neptune.data import (
    DATASET_CONDITIONING,
    DATASET_DATES_EXCLUDED,
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    DATASET_REGION,
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_SURFACE,
    VARIABLES_BOUNDS,
    VARIABLES_CLIPPING,
    VARIABLES_LOG10,
)
from neptune.data.tools import (
    assert_date_format,
    generate_paths,
)
from neptune.data.weights import (
    get_weights_mask,
    get_weights_stats,
    get_weights_stats_conditioning,
)


def get_paths_by_date() -> dict[str, list[str]]:
    r"""Group the paths of the NEMO files by date.

    Returns:
        paths : Paths of the files of each date 'YYYY-MM-DD'.
    """

    date_to_paths: dict[str, list[str]] = defaultdict(list)
    for paths in generate_paths().values():
        for path in paths:
            match = re.search(r"BS_1d_(\d{8})_", path)
            if match:
                d = match.group(1)
                date_to_paths[f"{d[:4]}-{d[4:6]}-{d[6:]}"].append(path)

    return dict(date_to_paths)


def load_states(
    paths: list[str],
    surface: bool = True,
    ocean: bool = True,
) -> tuple[Tensor | None, Tensor | None]:
    r"""Load the raw surface and ocean states of a single day, clipped and cropped to the domain.

    Arguments:
        paths   : Paths of the NEMO files of the day.
        surface : Whether to load the surface variables.
        ocean   : Whether to load the ocean variables.

    Returns:
        x_s : Surface state (C_s, Y, X), or None if not loaded.
        x_o : Ocean state (C_o, Z, Y, X), or None if not loaded.
    """

    variables_loaded = (DATASET_VARIABLES_SURFACE if surface else []) + (
        DATASET_VARIABLES_OCEAN if ocean else []
    )

    with xr.open_mfdataset(
        paths,
        combine="by_coords",
        compat="override",
        coords="minimal",
        data_vars="minimal",
    ) as raw:
        ds = raw.drop_vars(["nav_lat", "nav_lon"], errors="ignore")
        missing = [v for v in variables_loaded if v not in ds]
        if missing:
            raise KeyError(f"ERROR - Missing required variables: {missing}")

        # Extracting partial domain
        ds = (
            ds[variables_loaded]
            .isel(x=DATASET_REGION["x"], y=DATASET_REGION["y"], time_counter=0)
            .load()
        )

    # Clipping and extracting levels
    variables = {}
    for var in variables_loaded:
        da = ds[var]
        if var in VARIABLES_CLIPPING:
            lo, hi = VARIABLES_CLIPPING[var]
            da = da.clip(min=lo, max=hi)
        if var in DATASET_VARIABLES_OCEAN:
            depth_dim = next(d for d in da.dims if d.startswith("depth"))
            da = da.isel({depth_dim: DATASET_REGION["z"]})
        variables[var] = torch.as_tensor(da.values.copy(), dtype=torch.float32)

    x_s = torch.stack([variables[var] for var in DATASET_VARIABLES_SURFACE]) if surface else None
    x_o = torch.stack([variables[var] for var in DATASET_VARIABLES_OCEAN]) if ocean else None

    return x_s, x_o


class NeptuneDataset(Dataset):
    r"""Creates a Neptune dataset of consecutive (surface, ocean) states, with their conditioning.

    Arguments:
        date_start     : Start date of the date range (format: 'YYYY-MM-DD').
        date_end       : End date of the date range (format: 'YYYY-MM-DD').
        input_states   : Number of previous states N given as input.
        output_states  : Number of future states K to predict.
        standardized   : If True, standardize each variable using statistics.
        fill_with_nans : If True, land pixels are set to NaN instead of 0.
        group          : Group of variables to load, 'surface' or 'ocean', or None for both (the other one being empty).
    """

    def __init__(
        self,
        date_start: str,
        date_end: str,
        input_states: int = 1,
        output_states: int = 1,
        standardized: bool = True,
        fill_with_nans: bool = False,
        group: str | None = None,
    ) -> None:
        super().__init__()

        # Security
        assert_date_format(date_start)
        assert_date_format(date_end)
        valid = group in (None, "surface", "ocean")
        assert valid, f"ERROR - group must be 'surface', 'ocean' or None, got {group}."

        self.input_states = input_states
        self.output_states = output_states
        self.standardized = standardized
        self.fill_with_nans = fill_with_nans
        self.group = group
        self.mask_surface, self.mask_ocean = get_weights_mask()

        # The statistics are only needed to standardize, and are computed from unstandardized datasets
        if standardized:
            self.stats = get_weights_stats()
            self.stats_conditioning = get_weights_stats_conditioning()

        self.date_to_paths = get_paths_by_date()
        self.dates = sorted(d for d in self.date_to_paths if date_start <= d <= date_end)

        # A window of consecutive days starts at each date, unless it contains an excluded date
        days = input_states + output_states
        excluded = set(DATASET_DATES_EXCLUDED)
        self.windows = [
            i
            for i in range(len(self.dates) - days + 1)
            if not excluded.intersection(self.dates[i : i + days])
        ]

        # Security
        if len(self) <= 0:
            raise ValueError(
                f"ERROR - Not enough dates ({len(self.dates)}) for "
                f"input_states={input_states} + output_states={output_states}."
            )

    def __len__(self) -> int:
        r"""Return the number of windows of consecutive states in the date range."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[str]]:
        r"""Return a window of N previous and K future states, with the conditioning and the dates.

        Arguments:
            idx : Index of the window.

        Returns:
            x_inp_s : Previous surface states (N, C_s, Y, X).
            x_inp_o : Previous ocean states (N, C_o, Z, Y, X).
            x_out_s : Future surface states (K, C_s, Y, X).
            x_out_o : Future ocean states (K, C_o, Z, Y, X).
            c_inp   : Conditioning of the last input state of each forecasted step.
            dates   : Date strings 'YYYY-MM-DD' of the window (N + K).
        """

        start = self.windows[idx]
        dates = self.dates[start : start + self.input_states + self.output_states]
        states = [self.preprocess(date) for date in dates]

        x_s = torch.stack([x_s for x_s, _ in states])
        x_o = torch.stack([x_o for _, x_o in states])

        n, k = self.input_states, max(self.output_states, 1)
        c_inp = [self.preprocess_conditioning(date) for date in dates[n - 1 : n - 1 + k]]

        return x_s[:n], x_o[:n], x_s[n:], x_o[n:], torch.stack(c_inp), dates

    def _statistics(
        self,
        x_s: Tensor | None,
        x_o: Tensor | None,
        stats: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Return statistics on the device and dtype of the states.

        Arguments:
            x_s   : Surface states (..., C_s, Y, X), or None if not loaded.
            x_o   : Ocean states (..., C_o, Z, Y, X), or None if not loaded.
            stats : Statistics (mean_s, std_s, mean_o, std_o), from get_weights_stats.

        Returns:
            mean_s, std_s : Surface statistics (C_s, 1, 1).
            mean_o, std_o : Ocean statistics (C_o, Z, 1, 1).
        """
        x = x_s if x_s is not None else x_o

        return tuple(t.to(x.device, dtype=x.dtype) for t in stats)

    def standardize(
        self, x_s: Tensor | None, x_o: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None]:
        r"""Standardize surface and ocean states using precomputed statistics.

        Arguments:
            x_s : Surface states (..., C_s, Y, X), or None if not loaded.
            x_o : Ocean states (..., C_o, Z, Y, X), or None if not loaded.

        Returns:
            x_s : Standardized surface states, same shape.
            x_o : Standardized ocean states, same shape.
        """

        mean_s, std_s, mean_o, std_o = self._statistics(x_s, x_o, self.stats)

        if x_s is not None:
            x_s = (x_s - mean_s) / std_s
        if x_o is not None:
            x_o = (x_o - mean_o) / std_o

        return x_s, x_o

    def unstandardize(self, x_s: Tensor, x_o: Tensor) -> tuple[Tensor, Tensor]:
        r"""Reverse the standardization of surface and ocean states.

        Arguments:
            x_s : Standardized surface states (..., C_s, Y, X).
            x_o : Standardized ocean states (..., C_o, Z, Y, X).

        Returns:
            x_s : Surface states in physical units (log10 for VARIABLES_LOG10), same shape.
            x_o : Ocean states in physical units (log10 for VARIABLES_LOG10), same shape.
        """

        mean_s, std_s, mean_o, std_o = self._statistics(x_s, x_o, self.stats)

        return x_s * std_s + mean_s, x_o * std_o + mean_o

    def transform(
        self, x_s: Tensor | None, x_o: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None]:
        r"""Set the values outside their physical bounds to NaN, and take the log10 of the log-normal variables.

        Arguments:
            x_s : Surface states (..., C_s, Y, X), or None if not loaded.
            x_o : Ocean states (..., C_o, Z, Y, X), or None if not loaded.

        Returns:
            x_s : Transformed surface states, same shape.
            x_o : Transformed ocean states, same shape.
        """

        def _transform(x: Tensor, variables: list[str]) -> Tensor:
            r"""Transform the channels of one group of variables, along the dimension of the channels."""

            dim = -3 if variables == DATASET_VARIABLES_SURFACE else -4
            channels = []
            for var, xv in zip(variables, x.unbind(dim), strict=True):
                lo, hi = VARIABLES_BOUNDS[var]
                xv = xv.masked_fill((xv < lo) | (xv > hi), float("nan"))
                if var in VARIABLES_LOG10:
                    xv = torch.log10(xv + VARIABLES_LOG10[var])
                channels.append(xv)

            return torch.stack(channels, dim)

        if x_s is not None:
            x_s = _transform(x_s, DATASET_VARIABLES_SURFACE)
        if x_o is not None:
            x_o = _transform(x_o, DATASET_VARIABLES_OCEAN)

        return x_s, x_o

    def untransform(self, x_s: Tensor, x_o: Tensor) -> tuple[Tensor, Tensor]:
        r"""Bring the log-normal variables back from log10 to their physical units.

        Arguments:
            x_s : Unstandardized surface states (..., C_s, Y, X).
            x_o : Unstandardized ocean states (..., C_o, Z, Y, X).

        Returns:
            x_s : Surface states in physical units, same shape.
            x_o : Ocean states in physical units, same shape.
        """

        def _untransform(x: Tensor, variables: list[str]) -> Tensor:
            r"""Untransform the channels of one group of variables, along the dimension of the channels."""

            dim = -3 if variables == DATASET_VARIABLES_SURFACE else -4
            channels = [
                10**xv - VARIABLES_LOG10[var] if var in VARIABLES_LOG10 else xv
                for var, xv in zip(variables, x.unbind(dim), strict=True)
            ]

            return torch.stack(channels, dim)

        return (
            _untransform(x_s, DATASET_VARIABLES_SURFACE),
            _untransform(x_o, DATASET_VARIABLES_OCEAN),
        )

    def _fill_land(self, x: Tensor | None, mask: Tensor) -> Tensor:
        r"""Set land pixels to 0, or to NaN if fill_with_nans is True.

        Arguments:
            x    : Surface (..., C_s, Y, X) or ocean (..., C_o, Z, Y, X) states, or None if not loaded.
            mask : Matching land/sea mask, (1, Y, X) or (1, Z, Y, X).

        Returns:
            x : States with land pixels filled, same shape, or an empty tensor if not loaded.
        """

        if x is None:
            return torch.empty(0)

        if self.fill_with_nans:
            return x.masked_fill(mask == 0, float("nan"))

        return x.nan_to_num(0.0) * mask

    def preprocess(self, date: str) -> tuple[Tensor, Tensor]:
        r"""Load and preprocess the surface and ocean states of a single day.

        Arguments:
            date : Date string 'YYYY-MM-DD'.

        Returns:
            x_s : Surface state (C_s, Y, X), empty if not loaded.
            x_o : Ocean state (C_o, Z, Y, X), empty if not loaded.
        """

        surface, ocean = self.group in (None, "surface"), self.group in (None, "ocean")
        x_s, x_o = load_states(self.date_to_paths[date], surface, ocean)

        # Preprocessing, the values outside their bounds being NaN, then 0 or NaN as the land
        x_s, x_o = self.transform(x_s, x_o)
        if self.standardized:
            x_s, x_o = self.standardize(x_s, x_o)

        return self._fill_land(x_s, self.mask_surface), self._fill_land(x_o, self.mask_ocean)

    def preprocess_conditioning(self, date: str) -> Tensor:
        r"""Load and preprocess the atmospheric conditioning of a single day.

        Arguments:
            date : Date string 'YYYY-MM-DD'.

        Returns:
            c : Conditioning (C_c, Y, X).
        """

        with xr.open_zarr(PATH_CONDITIONING, chunks=None) as ds:
            c = ds[DATASET_CONDITIONING].sel(time=date)
            c = c.isel(x=DATASET_REGION["x"], y=DATASET_REGION["y"]).to_array().values

        c = torch.as_tensor(c, dtype=torch.float32)

        if self.standardized:
            mean_c, std_c = self.stats_conditioning
            c = (c - mean_c) / std_c

        return c


def get_datasets(
    input_states: int = 1,
    output_states: int = 1,
    standardized: bool = True,
    fill_with_nans: bool = False,
    group: str | None = None,
) -> tuple[NeptuneDataset, NeptuneDataset, NeptuneDataset]:
    r"""Create train, validation and test datasets using predefined date splits.

    Arguments:
        input_states   : Number of previous states N given as input.
        output_states  : Number of future states K to predict.
        standardized   : If True, standardize each variable using precomputed statistics.
        fill_with_nans : If True, land pixels are set to NaN instead of 0.
        group          : Group of variables to load, 'surface' or 'ocean', or None for both.

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
        "group": group,
    }

    return (
        NeptuneDataset(*DATASET_DATES_TRAINING, **kwargs),
        NeptuneDataset(*DATASET_DATES_VALIDATION, **kwargs),
        NeptuneDataset(*DATASET_DATES_TEST, **kwargs),
    )
