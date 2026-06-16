r"""A collection of tools designed for data module."""

__all__ = [
    "assert_date_format",
    "build_windows",
    "generate_paths",
]

import ast
import re

from collections.abc import Sequence
from datetime import date, timedelta

from neptune.config import (
    PATH_BTRC,
    PATH_GRID_T,
    PATH_GRID_U,
    PATH_GRID_V,
    PATH_GRID_W,
    PATH_PTRC,
)


def assert_date_format(date_string: str) -> None:
    r"""Asserts that date string follows correct format (YYYY-MM-DD)."""
    pattern = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
    if not re.match(pattern, date_string):
        raise ValueError("ERROR - The format is incorrect, it should be YYYY-MM-DD.")


def build_windows(date_start: str, date_end: str, timestep: int) -> list[tuple[str, str]]:
    r"""Build consecutive date windows of size timestep days.

    Arguments:
        date_start : First day of the range, format 'YYYY-MM-DD'.
        date_end   : Last day of the range, format 'YYYY-MM-DD'.
        timestep   : Number of days per window.

    Returns:
        windows : List of (start, end) tuples covering the full range.
    """

    # Sanity checks of date formats
    assert_date_format(date_start)
    assert_date_format(date_end)

    # Building windows
    start, end, windows = date.fromisoformat(date_start), date.fromisoformat(date_end), []
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=timestep - 1), end)
        windows.append((current.isoformat(), window_end.isoformat()))
        current = window_end + timedelta(days=1)
    return windows


def generate_paths() -> dict[str, Sequence[str]]:
    r"""Generate dictionary of paths to access Black Sea simulation monthly grouped results."""

    with open(PATH_GRID_U) as file:
        physics_data_U = ast.literal_eval(file.read())
    with open(PATH_GRID_V) as file:
        physics_data_V = ast.literal_eval(file.read())
    with open(PATH_GRID_W) as file:
        physics_data_W = ast.literal_eval(file.read())
    with open(PATH_GRID_T) as file:
        physics_data_T = ast.literal_eval(file.read())
    with open(PATH_BTRC) as file:
        biogeochemistry_data_btrc = ast.literal_eval(file.read())
    with open(PATH_PTRC) as file:
        biogeochemistry_data_ptrc = ast.literal_eval(file.read())

    paths = {}

    for date_month in physics_data_T.keys():
        paths[date_month] = (
            physics_data_U[date_month]
            + physics_data_V[date_month]
            + physics_data_W[date_month]
            + physics_data_T[date_month]
            + biogeochemistry_data_ptrc[date_month]
            + biogeochemistry_data_btrc[date_month]
        )

    return paths
