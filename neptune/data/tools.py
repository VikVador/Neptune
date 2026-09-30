r"""A collection of tools designed for data module."""

__all__ = [
    "assert_date_format",
    "generate_paths",
]

import ast
import re

from collections.abc import Sequence

from neptune.config import (
    PATH_BTRC,
    PATH_GRID_T,
    PATH_GRID_U,
    PATH_GRID_V,
    PATH_GRID_W,
    PATH_PTRC,
)


def assert_date_format(date_string: str) -> None:
    r"""Assert that a date string follows the expected format.

    Arguments:
        date_string : Date to validate, expected format 'YYYY-MM-DD'.
    """

    pattern = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
    if not re.match(pattern, date_string):
        raise ValueError("ERROR - The format is incorrect, it should be YYYY-MM-DD.")


def generate_paths() -> dict[str, Sequence[str]]:
    r"""Generate the paths to the Black Sea simulation results, grouped by month.

    Returns:
        paths : Paths of every physics and biogeochemistry file, keyed by month 'YYYY-MM'.
    """
    grids = [
        ast.literal_eval(path.read_text())
        for path in (PATH_GRID_U, PATH_GRID_V, PATH_GRID_W, PATH_GRID_T, PATH_PTRC, PATH_BTRC)
    ]

    return {month: [path for grid in grids for path in grid[month]] for month in grids[3]}
