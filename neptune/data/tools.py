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
