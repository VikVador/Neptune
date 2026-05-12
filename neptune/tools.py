r"""A collection of tools designed for training module."""

import secrets
import yaml

from itertools import product
from pathlib import Path
from typing import Any


def load_configuration(path: str | Path) -> list[dict[str, Any]]:
    r"""Load all combinations of parameters from a YAML configuration file."""

    def _generate_combinations(d: dict[str, Any]) -> list[dict[str, Any]]:
        r"""Recursively generate parameter combinations."""
        if isinstance(d, dict):
            combinations = {k: _generate_combinations(v) for k, v in d.items()}
            keys, values = zip(*combinations.items(), strict=False)
            return [dict(zip(keys, combo, strict=False)) for combo in product(*values)]
        return d if isinstance(d, list) else [d]

    # Open and read the YAML configuration file
    with open(path) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(
            f"ERROR - Expected a YAML mapping at the top level, got {type(config).__name__}."
        )

    # Generate combinations
    return _generate_combinations(config)


def generate_run_name_ae(
    hid_channels: list[int],
    lat_channels: int,
    hid_blocks: list[int],
    stride: int,
    in_channels: int,
    previous_run_name: str | None = None,
) -> str:
    r"""Generate a descriptive WandB run name encoding the autoencoder architecture.

    Arguments:
        hid_channels      : Hidden channels per stage; IC = hid_channels[0].
        lat_channels      : Number of latent channels.
        hid_blocks        : Number of blocks per stage; ST = len(hid_blocks).
        stride            : Spatial stride per stage.
        in_channels       : Number of physical input channels (surface + ocean × depth).
        previous_run_name : WandB name of the resumed run, if any.

    Returns:
        name : Run name of the form CAE_IC{}_LC{}_ST{}_CF{}_XXX[_YYY].
    """
    ic = hid_channels[0]
    lc = lat_channels
    st = len(hid_blocks)
    cf = round(stride**st * in_channels / lat_channels)
    xxx = secrets.token_hex(2).upper()
    name = f"CAE__ic{ic}_lc{lc}_st{st}_cf{cf}__{xxx}"

    if previous_run_name is not None:
        yyy = previous_run_name.split("__")[-1].split("_")[0]
        name = f"{name}_{yyy}"

    return name
