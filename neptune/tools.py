r"""A collection of tools designed for training module."""

__all__ = [
    "load_configuration",
    "generate_run_name_ae",
    "get_wandb_hyperparameters",
]

import secrets
import yaml

from itertools import product
from pathlib import Path
from typing import Any


def load_configuration(path: str | Path) -> list[dict[str, Any]]:
    r"""Load all combinations of parameters from a YAML configuration file.

    Arguments:
        path : Path to the YAML configuration file.

    Returns:
        configs : List of dicts, one per parameter combination (Cartesian product of list-valued keys).
    """

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
    in_channels: int,
    lat_channels: int,
    hid_channels: list[int],
    hid_blocks: list[int],
    stride: int,
    previous_run_name: str | None = None,
) -> str:
    r"""Generate a descriptive WandB run name encoding the autoencoder architecture.

    Arguments:
        in_channels       : Number of physical input channels.
        lat_channels      : Number of latent channels.
        hid_channels      : List of hidden channels per stage.
        hid_blocks        : List of blocks per stage
        stride            : Spatial stride per stage.
        previous_run_name : WandB name of the resumed run, if any.

    Returns:
        name : Run name of the form CAE_IC{}_LC{}_ST{}_CF{}_XXX[_YYY].
    """
    ic = hid_channels[0]
    lc = lat_channels
    st = len(hid_blocks) - 1
    cf = round(stride ** (2 * st) * in_channels / lat_channels)
    xxx = secrets.token_hex(2).upper()
    name = f"CAE__ic{ic}_lc{lc}_st{st}_cf{cf}__{xxx}"

    if previous_run_name is not None:
        yyy = previous_run_name.split("__")[-1].split("_")[0]
        name = f"{name}_{yyy}"

    return name


def get_wandb_hyperparameters(configs: list[dict]) -> dict[str, Any]:
    r"""Flatten a list of config dicts into a WandB-compatible hyperparameter dict for analysis.

    Arguments:
        configs : List of config dicts (e.g. config_training, config_arch).

    Returns:
        params : Flat dict mapping human-readable names to scalar values.
    """
    params = {}
    for cfg in configs:
        for k, v in cfg.items():
            if k == "learning_rate":
                params["Learning Rate"] = v
            elif k == "batch_size_per_step":
                params["Batch Size"] = v
            elif k == "hid_channels":
                params["Number of Stages"] = len(v)
                for i, h in enumerate(v):
                    params[f"Hidden Channels (Stage {i})"] = h
            elif k == "hid_blocks":
                for i, b in enumerate(v):
                    params[f"Hidden Blocks (Stage {i})"] = b
            elif k == "lat_channels":
                params["Latent Channels"] = v
            elif k == "kernel_size":
                params["Kernel Size"] = v
            elif k == "stride":
                params["Stride"] = v
            elif k == "ffn_factor":
                params["FFN Scaling Factor"] = v
            elif k == "dropout":
                params["Dropout"] = v

    return params
