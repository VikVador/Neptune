r"""A collection of tools for various tasks."""

__all__ = [
    "load_configuration",
    "generate_run_name_fgn",
    "extract_model_hash",
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


def generate_run_name_fgn(
    input_states: int,
    lat_channels: int,
    compression: float,
    tokens: int,
    hid_channels: int,
    hid_blocks: int,
    previous_run_name: str | None = None,
) -> str:
    r"""Generate a descriptive WandB run name encoding the FGN architecture.

    Arguments:
        input_states      : Number of previous states given as input.
        lat_channels      : Number of latent channels.
        compression       : Total compression factor of the states into latent tokens.
        tokens            : Number of latent tokens processed by the transformer.
        hid_channels      : Number of hidden channels of the transformer.
        hid_blocks        : Number of blocks of the transformer.
        previous_run_name : WandB name of the resumed run, if any.

    Returns:
        name : Run name of the form FGN_in{}_lc{}_cf{}_tk{}_hc{}_hb{}__XXX[_YYY].
    """

    xxx = secrets.token_hex(2).upper()
    name = (
        f"FGN_in{input_states}_lc{lat_channels}_cf{round(compression)}_tk{tokens}"
        f"_hc{hid_channels}_hb{hid_blocks}__{xxx}"
    )

    if previous_run_name is not None:
        name = f"{name}_{extract_model_hash(previous_run_name)}"

    return name


def extract_model_hash(run_name: str) -> str:
    r"""Extract the unique model-identifying hash from a run name.

    Arguments:
        run_name : Run name of the form FGN_..._hb{}__XXX[_YYY].

    Returns:
        hash : The XXX hash identifying this specific model.
    """
    return run_name.split("__")[-1].split("_")[0]


def get_wandb_hyperparameters(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    r"""Flatten a nested configuration into scalar hyperparameters, for WandB analysis.

    Arguments:
        config : Nested configuration, e.g. {"config_processor": {"hid_channels": 512}}.
        prefix : Prefix of the flattened names, used by the recursion.

    Returns:
        params : Hyperparameters, e.g. {"config_processor/hid_channels": 512}.
    """

    params = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            params |= get_wandb_hyperparameters(value, prefix=f"{name}/")
        elif isinstance(value, list):
            params |= {f"{name}/{i}": v for i, v in enumerate(value)}
        else:
            params[name] = value

    return params
