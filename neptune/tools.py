r"""A collection of tools for various tasks."""

__all__ = [
    "load_configuration",
    "generate_run_name_ae",
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


def generate_run_name_ae(
    joint_hash: str,
    in_channels: int,
    lat_channels: int,
    hid_channels: list[int],
    hid_blocks: list[int],
    stride: int,
    spatial: int = 2,
    previous_run_name: str | None = None,
) -> str:
    r"""Generate a descriptive WandB run name encoding the convolutional autoencoder architecture.

    Arguments:
        joint_hash        : Hash shared by all encoders launched together.
        in_channels       : Number of physical input channels.
        lat_channels      : Number of latent channels.
        hid_channels      : List of hidden channels per stage.
        hid_blocks        : List of blocks per stage
        stride            : Spatial stride per stage.
        spatial           : Number of spatial dimensions the stride is applied over.
        previous_run_name : WandB name of the resumed run, if any.

    Returns:
        name : Run name of the form CAE_{joint_hash}_{spatial}D_ic{}_lc{}_st{}_cf{}__XXX[_YYY].
    """
    ic = hid_channels[0]
    lc = lat_channels
    st = len(hid_blocks) - 1
    cf = round(stride ** (spatial * st) * in_channels / lat_channels)
    xxx = secrets.token_hex(2).upper()
    name = f"CAE_{joint_hash}_{spatial}D_ic{ic}_lc{lc}_st{st}_cf{cf}__{xxx}"

    if previous_run_name is not None:
        name = f"{name}_{extract_model_hash(previous_run_name)}"

    return name


def extract_model_hash(run_name: str) -> str:
    r"""Extract the unique model-identifying hash from a run name.

    Arguments:
        run_name : Run name of the form ..._{spatial}D_..._cf{}__XXX[_YYY].

    Returns:
        hash : The XXX hash identifying this specific model.
    """
    return run_name.split("__")[-1].split("_")[0]


def get_wandb_hyperparameters(configs: list[dict]) -> dict[str, Any]:
    r"""Flatten a configuration into a WandB-compatible hyperparameter dictionnary for analysis."""
    params = {}
    for cfg in configs:
        for k, v in cfg.items():
            if k == "learning_rate":
                params["Learning Rate"] = v
            elif k == "batch_size_per_step":
                params["Batch Size"] = v
            elif k == "hid_channels" and isinstance(v, list):
                params["Number of Stages"] = len(v)
                for i, h in enumerate(v):
                    params[f"Hidden Channels (Stage {i})"] = h
            elif k == "hid_blocks" and isinstance(v, list):
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
            elif k == "hid_channels" and isinstance(v, int):
                params["Hidden Channels"] = v
            elif k == "hid_blocks" and isinstance(v, int):
                params["Hidden Blocks"] = v
            elif k == "enc_features":
                params["Sine Encoding Features"] = v
            elif k == "patch_size":
                params["Patch Size"] = v
            elif k == "input_states":
                params["Input States"] = v
            elif k == "output_states":
                params["Output States"] = v

    return params
