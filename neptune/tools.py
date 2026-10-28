r"""A collection of tools for various tasks."""

__all__ = [
    "load_configuration",
    "generate_run_name",
    "extract_model_hash",
    "get_wandb_hyperparameters",
    "freeze_modules",
    "schedule_training",
]

import cloudpickle
import secrets
import yaml

from collections.abc import Callable, Sequence
from dawgz import job, schedule
from itertools import product
from pathlib import Path
from types import ModuleType
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


def generate_run_name(name: str, previous_run_name: str | None = None) -> str:
    r"""Generate a unique WandB run name, from a descriptive name of the model and a random hash.

    Arguments:
        name              : Descriptive name of the model, e.g. 'ENC_surface_lc128_tk576'.
        previous_run_name : WandB name of the resumed run, if any.

    Returns:
        run_name : Run name of the form {name}__XXX[_YYY], XXX identifying this run and YYY the resumed one.
    """

    run_name = f"{name}__{secrets.token_hex(2).upper()}"

    if previous_run_name is not None:
        run_name = f"{run_name}_{extract_model_hash(previous_run_name)}"

    return run_name


def extract_model_hash(run_name: str) -> str:
    r"""Extract the unique model-identifying hash from a run name.

    Arguments:
        run_name : Run name of the form {name}__XXX[_YYY].

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


def freeze_modules(modules: Sequence[ModuleType]) -> None:
    r"""Pickle modules by value, such that the jobs run the code as it is at submission.

    Arguments:
        modules : Modules of the project used by the jobs.
    """

    for module in modules:
        cloudpickle.register_pickle_by_value(module)


def schedule_training(
    function: Callable,
    name: str,
    config_cluster: dict[str, Any],
    modules: Sequence[ModuleType],
    array: int | None = None,
) -> None:
    r"""Schedule a distributed training on Slurm with dawgz, torchrun starting one process per GPU.

    Arguments:
        function       : Training to run, taking the index of the configuration when array is given.
        name           : Name of the Slurm job, e.g. 'NEPT-TRAIN'.
        config_cluster : Slurm resources (nodes, gpus-per-node, cpus-per-node, ram-per-node, time, account, partition).
        modules        : Modules of the project frozen at submission.
        array          : Number of configurations trained in a job array, None for a single job.
    """

    nodes, gpus_per_node = config_cluster["nodes"], config_cluster["gpus-per-node"]

    # Several nodes meet at a rendezvous on the first node, with a port derived from the job ID
    if nodes > 1:
        interpreter = (
            f"torchrun --nnodes {nodes} --nproc-per-node {gpus_per_node} "
            f"--rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:$((20000 + SLURM_JOB_ID % 10000)) "
            f"--rdzv_id=$SLURM_JOB_ID"
        )
    else:
        interpreter = f"torchrun --nnodes 1 --nproc-per-node {gpus_per_node} --standalone"

    freeze_modules(modules)

    train = job(
        function,
        array=array,
        nodes=nodes,
        gpus=gpus_per_node,
        cpus=config_cluster["cpus-per-node"],
        ram=config_cluster["ram-per-node"],
        time=config_cluster["time"],
        account=config_cluster["account"],
        partition=config_cluster["partition"],
    )

    schedule(train, name=name, backend="slurm", interpreter=interpreter, export="ALL")
