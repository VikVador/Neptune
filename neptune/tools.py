r"""A collection of tools designed for training module."""

import yaml

from itertools import product
from pathlib import Path
from typing import Any


def load_configuration(path: Path) -> list[dict[str, Any]]:
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
