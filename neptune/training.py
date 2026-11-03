r"""Tools to stop, save and resume a training."""

__all__ = [
    "has_stagnated",
    "save_state",
    "load_state",
    "get_run_id",
    "submit_resume",
]

import numpy as np
import os
import re
import subprocess
import torch

from collections.abc import Sequence
from pathlib import Path
from typing import Any

# fmt: off
#
# ----- Variables of the current job that must not reach the job resuming it
#
VARIABLES_JOB = ("SLURM_", "SBATCH_", "TORCHELASTIC_")
VARIABLES_TORCHRUN = (
    "RANK", "LOCAL_RANK", "GROUP_RANK", "ROLE_RANK", "ROLE_NAME", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "GROUP_WORLD_SIZE", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES",
)
# fmt: on


def has_stagnated(history: Sequence[float], window: int, threshold: float) -> bool:
    r"""Determine if a validation loss stopped improving, flat over its last values or above its best value.

    Arguments:
        history   : Validation losses, from the oldest to the newest.
        window    : Number of last validations considered.
        threshold : Relative standard deviation under which the last validations are flat.

    Returns:
        stagnated : Whether the training can stop.
    """

    # Security
    assert window > 1, f"ERROR - window must be > 1, got {window}."
    assert threshold > 0, f"ERROR - threshold must be > 0, got {threshold}."

    if len(history) < window:
        return False

    last = np.asarray(history[-window:])

    # A rising loss has a large spread, the second test stops it when no recent validation beats the older ones
    is_flat = bool(last.std() / last.mean() < threshold)
    no_progress = len(history) > window and bool(last.min() >= min(history[:-window]))

    return is_flat or no_progress


def save_state(path: Path, state: dict[str, Any]) -> None:
    r"""Save the state of a training, without ever leaving a partial file if the job stops meanwhile.

    Arguments:
        path  : Target file.
        state : State of the training.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.rename(path)


def load_state(path: Path, device: torch.device) -> dict[str, Any]:
    r"""Load the state of a training.

    Arguments:
        path   : File of the state.
        device : Target device of the tensors.

    Returns:
        state : State of the training.
    """

    return torch.load(path, map_location=device, weights_only=False)


def get_run_id() -> str | None:
    r"""Get the identifier shared by the Slurm jobs of a same training.

    Returns:
        run_id : Job ID of the first job of the training, None outside Slurm.
    """

    return os.environ.get("NEPTUNE_RUN_ID", os.environ.get("SLURM_JOB_ID"))


def submit_resume(restarts: int) -> None:
    r"""Submit the job that resumes this training if this one ends without success, e.g. at its time limit.

    Slurm drops the submitted job if this one completes, and the last job of a training submits none.

    Arguments:
        restarts : Number of jobs that may resume the training after the first one.
    """

    # Security
    assert restarts >= 0, f"ERROR - restarts must be >= 0, got {restarts}."

    job_id = os.environ.get("SLURM_JOB_ID")
    restart = int(os.environ.get("NEPTUNE_RESTART", 0))
    if job_id is None or restart >= restarts:
        return

    # The script of this job, written by dawgz, runs the same frozen training again
    job = subprocess.run(
        ["scontrol", "show", "job", "--oneliner", job_id],
        capture_output=True,
        text=True,
        check=True,
    )
    script = re.search(r"Command=(\S+)", job.stdout).group(1)

    # The next job keeps the environment of the submission, without the variables of this job and of torchrun
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(VARIABLES_JOB) and k not in VARIABLES_TORCHRUN
    }
    env.update(NEPTUNE_RUN_ID=get_run_id(), NEPTUNE_RESTART=str(restart + 1))

    subprocess.run(
        [
            "sbatch",
            f"--dependency=afternotok:{job_id}",
            "--kill-on-invalid-dep=yes",
            "--open-mode=append",
            "--export=ALL",
            script,
        ],
        env=env,
        check=True,
    )
