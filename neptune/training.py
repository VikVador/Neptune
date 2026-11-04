r"""Tools to resume a training on Slurm."""

__all__ = [
    "get_run_id",
    "submit_resume",
]

import os
import re
import subprocess

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
