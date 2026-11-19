r"""Launch the campaign: each job transfers forecast runs from Climato and compares them, a few jobs at a time."""

import argparse
import comparisons
import errors
import json
import observations

from comparisons import compute_run
from dawgz import ensure, job, schedule
from errors import PATH_PREPARATION, PATH_RUNS

from neptune.tools import freeze_modules, load_configuration

# fmt: off
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch the comparison of the forecast runs.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to the .yml configuration file.")
    parser.add_argument("--backend", "-b", type=str, default="slurm", choices=["slurm", "async"], help="Computation backend.")
    parser.add_argument("--runs", "-r", type=str, nargs="*", default=None, help="Runs 'YYYY-MM-DD' to compare, all by default.")
    args     = parser.parse_args()
    config   = load_configuration(args.config)[0]
    campaign = config["Campaign"]

    with open(PATH_PREPARATION / "runs.json") as file:
        inventory = json.load(file)

    # Security
    missing = sorted(set(args.runs or []) - set(inventory))
    assert not missing, f"ERROR - Runs {missing} are not in the inventory of Climato."

    runs   = args.runs or list(inventory)
    chunks = [runs[i : i + campaign["runs_per_job"]] for i in range(0, len(runs), campaign["runs_per_job"])]

    # The jobs run the code as it is at submission
    freeze_modules([errors, comparisons, observations])

    @ensure(lambda i: all((PATH_RUNS / f"{run}.nc").exists() for run in chunks[i]))
    @job(array=len(chunks), array_throttle=campaign["simultaneous_jobs"], **config["Cluster"]["runs"])
    def compare(i: int) -> None:
        r"""Transfer and compare the runs of the i-th chunk, one after the other, skipping those already done.

        Arguments:
            i : Index of the chunk of runs.
        """

        for run in chunks[i]:
            if not (PATH_RUNS / f"{run}.nc").exists():
                compute_run(run, [tuple(f) for f in inventory[run]["files"]])

    schedule(compare, name="FERR-RUNS", backend=args.backend, export="ALL", prune=True)
