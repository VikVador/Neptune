r"""Launch an autoencoder diagnostics pipeline."""

import argparse
import dawgz

from datetime import date, timedelta

from neptune.config import PATH_DIAGNOSTICS
from neptune.diagnostics import compute_and_save_power_spectra, compute_and_save_rmse
from neptune.tools import load_configuration


# fmt: off
#
def _build_windows(date_start: str, date_end: str, timestep: int) -> list[tuple[str, str]]:
    r"""Build consecutive date windows of size `timestep` days.

    Arguments:
        date_start : First day of the range, format 'YYYY-MM-DD'.
        date_end   : Last day of the range, format 'YYYY-MM-DD'.
        timestep   : Number of days per window.

    Returns:
        windows : List of (start, end) tuples covering the full range.
    """
    start, end, windows = date.fromisoformat(date_start), date.fromisoformat(date_end), []
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=timestep - 1), end)
        windows.append((current.isoformat(), window_end.isoformat()))
        current = window_end + timedelta(days=1)
    return windows


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submit autoencoder diagnostics jobs.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the diagnostics .yml configuration file.",
    )

    args           = parser.parse_args()
    configs        = load_configuration(args.config)[0]
    config_cluster = configs["Cluster"]

    checkpoint_name = configs["Autoencoder"]["checkpoint_name"]
    date_start      = configs["Evaluation"]["date_start"]
    date_end        = configs["Evaluation"]["date_end"]
    timestep        = configs["Evaluation"]["timestep"]

    WINDOWS = _build_windows(date_start, date_end, timestep)

    # Checking that paths for saving diagnostics exist
    (PATH_DIAGNOSTICS / checkpoint_name / "rmse").mkdir(parents=True, exist_ok=True)
    (PATH_DIAGNOSTICS / checkpoint_name / "power_spectra").mkdir(parents=True, exist_ok=True)

    @dawgz.job(array=len(WINDOWS), **config_cluster)
    def diagnostic(i: int) -> None:

        # Get the date window for this job
        start, end = WINDOWS[i]

        # Compute and save diagnostics for this window
        compute_and_save_rmse(checkpoint_name, start, end)
        compute_and_save_power_spectra(checkpoint_name, start, end)

    dawgz.schedule(
        diagnostic,
        name="AE-DIAG",
        backend="slurm",
        export="ALL"
    )
