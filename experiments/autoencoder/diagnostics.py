r"""Launch diagnostics of autoencoder."""

import argparse

from datetime import date, timedelta
from dawgz import after, job, schedule

from neptune.config import PATH_DIAGNOSTICS
from neptune.data.tools import assert_date_format
from neptune.diagnostics.metrics import (
    clean_se,
    compute_and_save_maps,
    compute_and_save_power_spectra,
    compute_and_save_reconstructions,
    compute_and_save_se,
    compute_and_save_stats_mse,
)
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

    # Sanity checks of date formats
    assert_date_format(date_start)
    assert_date_format(date_end)

    # Building windows
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

    args            = parser.parse_args()
    configs         = load_configuration(args.config)[0]
    config_cluster_gpu = configs["Cluster"]["gpu"]
    config_cluster_cpu = configs["Cluster"]["cpu"]

    checkpoint_name      = configs["Autoencoder"]["checkpoint_name"]
    dates_reconstruction = configs["Reconstruction"]["dates"]
    if not checkpoint_name:
        raise ValueError("ERROR - Checkpoint name must be a non-empty string")

    date_start = configs["Evaluation"]["date_start"]
    date_end   = configs["Evaluation"]["date_end"]
    timestep   = configs["Evaluation"]["timestep"]

    WINDOWS = _build_windows(date_start, date_end, timestep)

    # Checking that paths for saving diagnostics exist
    (PATH_DIAGNOSTICS / checkpoint_name / "reconstructions").mkdir(parents=True, exist_ok=True)
    (PATH_DIAGNOSTICS / checkpoint_name / "se").mkdir(parents=True, exist_ok=True)
    (PATH_DIAGNOSTICS / checkpoint_name / "power_spectra").mkdir(parents=True, exist_ok=True)
    (PATH_DIAGNOSTICS / checkpoint_name / "rmse").mkdir(parents=True, exist_ok=True)
    (PATH_DIAGNOSTICS / checkpoint_name / "maps").mkdir(parents=True, exist_ok=True)

    # ---- Step 0: compute reconstructions for specific dates (GPU, independent) --

    @job(array=1, **config_cluster_gpu)
    def reconstruct(i: int) -> None:
        compute_and_save_reconstructions(checkpoint_name, dates_reconstruction)

    # ---- Step 1: compute SE + power spectra per date window (GPU, array job) ---

    @job(array=len(WINDOWS), **config_cluster_gpu)
    def compute(i: int) -> None:
        start, end = WINDOWS[i]
        compute_and_save_se(checkpoint_name, start, end)
        compute_and_save_power_spectra(checkpoint_name, start, end)

    # ---- Step 2a: aggregate per-channel RMSE statistics (CPU, after all windows)

    @after(compute)
    @job(array=1, **config_cluster_cpu)
    def aggregate_rmse(i: int) -> None:
        compute_and_save_stats_mse(checkpoint_name)

    # ---- Step 2b: compute per-pixel RMSE maps (CPU, after all windows) ---------

    @after(compute)
    @job(array=1, **config_cluster_cpu)
    def aggregate_maps(i: int) -> None:
        compute_and_save_maps(checkpoint_name)

    # ---- Step 3: delete SE files once both aggregations are done ---------------

    @after(aggregate_rmse)
    @after(aggregate_maps)
    @job(array=1, **config_cluster_cpu)
    def cleanup(i: int) -> None:
        clean_se(checkpoint_name)

    schedule(
        reconstruct,
        compute,
        aggregate_rmse,
        aggregate_maps,
        cleanup,
        name="AE-DIAG",
        backend="slurm",
        export="ALL",
    )
