r"""Launch diagnostics of autoencoder."""

import argparse

from dawgz import after, job, schedule

from neptune.config import PATH_DIAGNOSTICS
from neptune.data.tools import build_windows
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
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submit autoencoder diagnostics jobs.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the diagnostics .yml configuration file.",
    )

    args = parser.parse_args()
    configs = load_configuration(args.config)[0]
    config_cluster_gpu = configs["Cluster"]["gpu"]
    config_cluster_cpu = configs["Cluster"]["cpu"]

    checkpoint_name = configs["Autoencoder"]["checkpoint_name"]
    dates_reconstruction = configs["Reconstruction"]["dates"]
    if not checkpoint_name:
        raise ValueError("ERROR - Checkpoint name must be a non-empty string")

    date_start = configs["Evaluation"]["date_start"]
    date_end = configs["Evaluation"]["date_end"]
    timestep = configs["Evaluation"]["timestep"]

    WINDOWS = build_windows(date_start, date_end, timestep)

    # Checking that paths for saving diagnostics exist
    (PATH_DIAGNOSTICS / checkpoint_name / "reconstructions").mkdir(parents=True, exist_ok=True)
    (PATH_DIAGNOSTICS / checkpoint_name / "se").mkdir(parents=True, exist_ok=True)
    (PATH_DIAGNOSTICS / checkpoint_name / "power_spectra").mkdir(parents=True, exist_ok=True)
    (PATH_DIAGNOSTICS / checkpoint_name / "rmse").mkdir(parents=True, exist_ok=True)
    (PATH_DIAGNOSTICS / checkpoint_name / "maps").mkdir(parents=True, exist_ok=True)

    @job(array=1, **config_cluster_gpu)
    def reconstruct(i: int) -> None:
        compute_and_save_reconstructions(checkpoint_name, dates_reconstruction)

    @job(array=len(WINDOWS), **config_cluster_gpu)
    def compute(i: int) -> None:
        start, end = WINDOWS[i]
        compute_and_save_se(checkpoint_name, start, end)
        compute_and_save_power_spectra(checkpoint_name, start, end)

    @after(compute)
    @job(array=1, **config_cluster_cpu)
    def aggregate_rmse(i: int) -> None:
        compute_and_save_stats_mse(checkpoint_name)

    @after(compute)
    @job(array=1, **config_cluster_cpu)
    def aggregate_maps(i: int) -> None:
        compute_and_save_maps(checkpoint_name)

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
