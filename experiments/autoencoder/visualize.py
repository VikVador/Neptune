r"""Launch visualization of autoencoder diagnostics."""

import argparse
import dawgz

from neptune.diagnostics.figures import (
    visualize_error_maps_x,
    visualize_error_maps_y,
    visualize_error_maps_z,
    visualize_error_vertical,
    visualize_reconstructions,
    visualize_spectra,
)
from neptune.tools import load_configuration

# fmt: off
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Submit autoencoder visualization jobs.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the visualization .yml configuration file.",
    )

    args     = parser.parse_args()
    all_jobs = []

    for configs in load_configuration(args.config):

        # Loading configuration parameters
        config_cluster  = configs["Cluster"]
        checkpoint_name = configs["Autoencoder"]["checkpoint_name"]

        # Security check for checkpoint name
        if not checkpoint_name:
            raise ValueError("ERROR - Checkpoint name must be a non-empty string")

        # Creating and launching jobs
        cp = checkpoint_name

        @dawgz.job(array=1, **config_cluster)
        def vis_vertical(i: int, cp: str=cp) -> None:
            visualize_error_vertical(cp)

        @dawgz.job(array=1, **config_cluster)
        def vis_spectra(i: int, cp: str=cp) -> None:
            visualize_spectra(cp)

        @dawgz.job(array=1, **config_cluster)
        def vis_maps_x(i: int, cp: str=cp) -> None:
            visualize_error_maps_x(cp)

        @dawgz.job(array=1, **config_cluster)
        def vis_maps_y(i: int, cp: str=cp) -> None:
            visualize_error_maps_y(cp)

        @dawgz.job(array=1, **config_cluster)
        def vis_maps_z(i: int, cp: str=cp) -> None:
            visualize_error_maps_z(cp)

        @dawgz.job(array=1, **config_cluster)
        def vis_recon(i: int, cp: str=cp) -> None:
            visualize_reconstructions(cp)

        all_jobs.extend([
            vis_vertical,
            vis_spectra,
            vis_maps_x,
            vis_maps_y,
            vis_maps_z,
            vis_recon,
        ])

    dawgz.schedule(
        *all_jobs,
        name="AE-VISU",
        backend="slurm",
        export="ALL",
    )
