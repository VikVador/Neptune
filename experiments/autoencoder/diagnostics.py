r"""Diagnostics Autoencoder."""

import argparse
import calendar

from dawgz import job, schedule

from neptune.data import DATASET_DATES_VALIDATION
from neptune.diagnostics import compute_and_save_spectra, compute_and_save_vrmse
from neptune.tools import load_configuration


# fmt: off
#
def _build_months(date_start: str, date_end: str) -> list[tuple[str, str]]:
    r"""Build (month_start, month_end) tuples covering the given date range.

    Arguments:
        date_start : First day of the range, format 'YYYY-MM-DD'.
        date_end   : Last day of the range, format 'YYYY-MM-DD'.

    Returns:
        months : List of (start, end) tuples, one per calendar month.
    """
    y_start, m_start = int(date_start[:4]), int(date_start[5:7])
    y_end,   m_end   = int(date_end[:4]),   int(date_end[5:7])

    months, y, m = [], y_start, m_start
    while (y, m) <= (y_end, m_end):
        _, last_day = calendar.monthrange(y, m)
        months.append((f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last_day:02d}"))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run autoencoder diagnostics.")
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the diagnostics .yml configuration file.",
    )
    parser.add_argument(
        "--backend", "-b", type=str, default="slurm",
        choices=["slurm", "async"],
        help="Computation backend: 'slurm' for cluster, 'async' for local testing.",
    )

    args            = parser.parse_args()
    configs         = load_configuration(args.config)
    config_cluster  = configs[0]["Cluster"]
    checkpoint_name = configs[0]["Autoencoder"]["checkpoint_name"]

    nodes         = config_cluster.get("nodes",         1)
    gpus_per_node = config_cluster.get("gpus-per-node", 1)
    cpus_per_node = config_cluster.get("cpus-per-node", 8)
    ram_per_node  = config_cluster.get("ram-per-node",  "60GB")
    partition     = config_cluster.get("partition",     "gpu")
    account       = config_cluster.get("account")
    time_limit    = config_cluster.get("time",          "00:30:00")

    MONTHS = _build_months(*DATASET_DATES_VALIDATION)

    # -------------------------------------------------------------------------
    # Local execution — runs a single month for quick testing
    # -------------------------------------------------------------------------
    if args.backend == "async":
        date_start, date_end = MONTHS[0]
        compute_and_save_vrmse(  checkpoint_name, date_start, date_end)
        compute_and_save_spectra(checkpoint_name, date_start, date_end)

    # -------------------------------------------------------------------------
    # Cluster execution — 36 VRMSE + 36 spectra jobs, then two analysis jobs
    # -------------------------------------------------------------------------
    else:
        compute_cfg = dict(
            nodes     = nodes,
            gpus      = gpus_per_node,
            cpus      = cpus_per_node,
            ram       = ram_per_node,
            time      = time_limit,
            account   = account,
            partition = partition,
        )
        analysis_cfg = dict(
            nodes     = 1,
            cpus      = 4,
            ram       = "16GB",
            time      = "00:30:00",
            account   = account,
            partition = "batch",
        )

        @job(**compute_cfg)
        def compute_vrmse(i: int) -> None:
            date_start, date_end = MONTHS[i]
            compute_and_save_vrmse(checkpoint_name, date_start, date_end)

        @job(**compute_cfg)
        def compute_spectra(i: int) -> None:
            date_start, date_end = MONTHS[i]
            compute_and_save_spectra(checkpoint_name, date_start, date_end)

        vrmse_jobs   = [compute_vrmse(i)   for i in range(len(MONTHS))]
        spectra_jobs = [compute_spectra(i) for i in range(len(MONTHS))]

        def _analyze_vrmse() -> None:
            pass  # TODO: load results and log visualisations to W&B

        def _analyze_spectra() -> None:
            pass  # TODO: load results and log visualisations to W&B

        analyze_vrmse_job   = job(_analyze_vrmse,   **analysis_cfg, name="AE-DIAG-VRMSE-ANALYSIS")  ().after(*vrmse_jobs)
        analyze_spectra_job = job(_analyze_spectra, **analysis_cfg, name="AE-DIAG-SPECTRA-ANALYSIS")().after(*spectra_jobs)

        schedule(
            analyze_vrmse_job,
            analyze_spectra_job,
            name    = "AE-DIAG",
            backend = "slurm",
            export  = "ALL",
        )
