r"""Launch the diagnostics of a FGN on a split of the dataset: metrics, maps and animations of its forecasts."""

import argparse
import cloudpickle
import os

from dawgz import after, job, schedule
from shaggy.tools import load_config

from neptune.config import PATH_MODELS
from neptune.data import DATASET_VARIABLES
from neptune.data.dataset import NeptuneDataset
from neptune.diagnostics import SPLITS
from neptune.diagnostics.figures import (
    log_figures,
    plot_animations,
    plot_score_global,
    plot_score_local,
    plot_score_maps,
)
from neptune.diagnostics.metrics import (
    aggregate_maps,
    aggregate_metrics,
    clean,
    compute_animations,
    compute_metrics,
)
from neptune.tools import load_configuration

# fmt: off
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch the diagnostics of a FGN on a split of the dataset.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the diagnostics .yml configuration file.",
    )

    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster-based scheduling and 'async' for local execution.",
    )

    args              = parser.parse_args()
    config            = load_configuration(args.config)[0]
    cluster           = config["Cluster"]
    evaluation        = config["Evaluation"]
    config_animations = config["Animations"]
    config_wandb      = config["WandB"]

    checkpoint_name = config["Model"]["checkpoint_name"]
    split           = evaluation["split"]
    days            = evaluation["days"]
    members         = evaluation["members"]
    chunk           = evaluation["chunk"]
    noise           = evaluation["noise_forcing_levels"] if evaluation["noise_forcing"] else [0.0] * days

    # Security
    assert len(noise) == days, f"ERROR - noise_forcing_levels must have {days} elements, got {len(noise)}."

    # Windows | Forecasts started every `stride` days of the split, animations from the dates of their last input state
    n       = load_config(PATH_MODELS / checkpoint_name).input_states
    dataset = NeptuneDataset(*SPLITS[split], input_states=n, output_states=days)
    starts  = [dataset.dates[i + n - 1] for i in range(len(dataset))]
    windows = list(range(0, len(dataset), evaluation["stride"]))
    chunks  = [windows[i : i + evaluation["dates_per_job"]] for i in range(0, len(windows), evaluation["dates_per_job"])]

    # Security
    missing = sorted(set(config_animations["dates"]) - set(starts))
    assert not missing, f"ERROR - Animations dates {missing} are not in the forecasts of the {split} split ({starts[0]} to {starts[-1]})."

    animate = [starts.index(date) for date in config_animations["dates"]]

    # GPU | Exported to the jobs, against the fragmentation of the memory by the varying sizes of the tensors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Freeze modules, such that the jobs use the code as it is at submission
    import neptune.data
    import neptune.data.dataset
    import neptune.data.weights
    import neptune.diagnostics
    import neptune.diagnostics.figures
    import neptune.diagnostics.metrics
    import neptune.model.fgn
    for _mod in [
        neptune.data,
        neptune.data.dataset,
        neptune.data.weights,
        neptune.diagnostics,
        neptune.diagnostics.figures,
        neptune.diagnostics.metrics,
        neptune.model.fgn,
    ]:
        cloudpickle.register_pickle_by_value(_mod)

    @job(array=len(chunks), **cluster["metrics"])
    def metrics(i: int) -> None:
        r"""Forecast the i-th chunk of windows and save their metrics."""
        compute_metrics(checkpoint_name, split, chunks[i], days, members, chunk, noise, job=i)

    @job(**cluster["forecasts"])
    def forecasts() -> None:
        r"""Forecast every animated window, one after the other, and save the fields of their animations."""
        for index in animate:
            compute_animations(
                checkpoint_name, split, index, days, members, chunk, noise,
                config_animations["levels"], config_animations["sections"],
            )

    @after(forecasts)
    @job(array=len(animate) * len(DATASET_VARIABLES), **cluster["figures"])
    def animations(i: int) -> None:
        r"""Animate one variable of one animated window."""
        date, variable = starts[animate[i // len(DATASET_VARIABLES)]], DATASET_VARIABLES[i % len(DATASET_VARIABLES)]
        plot_animations(checkpoint_name, split, date, variable, config_animations["duration"])

    @after(metrics)
    @job(**cluster["figures"])
    def metrics_figures() -> None:
        r"""Aggregate the metrics of every forecast and plot them per level (score-local) and per layer (score-global)."""

        scores = aggregate_metrics(checkpoint_name, split, members)
        plot_score_local(scores, checkpoint_name, split)
        plot_score_global(scores, checkpoint_name, split)

    @after(metrics)
    @job(array=2, **cluster["figures"])
    def maps_figures(i: int) -> None:
        r"""Aggregate and plot the maps of the first or last forecasted day."""
        plot_score_maps(aggregate_maps(checkpoint_name, split, (1, days)[i], members), checkpoint_name, split, ("first", "last")[i])

    @after(animations, metrics_figures, maps_figures)
    @job(**cluster["figures"])
    def log() -> None:
        r"""Log the figures and animations to Weights & Biases."""
        log_figures(checkpoint_name, split, config_wandb)

    @after(animations, metrics_figures, maps_figures)
    @job(**cluster["cleanup"])
    def cleanup() -> None:
        r"""Delete the intermediate results, i.e. the metrics, the sums of the maps and the animated fields."""
        clean(checkpoint_name, split)

    schedule(
        log,
        cleanup,
        name="NEPT-DIAG",
        backend=args.backend,
        export="ALL",
    )
