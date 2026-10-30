r"""Encode every split of the dataset into latent space with trained encoders, and compute the statistics of the latents."""

import argparse
import dask
import numpy as np
import torch
import xarray as xr

from dawgz import job, schedule
from pathlib import Path
from shaggy.tools import load
from torch.utils.data import DataLoader

import neptune.data
import neptune.data.dataloader
import neptune.data.dataset
import neptune.data.statistics
import neptune.data.weights
import neptune.model.encoders

from neptune.config import PATH_LATENTS, PATH_MODELS
from neptune.data import DATASET_DATES_TEST, DATASET_DATES_TRAINING, DATASET_DATES_VALIDATION
from neptune.data.dataloader import _init_worker
from neptune.data.dataset import NeptuneDataset
from neptune.data.statistics import OnlineStats
from neptune.model.encoders import Encoder, stack_dates
from neptune.tools import freeze_modules, load_configuration

# fmt: off
#
# ----- Splits of the dataset
#
SPLITS = {
    "training"   : DATASET_DATES_TRAINING,
    "validation" : DATASET_DATES_VALIDATION,
    "test"       : DATASET_DATES_TEST,
}


def update_statistics(stats: list[OnlineStats], z: np.ndarray) -> None:
    r"""Update the statistics of each latent channel with a batch of latents.

    Arguments:
        stats : Statistics of each latent channel (C_z,).
        z     : Latents (B, C_z, [Z',] Y', X').
    """

    for stats_channel, z_channel in zip(stats, z.swapaxes(0, 1), strict=True):
        stats_channel.update(z_channel)


def save_statistics(stats: list[OnlineStats], path: Path) -> None:
    r"""Save the mean and standard deviation of each latent channel in a statistics zarr store.

    Arguments:
        stats : Statistics of each latent channel (C_z,).
        path  : Path of the statistics zarr store.
    """

    statistics = xr.DataArray(
        np.array([[s.mean for s in stats], [s.std for s in stats]], dtype=np.float32),
        dims   = ["statistic", "channel"],
        coords = {"statistic": ["mean", "std"]},
    )

    xr.Dataset({"latents": statistics}).to_zarr(path, mode="w")


def encode(
    model           : Encoder,
    dates           : tuple[str, str],
    path            : Path,
    batch_size      : int,
    num_workers     : int,
    prefetch_factor : int,
) -> list[OnlineStats]:
    r"""Encode every state of a date range, in order, into a latent zarr store written one batch at a time.

    Arguments:
        model           : Encoder of a group of variables, in evaluation mode.
        dates           : First and last dates of the range 'YYYY-MM-DD'.
        path            : Path of the latent zarr store.
        batch_size      : Number of states encoded at once.
        num_workers     : Number of worker processes loading the states.
        prefetch_factor : Number of batches prefetched per worker.

    Returns:
        stats : Statistics of each latent channel over the date range (C_z,).
    """

    device  = next(model.parameters()).device
    dataset = NeptuneDataset(*dates, input_states=1, output_states=0, group=model.group)
    dims    = ["time", "channel", "level", "y", "x"] if model.group == "ocean" else ["time", "channel", "y", "x"]
    stats   = [OnlineStats() for _ in range(model.lat_channels)]

    # Every date is encoded, in order, the last batch being possibly smaller
    dataloader = DataLoader(
        dataset,
        batch_size      = batch_size,
        shuffle         = False,
        drop_last       = False,
        num_workers     = num_workers,
        prefetch_factor = prefetch_factor if num_workers > 0 else None,
        pin_memory      = True,
        worker_init_fn  = _init_worker,
    )

    for step, (x_inp_s, x_inp_o, _, _, _, dates_batch) in enumerate(dataloader):

        # The states are encoded with their date (the collate transposes the dates)
        states = (x_inp_s if model.group == "surface" else x_inp_o)[:, 0].to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z = model(stack_dates(states, dates_batch[0])).float().cpu().numpy()

        update_statistics(stats, z)

        # The store is created by the first batch, with one day per chunk, and extended by the next ones
        latents = xr.Dataset(
            {"latents": (dims, z)},
            coords={"time": np.array(dates_batch[0], dtype="datetime64[ns]")},
        )

        if step == 0:
            latents.to_zarr(path, mode="w", encoding={"latents": {"chunks": (1, *z.shape[1:])}})
        else:
            latents.to_zarr(path, append_dim="time")

    return stats


def encode_split(
    checkpoint_name : str,
    split           : str,
    batch_size      : int,
    num_workers     : int,
    prefetch_factor : int,
) -> None:
    r"""Encode a split of the dataset with a trained encoder, and save the statistics of the training latents.

    Arguments:
        checkpoint_name : Name of the checkpoint of the encoder in PATH_MODELS.
        split           : Split of the dataset, 'training', 'validation' or 'test'.
        batch_size      : Number of states encoded at once.
        num_workers     : Number of worker processes loading the states.
        prefetch_factor : Number of batches prefetched per worker.
    """

    # Prevent xarray/dask deadlocks inside DataLoader workers
    dask.config.set(scheduler="synchronous")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load(PATH_MODELS / checkpoint_name, Encoder, device=device)
    path   = PATH_LATENTS / checkpoint_name
    stats  = encode(model, SPLITS[split], path / f"{split}.zarr", batch_size, num_workers, prefetch_factor)

    # Only the training split defines the statistics used to standardize the latents
    if split == "training":
        save_statistics(stats, path / "statistics.zarr")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Encode every split of the dataset into latent space with trained encoders.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the .yml configuration of the encoding, e.g. configs/latents.yml.",
    )

    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster-based scheduling and 'async' for local execution.",
    )

    args            = parser.parse_args()
    config          = load_configuration(args.config)[0]
    config_encoding = config["Encoding"]

    # One task per pair of encoder and split, the encoders without checkpoint being skipped
    tasks = [(name, split) for name in config["Checkpoints"].values() if name is not None for split in SPLITS]

    # Security
    assert tasks, "ERROR - No checkpoint given, set the checkpoint of the surface or ocean encoder."

    # The jobs run the code as it is at submission
    freeze_modules([
        neptune.data,
        neptune.data.dataloader,
        neptune.data.dataset,
        neptune.data.statistics,
        neptune.data.weights,
        neptune.model.encoders,
    ])

    @job(array=len(tasks), **config["Cluster"])
    def latents(i: int) -> None:
        r"""Encode the split of the i-th task with its encoder."""
        encode_split(*tasks[i], **config_encoding)

    schedule(
        latents,
        name    = "NEPT-LATENTS",
        backend = args.backend,
        export  = "ALL",
    )
