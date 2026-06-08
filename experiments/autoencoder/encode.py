r"""Encode datasets into the latent space of a pre-trained autoencoder."""

import argparse
import dask
import torch

from datetime import date, timedelta
from dawgz import after, job, schedule
from shaggy.tools import load as s_load
from torch.utils.data import DataLoader

from neptune.config import PATH_EXP_AE_LATENTS, PATH_MODELS
from neptune.data.dataset import NeptuneDataset
from neptune.data.tools import assert_date_format
from neptune.data.weights import get_weights_mask
from neptune.tools import load_configuration


# fmt: off
#
def _build_windows(date_start: str, date_end: str, timestep: int) -> list[tuple[str, str]]:
    r"""Build consecutive date windows of size timestep days.

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


def encoding(
    checkpoint_name: str,
    split_name: str,
    date_start: str,
    date_end: str,
) -> None:
    r"""Encode a time window of a dataset split into latent space and save partial results.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
        split_name      : Dataset split name ('train', 'validation', or 'test').
        date_start      : Start date of the window, format 'YYYY-MM-DD'.
        date_end        : End date of the window, format 'YYYY-MM-DD'.
    """

    dask.config.set(scheduler="synchronous")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_mask = get_weights_mask(dim=2, device=device)
    model = s_load(PATH_MODELS / checkpoint_name, device=str(device)).eval()

    dataset = NeptuneDataset(date_start, date_end, standardized=True)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=1, pin_memory=device.type == "cuda")

    latent_list, dates_list = [], []
    with torch.no_grad():
        for x, batch_dates in dataloader:

            # Pushing to device and concatenating mask
            x = x.to(device)
            x_in = torch.cat([x, w_mask.expand(x.shape[0], -1, -1, -1)], dim=1)

            # Forward pass — only keep the latent representation
            z, _ = model(x_in)
            latent_list.append(z.cpu())
            dates_list.extend(list(batch_dates))

    # Saving partial results
    save_dir = PATH_EXP_AE_LATENTS /checkpoint_name / "parts"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": torch.cat(latent_list, dim=0), "dates": dates_list}, save_dir / f"{split_name}_{date_start}_{date_end}.pt")


def aggregate(
    checkpoint_name: str,
    split_name: str,
    date_start: str,
    date_end: str,
) -> None:
    r"""Aggregate partial latent files into a single dataset file and clean up.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
        split_name      : Dataset split name ('train', 'validation', or 'test').
        date_start      : Start date of the full split, format 'YYYY-MM-DD'.
        date_end        : End date of the full split, format 'YYYY-MM-DD'.
    """

    parts_dir = PATH_EXP_AE_LATENTS /checkpoint_name / "parts"
    save_dir = PATH_EXP_AE_LATENTS /checkpoint_name
    part_paths = sorted(parts_dir.glob(f"{split_name}_*.pt"))

    latent_list, dates_list = [], []
    for path in part_paths:
        d = torch.load(path, map_location="cpu")
        latent_list.append(d["latent"])
        dates_list.extend(d["dates"])

    torch.save(torch.cat(latent_list, dim=0), save_dir / f"{split_name}_{date_start}_{date_end}.pt")
    torch.save(dates_list, save_dir / f"{split_name}_dates_{date_start}_{date_end}.pt")

    # Deleting partial files
    for path in part_paths:
        path.unlink()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Encode datasets into the latent space of a pre-trained autoencoder.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the encoding .yml configuration file.",
    )

    args     = parser.parse_args()
    all_jobs = []

    for configs in load_configuration(args.config):

        config_cluster_encode    = configs["Cluster"]["encode"]
        config_cluster_aggregate = configs["Cluster"]["aggregate"]

        checkpoint_name = configs["Autoencoder"]["checkpoint_name"]
        if not checkpoint_name:
            raise ValueError("ERROR - Checkpoint name must be a non-empty string")

        timestep = configs["Encoding"]["timestep"]
        splits   = configs["Encoding"]["Splits"]

        # Checking that paths for saving latent representations exist
        (PATH_EXP_AE_LATENTS /checkpoint_name / "parts").mkdir(parents=True, exist_ok=True)

        for split_name, split_dates in splits.items():

            ds_start = split_dates["date_start"]
            ds_end   = split_dates["date_end"]

            WINDOWS = _build_windows(ds_start, ds_end, timestep)

            cp  = checkpoint_name
            sp  = split_name
            dss = ds_start
            dse = ds_end

            @job(array=len(WINDOWS), **config_cluster_encode)
            def encode_split(i: int, w: list = WINDOWS, cp: str = cp, sp: str = sp) -> None:
                start, end = w[i]
                encoding(cp, sp, start, end)

            @after(encode_split)
            @job(array=1, **config_cluster_aggregate)
            def aggregate_split(i: int, cp: str = cp, sp: str = sp, dss: str = dss, dse: str = dse) -> None:
                aggregate(cp, sp, dss, dse)

            all_jobs.extend([encode_split, aggregate_split])

    schedule(
        *all_jobs,
        name="AE-ENCODE",
        backend="slurm",
        export="ALL",
    )
