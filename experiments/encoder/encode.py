r"""Encode datasets into the latent space of pre-trained surface and ocean encoders."""

import argparse
import dask
import torch

from dawgz import after, job, schedule
from pathlib import Path
from shaggy.models.cae import ConvEncoder
from shaggy.tools import load as s_load
from torch.utils.data import DataLoader

from neptune.config import PATH_LATENTS, PATH_MODELS
from neptune.data import DATASET_DATES_TEST, DATASET_DATES_TRAINING, DATASET_DATES_VALIDATION
from neptune.data.dataset import NeptuneDataset
from neptune.data.tools import build_windows
from neptune.data.weights import get_weights_mask
from neptune.tools import extract_model_hash, load_configuration


# fmt: off
#
def _load_encoder(checkpoint_name: str, device: torch.device) -> torch.nn.Module:
    r"""Load a pre-trained ConvEncoder checkpoint in eval mode."""
    return s_load(PATH_MODELS / checkpoint_name, ConvEncoder, device=str(device))


def encoding(
    checkpoint_surface  : str,
    checkpoint_ocean    : str,
    split               : str,
    date_start          : str,
    date_end            : str,
    save_dir            : Path,
) -> None:
    r"""Encode a time window of a dataset split with both encoders and save the partial result.

    Arguments:
        checkpoint_surface  : Name of the surface encoder checkpoint directory under PATH_MODELS.
        checkpoint_ocean    : Name of the ocean encoder checkpoint directory under PATH_MODELS.
        split               : Dataset split name ('train', 'validation' or 'test').
        date_start          : Start date of the window, format 'YYYY-MM-DD'.
        date_end            : End date of the window, format 'YYYY-MM-DD'.
        save_dir            : Directory where the partial latent file is saved.
    """
    dask.config.set(scheduler="synchronous")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_full      = get_weights_mask(dim=1, device=device)
    w_mask_surface = mask_full[0][None, None]
    w_mask_ocean   = mask_full[None, None]

    encoder_surface = _load_encoder(checkpoint_surface, device)
    encoder_ocean   = _load_encoder(checkpoint_ocean, device)

    dataset    = NeptuneDataset(date_start, date_end, standardized=True)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=1, pin_memory=device.type == "cuda")

    z_surface_list, z_ocean_list, dates_list = [], [], []
    with torch.no_grad():
        for x_s, x_o, dates in dataloader:
            x_s, x_o = x_s.to(device), x_o.to(device)
            x_s_in = torch.cat([x_s, w_mask_surface.expand(x_s.shape[0], *([-1] * (w_mask_surface.dim() - 1)))], dim=1)
            x_o_in = torch.cat([x_o, w_mask_ocean.expand(x_o.shape[0], *([-1] * (w_mask_ocean.dim() - 1)))], dim=1)

            z_surface_list.append(encoder_surface(x_s_in).cpu())
            z_ocean_list.append(encoder_ocean(x_o_in).cpu())
            dates_list.extend(dates)

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "z_surface" : torch.cat(z_surface_list, dim=0),
            "z_ocean"   : torch.cat(z_ocean_list, dim=0),
            "dates"     : dates_list,
        },
        save_dir / f"{split}_{date_start}_{date_end}.pt",
    )


def aggregate(split: str, parts_dir: Path, save_dir: Path) -> None:
    r"""Aggregate the partial latent files of a split into a single file and clean up.

    Arguments:
        split     : Dataset split name ('train', 'validation' or 'test').
        parts_dir : Directory containing the partial {split}_{date_start}_{date_end}.pt files.
        save_dir  : Directory where the aggregated {split}.pt file is saved.
    """
    part_paths = sorted(parts_dir.glob(f"{split}_*.pt"))

    z_surface_list, z_ocean_list, dates_list = [], [], []
    for path in part_paths:
        part = torch.load(path, map_location="cpu", weights_only=False)
        z_surface_list.append(part["z_surface"])
        z_ocean_list.append(part["z_ocean"])
        dates_list.extend(part["dates"])

    torch.save(
        {
            "z_surface" : torch.cat(z_surface_list, dim=0),
            "z_ocean"   : torch.cat(z_ocean_list, dim=0),
            "dates"     : dates_list,
        },
        save_dir / f"{split}.pt",
    )

    for path in part_paths:
        path.unlink()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Encode datasets into the latent space of pre-trained encoders.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the encoding .yml configuration file.",
    )

    args = parser.parse_args()
    all_jobs = []

    for configs in load_configuration(args.config):
        config_cluster_encode    = configs["Cluster"]["encode"]
        config_cluster_aggregate = configs["Cluster"]["aggregate"]

        checkpoint_surface = configs["Encoders"]["checkpoint_surface"]
        checkpoint_ocean   = configs["Encoders"]["checkpoint_ocean"]
        if not checkpoint_surface or not checkpoint_ocean:
            raise ValueError("ERROR - Both checkpoint_surface and checkpoint_ocean must be non-empty strings.")

        timestep = configs["Encoding"]["timestep"]

        # Latents live next to the models, named after the two encoders they come from
        hash_surface = extract_model_hash(checkpoint_surface)
        hash_ocean   = extract_model_hash(checkpoint_ocean)
        save_dir     = PATH_LATENTS / f"latent_{hash_surface}_{hash_ocean}"
        parts_dir    = save_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)

        splits = {
            "train"      : DATASET_DATES_TRAINING,
            "validation" : DATASET_DATES_VALIDATION,
            "test"       : DATASET_DATES_TEST,
        }

        for split, (ds_start, ds_end) in splits.items():
            windows = build_windows(ds_start, ds_end, timestep)

            @job(array=len(windows), **config_cluster_encode)
            def encode_split(
                i    : int,
                w    : list = windows,
                sp   : str  = split,
                cs   : str  = checkpoint_surface,
                co   : str  = checkpoint_ocean,
                pdir : Path = parts_dir,
            ) -> None:
                start, end = w[i]
                encoding(cs, co, sp, start, end, pdir)

            @after(encode_split)
            @job(array=1, **config_cluster_aggregate)
            def aggregate_split(
                i    : int,
                sp   : str  = split,
                pdir : Path = parts_dir,
                sdir : Path = save_dir,
            ) -> None:
                aggregate(sp, pdir, sdir)

            all_jobs.extend([encode_split, aggregate_split])

    schedule(
        *all_jobs,
        name="E3D-ENCODE",
        backend="slurm",
        export="ALL",
    )
