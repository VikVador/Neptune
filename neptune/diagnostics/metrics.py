r"""A collection of functions for diagnosing autoencoders."""

__all__ = [
    "power_spectrum",
    "compute_and_save_power_spectra",
    "compute_and_save_rmse",
]

import dask
import torch

from shaggy.tools import load as s_load
from torch import Tensor
from torch.utils.data import DataLoader

from neptune.config import PATH_DIAGNOSTICS, PATH_MODELS
from neptune.data.dataset import NeptuneDataset
from neptune.data.weights import get_weights_mask, get_weights_state_mask


# fmt: off
#
def power_spectrum(u: Tensor, mask: Tensor | None = None) -> Tensor:
    r"""Compute the isotropic 2D power spectrum per channel, per batch element.

    Arguments:
        u    : State tensor of shape (B, C, Y, X).
        mask : Ocean mask of shape (1, C, Y, X).

    Returns:
        spectrum : Isotropic power spectrum per sample per channel (B, C, K).
    """

    # Extract dimensions and device info
    B, C_in, Y_in, X_in = u.shape
    device = u.device

    if mask is not None:
        assert mask.shape == (
            1,
            C_in,
            Y_in,
            X_in,
        ), f"Expected mask shape (1, {C_in}, {Y_in}, {X_in}), got {mask.shape}"
        u = u.masked_fill(mask.expand_as(u) == 0, 0.0)
    else:
        u = u.nan_to_num(0.0)

    # Computing 2D Fast Fourier Transform
    fft = torch.fft.rfft2(u, norm="ortho")
    psd = fft.real**2 + fft.imag**2

    # Building radial frequency grid
    ky     = torch.fft.fftfreq( Y_in, device=device) * Y_in
    kx     = torch.fft.rfftfreq(X_in, device=device) * X_in
    k_grid = torch.round(torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).long()
    K      = int(min(Y_in, X_in)) // 2 + 1
    k_flat = k_grid.clamp(0, K - 1).reshape(-1)

    # Counting pixels per radial bin
    count = torch.zeros(K, device=device)
    count.scatter_add_(0, k_flat, torch.ones(k_flat.numel(), device=device))
    count = count.clamp(min=1)

    # Averaging power in each radial bin to get isotropic spectrum
    psd_flat = psd.reshape(B, C_in, -1)
    k_exp    = k_flat[None, None, :].expand(B, C_in, -1)
    spectrum = torch.zeros(B, C_in, K, device=device)
    spectrum.scatter_add_(dim=2, index=k_exp, src=psd_flat)
    spectrum = spectrum / count[None, None, :]

    return spectrum


def _setup(checkpoint_name: str) -> tuple:
    r"""Shared initialisation for diagnostics functions.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.

    Returns:
        device       : Torch device (cuda if available, else cpu).
        w_mask       : 3D ocean mask  (1, Z, Y, X).
        w_state_mask : Channel-aligned mask (1, C, Y, X).
        model        : Loaded model in eval mode.
    """
    dask.config.set(scheduler="synchronous")
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_mask       = get_weights_mask(dim=2,       device=device)
    w_state_mask = get_weights_state_mask(dim=2, device=device)
    model        = s_load(PATH_MODELS / checkpoint_name, device=str(device)).eval()
    return device, w_mask, w_state_mask, model


def compute_and_save_power_spectra(checkpoint_name: str, date_start: str, date_end: str) -> None:
    r"""Load the model, run inference, compute and saves power spectra.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
        date_start      : Start date of the subset, format 'YYYY-MM-DD'.
        date_end        : End date of the subset, format 'YYYY-MM-DD'.
    """

    device, w_mask, w_state_mask, model = _setup(checkpoint_name)

    # Loading dataset and creating DataLoader
    dataset = NeptuneDataset(
        date_start,
        date_end,
        standardized=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=device.type == "cuda"
    )

    gt_list, rec_list = [], []
    with torch.no_grad():
        for x, _ in dataloader:

            # Pushing to device and concatenating mask
            x = x.to(device)
            x_in = torch.cat([x, w_mask.expand(x.shape[0], -1, -1, -1)], dim=1)

            # Forward pass
            _, x_hat = model(x_in)

            # Compute power spectra with mask applied (land → nan)
            gt_list.append( power_spectrum(x,     mask=w_state_mask).cpu())
            rec_list.append(power_spectrum(x_hat, mask=w_state_mask).cpu())

    # Saving results
    save_dir = PATH_DIAGNOSTICS / checkpoint_name / "power_spectra"
    data = {"ground_truth": torch.cat(gt_list, dim=0), "reconstruction": torch.cat(rec_list, dim=0)}
    torch.save(data, save_dir / f"power_spectra_{date_start}_{date_end}.pt")


def compute_and_save_rmse(checkpoint_name: str, date_start: str, date_end: str) -> None:
    r"""Load the model, run inference, compute and save per-day RMSE.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
        date_start      : Start date of the subset, format 'YYYY-MM-DD'.
        date_end        : End date of the subset, format 'YYYY-MM-DD'.
    """

    device, w_mask, w_state_mask, model = _setup(checkpoint_name)

    # Loading datasets and creating dataLoaders
    dataset_std, dataset_raw = (
        NeptuneDataset(date_start, date_end, standardized=s) for s in (True, False)
    )

    dataloader_std, dataloader_raw = (
        DataLoader(ds, batch_size=4, num_workers=1, pin_memory=device.type == "cuda") for ds in (dataset_std, dataset_raw)
    )

    rmse_std_list, rmse_raw_list = [], []
    with torch.no_grad():
        for (x_std, _), (x_raw, _) in zip(dataloader_std, dataloader_raw, strict=False):

            # Pushing to device and concatenating mask
            x_std = x_std.to(device)
            x_raw = x_raw.to(device)
            x_in = torch.cat([x_std, w_mask.expand(x_std.shape[0], -1, -1, -1)], dim=1)

            # Forward pass
            _, x_hat = model(x_in)

            # Applying mask on reconstructions
            x_hat_m = x_hat.masked_fill(w_state_mask.expand_as(x_hat) == 0, float("nan"))

            # Computing standardised RMSE
            x_std_m  = x_std.masked_fill(w_state_mask.expand_as(x_std) == 0, float("nan"))
            rmse_std = torch.sqrt(torch.nanmean((x_std_m - x_hat_m) ** 2, dim=(2, 3)))
            rmse_std_list.append(rmse_std.cpu())

            # Computing physical RMSE
            x_hat_raw = dataset_std.unstandardize(x_hat_m)
            x_raw_m   = x_raw.masked_fill(w_state_mask.expand_as(x_raw) == 0, float("nan"))
            rmse_raw  = torch.sqrt(torch.nanmean((x_raw_m - x_hat_raw) ** 2, dim=(2, 3)))
            rmse_raw_list.append(rmse_raw.cpu())

    # Saving results
    save_dir = PATH_DIAGNOSTICS / checkpoint_name / "rmse"
    data = {"rmse_standardized": torch.cat(rmse_std_list, dim=0), "rmse_physical": torch.cat(rmse_raw_list, dim=0)}
    torch.save(data, save_dir / f"rmse_{date_start}_{date_end}.pt")
