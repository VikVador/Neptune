r"""Metric functions for evaluating autoencoder reconstruction quality."""

__all__ = [
    "vrmse",
    "power_spectrum",
    "compute_and_save_vrmse",
    "compute_and_save_spectra",
]

import dask
import numpy as np
import torch

from shaggy.tools import load as s_load
from torch import Tensor
from torch.utils.data import DataLoader

from neptune.config import PATH_DIAGNOSTICS, PATH_MODELS
from neptune.data import DATASET_REGION, DATASET_VARIABLES_OCEAN, DATASET_VARIABLES_SURFACE
from neptune.data.dataset import NeptuneDataset
from neptune.data.weights import get_weights_mask, get_weights_state_mask

# fmt: off
#
# Constants derived from the dataset layout
Z = DATASET_REGION["deptht"].stop - DATASET_REGION["deptht"].start
C = len(DATASET_VARIABLES_SURFACE) + len(DATASET_VARIABLES_OCEAN) * Z


def vrmse(
    u: Tensor,
    v: Tensor,
    mask: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    r"""Compute the Variance-normalised RMSE per channel over a batch.

    Arguments:
        u    : Original standardised tensor of shape (B, C, Y, X) [].
        v    : Reconstructed standardised tensor of shape (B, C, Y, X) [].
        mask : Ocean mask of shape (C, Y, X), 1 on sea and 0 on land [].
        eps  : Small constant added to the denominator for numerical stability [].

    Returns:
        score : VRMSE per channel, shape (C,) [].
    """
    # Count valid (ocean) pixels per channel — shape (C,)
    n_valid = mask.sum(dim=(-2, -1)).clamp(min=1)  # (C,)

    # Expand mask to batch dimension for masked reductions
    mask_b = mask.unsqueeze(0)  # (1, C, Y, X)

    # Masked spatial+batch mean of u: sum over (B, Y, X), divide by B * n_valid
    B = u.shape[0]
    u_bar = (u * mask_b).sum(dim=(0, -2, -1)) / (B * n_valid)  # (C,)

    # Masked MSE: mean over valid pixels across batch
    mse = ((u - v) ** 2 * mask_b).sum(dim=(0, -2, -1)) / (B * n_valid)  # (C,)

    # Masked spatial variance of u: mean over valid pixels across batch
    u_bar_b = u_bar[None, :, None, None]  # (1, C, 1, 1)
    var = ((u - u_bar_b) ** 2 * mask_b).sum(dim=(0, -2, -1)) / (B * n_valid)  # (C,)

    return torch.sqrt(mse / (var + eps))


def power_spectrum(
    u: Tensor,
    mask: Tensor,
) -> Tensor:
    r"""Compute the isotropic 2D power spectrum per channel, averaged over a batch.

    Arguments:
        u    : Standardised tensor of shape (B, C, Y, X) [].
        mask : Ocean mask of shape (C, Y, X), 1 on sea and 0 on land [].

    Returns:
        spectrum : Mean isotropic power spectrum per channel, shape (C, K) [].
                   K = min(Y, X) // 2 + 1. Wavenumber bin k corresponds to
                   radial frequency k (in grid units).
    """
    B, C_in, Y, X = u.shape
    device = u.device

    # Apply mask before FFT (land pixels remain 0)
    u_masked = u * mask.unsqueeze(0)  # (B, C, Y, X)

    # 2D FFT — rfft2 exploits real-valued input; output shape (B, C, Y, X//2+1)
    fft = torch.fft.rfft2(u_masked, norm="ortho")
    psd = fft.real ** 2 + fft.imag ** 2  # (B, C, Y, X//2+1)

    # Build radial frequency grid (k in grid units)
    ky = torch.fft.fftfreq(Y, device=device) * Y    # (Y,)
    kx = torch.fft.rfftfreq(X, device=device) * X   # (X//2+1,)
    k_grid = torch.round(torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).long()  # (Y, X//2+1)

    K = int(min(Y, X)) // 2 + 1
    k_flat = k_grid.clamp(0, K - 1).reshape(-1)     # (Y*(X//2+1),)

    # Count pixels per radial bin
    count = torch.zeros(K, device=device)
    count.scatter_add_(0, k_flat, torch.ones(k_flat.numel(), device=device))
    count = count.clamp(min=1)

    # Flatten spatial dims, scatter-add PSD into radial bins
    psd_flat = psd.reshape(B, C_in, -1)                       # (B, C, Y*(X//2+1))
    k_exp    = k_flat[None, None, :].expand(B, C_in, -1)      # (B, C, Y*(X//2+1))
    spectrum = torch.zeros(B, C_in, K, device=device)
    spectrum.scatter_add_(dim=2, index=k_exp, src=psd_flat)
    spectrum = spectrum / count[None, None, :]                 # normalise by bin count

    return spectrum.mean(dim=0)  # (C, K)


def compute_and_save_vrmse(
    checkpoint_name: str,
    date_start: str,
    date_end: str,
) -> None:
    r"""Load the model, run inference, compute and save the monthly VRMSE.

    Accumulates numerator and denominator across all days before taking the
    square root, producing the true monthly VRMSE rather than an average of
    per-batch VRMSEs.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
        date_start      : Start date of the subset, format 'YYYY-MM-DD'.
        date_end        : End date of the subset, format 'YYYY-MM-DD'.
    """
    dask.config.set(scheduler="synchronous")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_mask  = get_weights_state_mask(dim=1, device=device)  # (C, Y, X)
    w_depth = get_weights_mask(dim=1, device=device)        # (Z, Y, X)
    model   = s_load(PATH_MODELS / checkpoint_name, device=str(device)).eval()

    dataset = NeptuneDataset(date_start, date_end, standardized=True)
    loader  = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=device.type == "cuda")

    # Accumulate numerator and denominator separately for exact monthly VRMSE
    n_valid  = w_mask.sum(dim=(-2, -1)).clamp(min=1).cpu()  # (C,)
    num_acc  = torch.zeros(C)   # sum of masked squared errors
    den_acc  = torch.zeros(C)   # sum of masked squared deviations from mean
    mean_acc = torch.zeros(C)   # sum of masked spatial means (for u_bar over full month)
    n_total  = 0

    with torch.no_grad():
        for x, _ in loader:
            x    = x.to(device)
            x_in = torch.cat([x, w_depth.expand(x.shape[0], -1, -1, -1)], dim=1)
            _, x_hat = model(x_in)

            B = x.shape[0]
            mask_b = w_mask.unsqueeze(0)  # (1, C, Y, X)

            # Accumulate per-pixel sums (will normalise after full dataset pass)
            num_acc  += ((x - x_hat) ** 2 * mask_b).sum(dim=(0, -2, -1)).cpu()
            mean_acc += (x * mask_b).sum(dim=(0, -2, -1)).cpu()
            n_total  += B

    # Compute global u_bar over the full month
    u_bar = mean_acc / (n_total * n_valid)  # (C,)

    # Second pass for variance accumulation — no model inference needed
    u_bar_device = u_bar.to(device)[None, :, None, None]  # (1, C, 1, 1)
    with torch.no_grad():
        for x, _ in loader:
            x      = x.to(device)
            mask_b = w_mask.unsqueeze(0)
            den_acc += ((x - u_bar_device) ** 2 * mask_b).sum(dim=(0, -2, -1)).cpu()

    eps = 1e-6
    mse    = num_acc / (n_total * n_valid)
    var    = den_acc / (n_total * n_valid)
    result = torch.sqrt(mse / (var + eps)).numpy()

    out_dir = PATH_DIAGNOSTICS / checkpoint_name / "vrmse"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"vrmse-{date_start[:7]}.npy", result)


def compute_and_save_spectra(
    checkpoint_name: str,
    date_start: str,
    date_end: str,
) -> None:
    r"""Load the model, run inference, compute and save daily power spectra.

    Saves two files preserving daily resolution (no temporal averaging):
    one for the original data (GT) and one for the reconstruction.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
        date_start      : Start date of the subset, format 'YYYY-MM-DD'.
        date_end        : End date of the subset, format 'YYYY-MM-DD'.
    """
    dask.config.set(scheduler="synchronous")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_mask  = get_weights_state_mask(dim=1, device=device)  # (C, Y, X)
    w_depth = get_weights_mask(dim=1, device=device)        # (Z, Y, X)
    model   = s_load(PATH_MODELS / checkpoint_name, device=str(device)).eval()

    dataset = NeptuneDataset(date_start, date_end, standardized=True)
    loader  = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=device.type == "cuda")

    gt_list, rec_list = [], []

    with torch.no_grad():
        for x, _ in loader:
            x    = x.to(device)
            x_in = torch.cat([x, w_depth.expand(x.shape[0], -1, -1, -1)], dim=1)
            _, x_hat = model(x_in)

            # Compute per-sample spectra by processing batch sample by sample
            # to preserve daily resolution (shape: (B, C, K) per batch)
            for b in range(x.shape[0]):
                gt_list.append(power_spectrum(x[b:b+1],      w_mask).cpu().numpy())
                rec_list.append(power_spectrum(x_hat[b:b+1], w_mask).cpu().numpy())

    # Stack along day axis: (B_month, C, K)
    gt_arr  = np.stack(gt_list,  axis=0)
    rec_arr = np.stack(rec_list, axis=0)

    month   = date_start[:7]
    out_dir = PATH_DIAGNOSTICS / checkpoint_name / "spectra"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"spectre-gt-{month}.npy",  gt_arr)
    np.save(out_dir / f"spectre-rec-{month}.npy", rec_arr)
