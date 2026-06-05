r"""Metrics for diagnosing autoencoders."""

__all__ = [
    "power_spectrum",
    "compute_and_save_power_spectra",
    "compute_and_save_se",
    "compute_and_save_stats_mse",
    "compute_and_save_maps",
    "clean_se",
    "compute_and_save_reconstructions",
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
def power_spectrum(
    u: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
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
        if mask.shape != (1, C_in, Y_in, X_in):
            raise ValueError(f"Expected mask shape (1, {C_in}, {Y_in}, {X_in}), got {mask.shape}")
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


def compute_and_save_power_spectra(
    checkpoint_name: str,
    date_start: str,
    date_end: str,
) -> None:
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

            # Compute power spectra with mask applied (land → 0.0)
            gt_list.append( power_spectrum(x,     mask=w_state_mask).cpu())
            rec_list.append(power_spectrum(x_hat, mask=w_state_mask).cpu())

    # Saving results
    save_dir = PATH_DIAGNOSTICS / checkpoint_name / "power_spectra"
    save_dir.mkdir(parents=True, exist_ok=True)
    data = {"ground_truth": torch.cat(gt_list, dim=0), "reconstruction": torch.cat(rec_list, dim=0)}
    torch.save(data, save_dir / f"power_spectra_{date_start}_{date_end}.pt")


def compute_and_save_se(
    checkpoint_name: str,
    date_start: str,
    date_end: str,
) -> None:
    r"""Load the model, run inference, compute and save per-pixel squared errors.

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

    se_std_list, se_raw_list = [], []
    with torch.no_grad():
        for (x_std, _), (x_raw, _) in zip(dataloader_std, dataloader_raw, strict=False):

            # Pushing to device and concatenating mask
            x_std = x_std.to(device)
            x_raw = x_raw.to(device)
            x_in  = torch.cat([x_std, w_mask.expand(x_std.shape[0], -1, -1, -1)], dim=1)

            # Forward pass
            _, x_hat = model(x_in)

            # Applying mask on reconstructions
            x_hat_m = x_hat.masked_fill(w_state_mask.expand_as(x_hat) == 0, float("nan"))

            # Computing standardized per-pixel SE
            x_std_m = x_std.masked_fill(w_state_mask.expand_as(x_std) == 0, float("nan"))
            se_std_list.append((x_std_m - x_hat_m).pow(2).cpu())

            # Computing physical per-pixel SE
            x_hat_raw = dataset_std.unstandardize(x_hat_m)
            x_raw_m   = x_raw.masked_fill(w_state_mask.expand_as(x_raw) == 0, float("nan"))
            se_raw_list.append((x_raw_m - x_hat_raw).pow(2).cpu())

    # Saving results
    save_dir = PATH_DIAGNOSTICS / checkpoint_name / "se"
    save_dir.mkdir(parents=True, exist_ok=True)
    data = {"se_standardized": torch.cat(se_std_list, dim=0), "se_physical": torch.cat(se_raw_list, dim=0)}
    torch.save(data, save_dir / f"se_{date_start}_{date_end}.pt")


def compute_and_save_stats_mse(checkpoint_name: str) -> None:
    r"""Load per-pixel SE files one by one and compute per-channel error statistics.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
    """

    se_paths = sorted((PATH_DIAGNOSTICS / checkpoint_name / "se").glob("*.pt"))
    mse_std_list, mse_raw_list = [], []
    for path in se_paths:
        d = torch.load(path, map_location="cpu")
        mse_std_list.append(torch.nanmean(d["se_standardized"], dim=(2, 3)))  # (B, C)
        mse_raw_list.append(torch.nanmean(d["se_physical"],     dim=(2, 3)))  # (B, C)

    def _stats(mse: Tensor) -> dict:
        r = mse.sqrt()
        return {
            "rmse": mse.mean(0).sqrt(),
            "std" : r.std(0),
            "q25" : r.quantile(0.25, dim=0),
            "q50" : r.quantile(0.50, dim=0),
            "q75" : r.quantile(0.75, dim=0),
        }

    save_dir = PATH_DIAGNOSTICS / checkpoint_name / "rmse"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_stats(torch.cat(mse_std_list, dim=0)), save_dir / "rmse_standardized.pt")
    torch.save(_stats(torch.cat(mse_raw_list, dim=0)), save_dir / "rmse_physical.pt")


def compute_and_save_maps(checkpoint_name: str) -> None:
    r"""Compute per-pixel error and std maps using an online single-pass algorithm.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
    """

    se_paths = sorted((PATH_DIAGNOSTICS / checkpoint_name / "se").glob("*.pt"))

    # Online accumulators: sum(SE), sum(SE²), count of valid (ocean) samples per pixel
    sum_std,    sum_std_sq    = None, None
    sum_raw,    sum_raw_sq    = None, None
    count = None

    for path in se_paths:
        d      = torch.load(path, map_location="cpu")
        se_std = d["se_standardized"]   # (B, C, Y, X)
        se_raw = d["se_physical"]       # (B, C, Y, X)

        valid      = (~torch.isnan(se_std)).sum(0).float()   # (C, Y, X)
        se_std_0   = se_std.nan_to_num(0.0)
        se_raw_0   = se_raw.nan_to_num(0.0)

        if sum_std is None:
            sum_std    = se_std_0.sum(0)
            sum_std_sq = se_std_0.pow(2).sum(0)
            sum_raw    = se_raw_0.sum(0)
            sum_raw_sq = se_raw_0.pow(2).sum(0)
            count      = valid
        else:
            sum_std    += se_std_0.sum(0)
            sum_std_sq += se_std_0.pow(2).sum(0)
            sum_raw    += se_raw_0.sum(0)
            sum_raw_sq += se_raw_0.pow(2).sum(0)
            count      += valid

    n        = count.clamp(min=1)
    land_nan = count == 0

    def _maps(s: Tensor, s_sq: Tensor) -> tuple[Tensor, Tensor]:
        r"""Compute online mean and std maps from accumulators."""
        mean_se = s / n
        var_se  = (s_sq / n - mean_se.pow(2)).clamp(min=0)
        rmse    = mean_se.clamp(min=0).sqrt()
        std     = var_se.sqrt()
        rmse[land_nan] = float("nan")
        std[land_nan]  = float("nan")
        return rmse, std

    rmse_std, std_std = _maps(sum_std, sum_std_sq)
    rmse_raw, std_raw = _maps(sum_raw, sum_raw_sq)

    save_dir = PATH_DIAGNOSTICS / checkpoint_name / "maps"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"rmse_standardized": rmse_std, "std_standardized" : std_std, "rmse_physical" : rmse_raw, "std_physical" : std_raw}, save_dir / "maps.pt")


def clean_se(checkpoint_name: str) -> None:
    r"""Delete all per-pixel SE files to free disk space after aggregation.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
    """
    for path in (PATH_DIAGNOSTICS / checkpoint_name / "se").glob("*.pt"):
        path.unlink()


def compute_and_save_reconstructions(
    checkpoint_name: str,
    dates: list[str],
) -> None:
    r"""Load the model, run inference, and save ground truths, reconstructions and dates.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
        dates           : List of dates (format 'YYYY-MM-DD') to reconstruct.
    """

    device, w_mask, w_state_mask, model = _setup(checkpoint_name)

    # Build dataset over the full range, then restrict to requested dates
    dates_sorted  = sorted(dates)
    dataset       = NeptuneDataset(dates_sorted[0], dates_sorted[-1], standardized=True)
    dates_set     = set(dates)
    dataset.dates = [d for d in dataset.dates if d in dates_set]

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=1,
        pin_memory=device.type == "cuda",
    )

    gt_list, rec_list, dates_list = [], [], []
    with torch.no_grad():
        for x, batch_dates in dataloader:

            # Pushing to device and concatenating mask
            x    = x.to(device)
            x_in = torch.cat([x, w_mask.expand(x.shape[0], -1, -1, -1)], dim=1)

            # Forward pass
            _, x_hat = model(x_in)
            x_m      = x.masked_fill(    w_state_mask.expand_as(x)     == 0, float("nan"))
            x_hat_m  = x_hat.masked_fill(w_state_mask.expand_as(x_hat) == 0, float("nan"))

            gt_list.append(x_m.cpu())
            rec_list.append(x_hat_m.cpu())
            dates_list.extend(list(batch_dates))

    # Saving results
    save_dir = PATH_DIAGNOSTICS / checkpoint_name / "reconstructions"
    save_dir.mkdir(parents=True, exist_ok=True)
    data = {"ground_truths": torch.cat(gt_list,  dim=0), "reconstructions": torch.cat(rec_list, dim=0), "dates": dates_list}
    torch.save(data, save_dir / f"reconstructions_{dates_sorted[0]}_{dates_sorted[-1]}.pt")
