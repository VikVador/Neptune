r"""Neural network architecture and tools for latent diffusion."""

__all__ = [
    "LatentViT",
    "generate_forecast",
    "save",
    "load",
]

import torch

from azula.denoise import KarrasDenoiser
from azula.nn.dit import DiT
from azula.nn.layers import SineEncoding
from azula.nn.vit import ViT
from azula.noise import RectifiedSchedule
from azula.sample import zABSampler
from datetime import date as Date
from datetime import timedelta
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch import Tensor
from tqdm import tqdm


# fmt: off
#
class LatentViT(ViT):
    r"""ViT backbone for latent diffusion, conditioned on diffusion time and seasonal context.

    Arguments:
        lat_channels    : Number of latent channels C.
        cond_channels   : Number of conditioning channels.
        enc_features    : Number of encoding features for diffusion-time (must be even).
        hid_channels    : Hidden channel dimension of each transformer block.
        hid_blocks      : Number of transformer blocks.
        patch_size      : Spatial patch size for the tokenizer.
        attention_heads : Number of self-attention heads per block.
        ffn_factor      : Feed-forward channel scaling.
        dropout         : Dropout probability in attention and FFN.
        checkpointing   : Enable gradient checkpointing per block to save memory.
    """

    def __init__(
        self,
        lat_channels: int,
        cond_channels: int,
        enc_features: int = 32,
        hid_channels: int = 512,
        hid_blocks: int = 8,
        patch_size: int = 1,
        attention_heads: int = 2,
        ffn_factor: int = 4,
        dropout: float | None = None,
        checkpointing: bool = False,
    ) -> None:
        super().__init__(
            in_channels=lat_channels,
            out_channels=lat_channels,
            cond_channels=cond_channels,
            mod_features=enc_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            patch_size=patch_size,
            attention_heads=attention_heads,
            ffn_factor=ffn_factor,
            dropout=dropout,
            checkpointing=checkpointing,
            spatial=2,
        )

        # Number of conditioning channels
        self._cond_channels = cond_channels

        # Encoder for diffusion timestep
        self.time_enc = SineEncoding(enc_features)

    @staticmethod
    def day_of_year_to_conditioning(
        dates: list[str],
        H: int,
        W: int,
        device: torch.device,
    ) -> Tensor:
        r"""Convert date strings to a spatial progress-of-year conditioning tensor.

        Arguments:
            dates  : Date strings in 'YYYY-MM-DD' format (D).
            H      : Spatial height of the latent grid.
            W      : Spatial width of the latent grid.
            device : Target device.

        Returns:
            cond : Tensor of shape (B, 1, H, W) with values in [0, 1].
        """
        progress = []
        for d in dates:
            dt = Date.fromisoformat(d)
            p = (dt.timetuple().tm_yday - 1) / 365
            progress.append(p)

        cond = torch.tensor(progress, dtype=torch.float32, device=device)
        return cond.view(-1, 1, 1, 1).expand(-1, 1, H, W).contiguous()

    def forward(self, x: Tensor, mod: Tensor, cond: Tensor | list[Tensor] | None = None) -> Tensor:
        r"""Forward denoise a noisy state given diffusion time and optional conditioning.

        Arguments:
            x    : Noisy latent z (B, C, H, W).
            mod  : Modulation tensor (B,).
            cond : Conditioning tensor(s) (B, *, H, W).

        Returns:
            out : Denoised latent (B, C, H, W).
        """

        # Merge list of conditioning tensors into a single spatial tensor along channels
        if isinstance(cond, list):
            cond = torch.cat(cond, dim=1)

        # Ensure the total cond channels match what the model was built with
        if cond is not None and cond.shape[1] != self._cond_channels:
            raise ValueError(
                f"ERROR - cond has {cond.shape[1]} channels but LatentViT was built with "
                f"cond_channels={self._cond_channels}."
            )

        # Patchify (B, C, H, W) → (B, H', W', C·p²)
        x = self.patch(x)
        if cond is not None:
            cond = self.patch(cond).flatten(1, -2)

        # Build grid covering patch coordinates
        shape = x.shape[1:-1]
        axes  = [torch.arange(s, dtype=x.dtype, device=x.device) for s in shape]
        pos   = torch.cartesian_prod(*axes).reshape(-1, len(shape))

        # Flatten spatial dims to token sequence: (B, H', W', C·p²) → (B, L, C·p²)
        l = x.flatten(1, -2)

        # Calling diffusion transformer and restoring spatial layout
        l = DiT.forward(self, l, self.time_enc(mod), pos=pos, cond=cond)
        l = l.unflatten(-2, shape)
        x = self.unpatch(l)

        return x


def _build_rolling_window(
    cond: Tensor,
    n_members: int,
    device: torch.device,
) -> tuple[Tensor, int, int, int]:
    r"""Initialise the per-member autoregressive conditioning window on device.

    Arguments:
        cond      : Standardized latents (input_states, C, H, W) or (N, input_states, C, H, W).
        n_members : Number of ensemble members.
        device    : Target device.

    Returns:
        z_rolling : Tensor (N, input_states, C, H, W) on device.
        C_lat     : Latent channel count.
        H_lat     : Latent height.
        W_lat     : Latent width.
    """
    if cond.ndim == 4:
        _, C_lat, H_lat, W_lat = cond.shape
        z_rolling = cond.unsqueeze(0).expand(n_members, -1, -1, -1, -1).clone().to(device)
    elif cond.ndim == 5:
        N_cond, _, C_lat, H_lat, W_lat = cond.shape
        if N_cond != n_members:
            raise ValueError(f"ERROR - cond has {N_cond} members but n_members={n_members}.")
        z_rolling = cond.clone().to(device)
    else:
        raise ValueError(f"ERROR - cond must be 4D or 5D, got {cond.ndim}D.")
    return z_rolling, C_lat, H_lat, W_lat


@torch.no_grad()
def generate_forecast(
    backbone: LatentViT,
    cond: Tensor,
    date: str,
    n_members: int,
    n_days: int,
    n_steps: int = 64,
    alpha_min: float = 0.001,
    sigma_min: float = 0.001,
) -> Tensor:
    r"""Generate an autoregressive latent diffusion ensemble forecast.

    Arguments:
        backbone  : Trained LatentViT backbone.
        cond      : Standardized conditioning latents (input_states, C, H, W) or (N, input_states, C, H, W).
        date      : Date of the last conditioning state ('YYYY-MM-DD') to initialize the forecast.
        n_members : Number of ensemble members per forecast day.
        n_days    : Forecast horizon (autoregressive steps).
        n_steps   : Diffusion sampling steps per day.
        alpha_min : RectifiedSchedule alpha_min.
        sigma_min : RectifiedSchedule sigma_min.

    Returns:
        forecast : Standardized latent ensemble (T, N, C, H, W) on CPU.
    """

    # Retrieve device of diffusion backbone
    device = next(backbone.parameters()).device

    # Initialization of diffusion tools
    schedule = RectifiedSchedule(alpha_min=alpha_min, sigma_min=sigma_min)
    denoiser = KarrasDenoiser(backbone, schedule).to(device)
    sampler  = zABSampler(denoiser, order=3, steps=n_steps, silent=True)

    # Building conditioning window for autoregressive forecast
    z_rolling, C_lat, H_lat, W_lat = _build_rolling_window(cond, n_members, device)

    # Creating date tensors
    t0             = Date.fromisoformat(date)
    forecast_dates = [(t0 + timedelta(days=d + 1)).isoformat() for d in range(n_days)]

    # Stores the forecasted latents
    forecast = []

    for forecast_date in tqdm(forecast_dates, desc="Forecast", unit="day"):

        # Building progress-of-the-year conditioning
        year_cond = LatentViT.day_of_year_to_conditioning([forecast_date] * n_members, H_lat, W_lat, device)
        cond_t    = torch.cat([year_cond, z_rolling.flatten(1, 2)], dim=1)

        # Sampling
        z_new = sampler(torch.randn(n_members, C_lat, H_lat, W_lat, device=device), cond=cond_t)
        forecast.append(z_new.cpu())

        # Shift rolling window
        z_rolling = z_rolling.roll(-1, dims=1)
        z_rolling[:, -1] = z_new

        # Cleanup
        del year_cond, cond_t, z_new
        torch.cuda.empty_cache()

    # Return forecast tensor (T, N, C, H, W)
    return torch.stack(forecast, dim=0).cpu()


def save(backbone: LatentViT, config: DictConfig, path: Path | str,) -> None:
    r"""Save a diffusion backbone and its training configuration.

    Arguments:
        backbone : LatentViT backbone to save.
        config   : OmegaConf config with architecture and schedule parameters.
        path     : Directory where config.yaml and model.pth are written.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path / "config.yaml")
    torch.save(backbone.state_dict(), path / "model.pth")


def load(path: Path | str, device: str = "cpu",) -> LatentViT:
    r"""Load a diffusion backbone from a checkpoint directory.

    Arguments:
        path   : Directory containing config.yaml and model.pth.
        device : Device to map the loaded weights to.

    Returns:
        backbone : Reconstructed LatentViT in eval mode.
    """
    path   = Path(path)
    config = OmegaConf.load(path / "config.yaml")

    backbone = LatentViT(
        lat_channels    = config.lat_channels,
        cond_channels   = config.cond_channels,
        enc_features    = config.enc_features,
        hid_channels    = config.hid_channels,
        hid_blocks      = config.hid_blocks,
        patch_size      = config.patch_size,
        attention_heads = config.attention_heads,
        ffn_factor      = config.ffn_factor,
        dropout         = config.dropout,
        checkpointing   = config.checkpointing,
    )

    state = torch.load(path / "model.pth", map_location=device, weights_only=True)
    backbone.load_state_dict(state)
    return backbone.to(device).eval()
