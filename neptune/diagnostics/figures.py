r"""Figures and animations of forecasts."""

__all__ = [
    "animate_forecast",
]

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from PIL import Image

from neptune.diagnostics import CMAPS_FIELD, TRANSLATIONS, UNITS
from neptune.diagnostics.geography import draw_geography, load_grid


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    r"""Compute the root mean squared error between two fields, ignoring NaNs (land).

    Arguments:
        x : Forecasted field (Y, X).
        y : Ground truth field (Y, X).

    Returns:
        rmse : Root mean squared error over the sea.
    """
    return float(np.sqrt(np.nanmean((x - y) ** 2)))


def animate_forecast(
    truth: np.ndarray,
    forecast: np.ndarray,
    initial: np.ndarray,
    variable: str,
    date: str,
    path: Path,
    duration: float = 0.5,
) -> None:
    r"""Animate an ensemble forecast of a 2D field against the ground truth, one frame per day.

    The ground truth and the ensemble mean are shown on top, and up to 4 members below. The RMSE
    over the sea of each forecast is displayed in the upper right corner of its map, in red when it
    is larger than the one of the ensemble mean. Every map shares the colors, bounded by the 5% and
    95% quantiles of the ground truth. On the top right, the improvement of the ensemble mean over
    persistence, i.e. the last input state kept as forecast, is shown for each day: the relative
    decrease of RMSE [%], in green when the ensemble mean is better and in red when it is worse.

    Arguments:
        truth    : Ground truth in physical units, NaN on land (K, Y, X).
        forecast : Forecasted members in physical units, NaN on land (E, K, Y, X).
        initial  : Last input state x_t in physical units, NaN on land (Y, X).
        variable : Name of the variable, e.g. 'votemper'.
        date     : Date of the last input state 'YYYY-MM-DD', the forecast starting the day after.
        path     : Path to the output .gif file.
        duration : Time between two frames (days) [s].
    """

    days = len(truth)
    members = min(len(forecast), 4)
    mean = forecast.mean(axis=0)
    vmin, vmax = np.nanquantile(truth, [0.05, 0.95])
    lon, lat = load_grid()

    # Persistence | Relative decrease of RMSE from persistence to the ensemble mean, per day [%]
    rmse_persistence = np.array([_rmse(initial, truth[k]) for k in range(days)])
    rmse_mean = np.array([_rmse(mean[k], truth[k]) for k in range(days)])
    improvement = 100 * (rmse_persistence - rmse_mean) / np.maximum(rmse_persistence, 1e-12)

    # Figure | Ground truth, ensemble mean and persistence on top, members below, shared colorbar
    fig = plt.figure(figsize=(16, 6.5), facecolor="white")
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.12], hspace=0.3, wspace=0.15)
    panels = [("Ground Truth", grid[0, 0]), ("Ensemble Mean", grid[0, 1])]
    panels += [(f"Member {m + 1}", grid[1, m]) for m in range(members)]

    meshes, texts = [], []
    for title, cell in panels:
        ax = fig.add_subplot(cell, projection=ccrs.PlateCarree())
        ax.set_facecolor("lightgray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        draw_geography(ax)

        meshes.append(
            ax.pcolormesh(
                lon,
                lat,
                truth[0],
                cmap=CMAPS_FIELD[variable],
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
        )

        texts.append(
            ax.text(
                0.98,
                0.97,
                "",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1),
                zorder=4,
            )
        )

    colorbar = fig.colorbar(meshes[0], cax=fig.add_subplot(grid[2, :]), orientation="horizontal")
    colorbar.ax.tick_params(labelsize=12)
    texts[0].set_visible(False)

    # Persistence | Same size as the maps, improvement written on each bar instead of a y-axis
    ax_persistence = fig.add_subplot(grid[0, 2])
    ax_persistence.set_box_aspect((lat.max() - lat.min()) / (lon.max() - lon.min()))
    bars = ax_persistence.bar(
        np.arange(1, days + 1),
        improvement,
        color=np.where(improvement >= 0, "tab:green", "tab:red"),
    )
    ax_persistence.bar_label(bars, fmt="%+.0f%%", fontsize=6, padding=1)
    ax_persistence.axhline(0, color="black", linewidth=0.8)
    limit = 1.3 * max(np.abs(improvement).max(), 1.0)
    ax_persistence.set_ylim(-limit, limit)
    ax_persistence.set_title("Improvement over Persistence", fontsize=11, fontweight="bold")
    ax_persistence.set_xticks(np.arange(1, days + 1))
    ax_persistence.tick_params(axis="x", labelsize=7)
    ax_persistence.set_yticks([])
    ax_persistence.spines[["left", "top", "right", "bottom"]].set_visible(False)

    # Frames | One per forecasted day, in RGB since GIF transparency would erase the black pixels
    frames = []
    for k in range(days):
        fields = [truth[k], mean[k], *forecast[:members, k]]

        for mesh, text, field in zip(meshes, texts, fields, strict=True):
            rmse = _rmse(field, truth[k])
            mesh.set_array(np.ma.masked_invalid(field))
            text.set_text(f"RMSE = {rmse:.3f}")
            text.set_color("red" if rmse > rmse_mean[k] else "black")

        # Persistence | The bar of the current day is outlined
        for day, bar in enumerate(bars):
            bar.set_edgecolor("black" if day == k else "none")
            bar.set_linewidth(2)

        fig.suptitle(
            f"{TRANSLATIONS[variable]} ${UNITS[variable]}$ — {date} + {k + 1} day(s)",
            fontsize=14,
            fontweight="bold",
        )

        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))

    plt.close(fig)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 * duration),
        loop=0,
    )
