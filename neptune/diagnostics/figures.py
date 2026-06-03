r"""Visualizations of autoencoders diagnostics."""

__all__ = [
    "visualize_error_vertical",
]

import matplotlib.pyplot as plt
import numpy as np
import re
import torch

from matplotlib.ticker import ScalarFormatter

from neptune.config import PATH_DIAGNOSTICS, PATH_EXP_AE_FIGURES
from neptune.data import (
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_OCEAN_BIO,
    DATASET_VARIABLES_OCEAN_PHY,
    DATASET_VARIABLES_SURFACE,
    Z,
)
from neptune.diagnostics import CMAPS_LINE, DEPTHS, TRANSLATIONS, UNITS


# fmt: off
#
class _SciFormatter(ScalarFormatter):
    r"""ScalarFormatter with bold exponent only (e.g. ×10^**-3**)."""

    def get_offset(self) -> str:
        return re.sub(r"\^\{([^}]+)\}", r"^{\\mathbf{\1}}", super().get_offset())


def _apply_sci_fmt(ax: plt.Axes) -> None:
    r"""Apply _SciFormatter to the x axis of ax."""
    fmt = _SciFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(fmt)


def _plot_surface(ax: plt.Axes, var: str, q25: float, q50: float, q75: float) -> None:
    r"""Horizontal bar 0→Q50 with IQR errorbar (Q25–Q75) at the right edge."""

    color = CMAPS_LINE[var]
    ax.barh(0, q50, color=color, alpha=0.80, height=0.4)
    ax.errorbar(q50, 0, xerr=[[q50 - q25], [q75 - q50]],
                fmt="none", color="black", capsize=6, linewidth=2, elinewidth=2)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", linestyle=":", alpha=0.75)
    _apply_sci_fmt(ax)


def _plot_volume(ax: plt.Axes, var: str, depths: np.ndarray, q25: np.ndarray, q50: np.ndarray, q75: np.ndarray) -> None:
    r"""Plot a volume variable as an error-vs-depth profile with Q25–Q75 shaded band."""

    color = CMAPS_LINE[var]
    ax.plot(q50, depths, color=color, linewidth=3, linestyle="-")
    ax.fill_betweenx(depths, q25, q75, color=color, alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(depths[-1] * 1.05, depths[0] * 0.9)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle=":", alpha=0.75)
    _apply_sci_fmt(ax)


def visualize_error_vertical(checkpoint_name: str) -> None:
    r"""Generate error-vs-depth figures for all variables (standardized and physical).

    Two PNG files are produced — one per unit system — each containing a 4×4 grid:

        Row 0 : surface variables  (windsp, tauuo, tauvo, ssh)          — horizontal bar plots
        Row 1 : physical ocean vars (uo, vo, votemper, vosaline)         — depth profiles
        Row 2 : BGC vars part 1    (CHL, DOX, PAR, PHO)                 — depth profiles
        Row 3 : BGC vars part 2    (SIO, NOS, hidden, hidden)           — depth profiles

    Arguments:
        checkpoint_name : Name of the model checkpoint (must have computed RMSE stats).
    """

    depths = np.array([float(DEPTHS[i]) for i in range(Z)])
    n_surf = len(DATASET_VARIABLES_SURFACE)

    # Load per-channel RMSE statistics for both unit systems
    rmse_dir  = PATH_DIAGNOSTICS / checkpoint_name / "rmse"
    stats_std = torch.load(rmse_dir / "rmse_standardized.pt", map_location="cpu", weights_only=True)
    stats_raw = torch.load(rmse_dir / "rmse_physical.pt",     map_location="cpu", weights_only=True)

    # Output directory
    save_dir = PATH_EXP_AE_FIGURES / checkpoint_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for stats, label, x_label, show_units in [
        (stats_std, "standardized", "Root Mean Square Error (standardized data)",     False),
        (stats_raw, "physical",     "Root Mean Square Error",                  True),
    ]:
        q25 = stats["q25"].numpy()
        q50 = stats["q50"].numpy()
        q75 = stats["q75"].numpy()

        fig, axes = plt.subplots(4, 4, figsize=(14, 18))
        fig.subplots_adjust(hspace=0.30, wspace=0.12)

        # ── Row 0: surface variables ─────────────────────────────────────────

        for col, var in enumerate(DATASET_VARIABLES_SURFACE):
            ax    = axes[0][col]
            title = f"{TRANSLATIONS[var]} ${UNITS[var]}$" if show_units else f"{TRANSLATIONS[var]} $[-]$"
            _plot_surface(ax, var, float(q25[col]), float(q50[col]), float(q75[col]))
            ax.set_title(title, fontweight="bold")
            if col > 0 and not show_units:
                axes[0][col].sharex(axes[0][0])
            if not show_units:
                ax.set_xlim(0, 0.08)
                ax.set_ylim(-1, 1)

        # ── Rows 1–3: volume variables ───────────────────────────────────────

        # Assemble all (row, col, var) assignments for volume subplots
        volume_cells = []
        for col, var in enumerate(DATASET_VARIABLES_OCEAN_PHY):
            volume_cells.append((1, col, var))
        for i, var in enumerate(DATASET_VARIABLES_OCEAN_BIO):
            volume_cells.append((2 + i // 4, i % 4, var))

        # Share Y axis across all volume subplots
        ref_ax = axes[volume_cells[0][0]][volume_cells[0][1]]
        for row, col, _ in volume_cells[1:]:
            axes[row][col].sharey(ref_ax)

        for row, col, var in volume_cells:
            ax        = axes[row][col]
            ocean_idx = DATASET_VARIABLES_OCEAN.index(var)
            c0        = n_surf + ocean_idx * Z
            title     = f"{TRANSLATIONS[var]} ${UNITS[var]}$" if show_units else f"{TRANSLATIONS[var]} $[-]$"
            _plot_volume(ax, var, depths, q25[c0:c0 + Z], q50[c0:c0 + Z], q75[c0:c0 + Z])
            ax.set_title(title, fontweight="bold")
            if not show_units:
                ax.set_xlim(0, 0.08 if var in ("votemper", "vosaline") else 0.5)
            if col == 0:
                ax.set_ylabel("Depth [m]")
            else:
                ax.tick_params(labelleft=False)

        # ── Single centered x-label at the bottom of the figure ─────────────

        fig.supxlabel(x_label, fontsize=18, y=0.07)

        # ── Hide unused subplots in last row ─────────────────────────────────

        axes[3][2].set_visible(False)
        axes[3][3].set_visible(False)

        # ── Save ──────────────────────────────────────────────────────────────

        fig.savefig(save_dir / f"error_vertical_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
