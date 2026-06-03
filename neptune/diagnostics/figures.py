r"""Visualizations of autoencoders diagnostics."""

__all__ = [
    "scale_fig_properties",
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
from neptune.diagnostics import (
    CMAPS_LINE,
    DEPTHS,
    FIG_PROPERTIES,
    TRANSLATIONS,
    UNITS,
)

# fmt: off
#
# ================================
#
#          LOCAL CONSTANTS
#
# ================================
#
# ========================
# visualize_error_vertical
# ========================
_FIGSIZE_A        = (14, 16)
_XLIM_STD_SURFACE = 0.08
_XLIM_STD_VOLUME  = 0.50


class _SciFormatter(ScalarFormatter):
    r"""ScalarFormatter with bold exponent only."""
    def get_offset(self) -> str:
        return re.sub(r"\^\{([^}]+)\}", r"^{\\mathbf{\1}}", super().get_offset())

def _apply_sci_fmt(ax: plt.Axes) -> None:
    r"""Apply _SciFormatter to the x axis of ax."""
    fmt = _SciFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(fmt)


def scale_fig_properties(figsize: tuple[float, float]) -> dict:
    r"""Return properties with font sizes and linewidth scaled to figsize.

    Arguments:
        figsize : Target figure size (width, height) in inches.

    Returns:
        props : Copy of FIG_PROPERTIES with numerical sizes scaled.
    """
    ref_w, ref_h = FIG_PROPERTIES["figsize"]
    scale = ((figsize[0] * figsize[1]) / (ref_w * ref_h)) ** 0.5
    props = dict(FIG_PROPERTIES)
    for key in ("fs_title", "fs_x_label", "fs_y_label", "fs_x_tick", "fs_y_tick", "fs_sup_x_label", "fs_sup_y_label", "line_width"):
        props[key] = FIG_PROPERTIES[key] * scale
    return props


def _plot_surface(ax: plt.Axes, var: str, q25: float, q50: float, q75: float, props: dict) -> None:
    r"""Horizontal bar 0→Q50 with IQR errorbar (Q25–Q75) at the right edge."""

    color = CMAPS_LINE[var]
    ax.barh(0, q50, color=color, alpha=props["line_opacity"], height=0.4)
    ax.errorbar(q50, 0, xerr=[[q50 - q25], [q75 - q50]], fmt="none", color="black", capsize=6, linewidth=2, elinewidth=2)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", linestyle=":", alpha=0.75)
    ax.tick_params(axis="x", labelsize=props["fs_x_tick"])
    ax.tick_params(axis="y", labelsize=props["fs_y_tick"])
    _apply_sci_fmt(ax)


def _plot_volume(ax: plt.Axes, var: str, depths: np.ndarray, q25: np.ndarray, q50: np.ndarray, q75: np.ndarray, props: dict) -> None:
    r"""Plot a volume variable as an error-vs-depth profile with Q25–Q75 shaded band."""

    color = CMAPS_LINE[var]
    ax.plot(q50, depths, color=color, linewidth=props["line_width"], linestyle="-")
    ax.fill_betweenx(depths, q25, q75, color=color, alpha=props["line_opacity_fill_between"])
    ax.set_yscale("log")
    ax.set_ylim(depths[-1] * 1.05, depths[0] * 0.9)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle=":", alpha=0.75)
    ax.tick_params(axis="x", labelsize=props["fs_x_tick"])
    ax.tick_params(axis="y", labelsize=props["fs_y_tick"])
    _apply_sci_fmt(ax)


def visualize_error_vertical(checkpoint_name: str) -> None:
    r"""Generate error-vs-depth figures for all variables (standardized and physical).

    Arguments:
        checkpoint_name : Name of the model checkpoint (must have computed RMSE stats).
    """

    props  = scale_fig_properties(_FIGSIZE_A)
    depths = np.array([float(DEPTHS[i]) for i in range(Z)])
    n_surf = len(DATASET_VARIABLES_SURFACE)

    # Load per-channel RMSE statistics for both unit systems
    rmse_dir  = PATH_DIAGNOSTICS / checkpoint_name / "rmse"
    stats_std = torch.load(rmse_dir / "rmse_standardized.pt", map_location="cpu", weights_only=True)
    stats_raw = torch.load(rmse_dir / "rmse_physical.pt",     map_location="cpu", weights_only=True)

    # Output directory
    save_dir = PATH_EXP_AE_FIGURES / checkpoint_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for stats, label, x_label in [
        (stats_std, "standardized", "Root Mean Square Error (standardized data)"),
        (stats_raw, "physical",     "Root Mean Square Error"),
    ]:
        show_units = label == "physical"

        q25 = stats["q25"].numpy()
        q50 = stats["q50"].numpy()
        q75 = stats["q75"].numpy()

        fig, axes = plt.subplots(4, 4, figsize=_FIGSIZE_A)
        fig.subplots_adjust(hspace=0.30, wspace=0.12)

        # First row: surface variables
        for col, var in enumerate(DATASET_VARIABLES_SURFACE):
            ax    = axes[0][col]
            title = f"{TRANSLATIONS[var]} ${UNITS[var]}$" if show_units else f"{TRANSLATIONS[var]} $[-]$"
            _plot_surface(ax, var, float(q25[col]), float(q50[col]), float(q75[col]), props)
            ax.set_title(title, fontweight="bold", fontsize=props["fs_title"])
            if not show_units:
                ax.set_xlim(0, _XLIM_STD_SURFACE)

        # Rows 1-3: volume variables
        volume_cells = (
            [(1, col, var) for col, var in enumerate(DATASET_VARIABLES_OCEAN_PHY)] +
            [(2 + i // 4, i % 4, var) for i, var in enumerate(DATASET_VARIABLES_OCEAN_BIO)]
        )

        # Share Y axis across all volume subplots
        ref_ax = axes[volume_cells[0][0]][volume_cells[0][1]]
        for row, col, _ in volume_cells[1:]:
            axes[row][col].sharey(ref_ax)

        for row, col, var in volume_cells:
            ax        = axes[row][col]
            ocean_idx = DATASET_VARIABLES_OCEAN.index(var)
            c0        = n_surf + ocean_idx * Z
            title     = f"{TRANSLATIONS[var]} ${UNITS[var]}$" if show_units else f"{TRANSLATIONS[var]} $[-]$"
            _plot_volume(ax, var, depths, q25[c0:c0 + Z], q50[c0:c0 + Z], q75[c0:c0 + Z], props)
            ax.set_title(title, fontweight="bold", fontsize=props["fs_title"])
            if not show_units:
                ax.set_xlim(0, _XLIM_STD_SURFACE if var in ("votemper", "vosaline") else _XLIM_STD_VOLUME)
            if col == 0:
                ax.set_ylabel("Depth [m]", fontsize=props["fs_y_label"])
            else:
                ax.tick_params(labelleft=False)

        # Single centered x-label
        fig.supxlabel(x_label, fontsize=props["fs_sup_x_label"], y=0.06)

        # Hide unused subplots in last row
        axes[3][2].set_visible(False)
        axes[3][3].set_visible(False)

        # Save
        fig.savefig(save_dir / f"error_vertical_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
