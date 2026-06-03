r"""Visualizations of autoencoders diagnostics."""

__all__ = [
    "scale_fig_properties",
    "visualize_error_vertical",
    "visualize_spectra",
]

import matplotlib.pyplot as plt
import numpy as np
import re
import torch

from matplotlib.ticker import ScalarFormatter

from neptune.config import (
    PATH_DIAGNOSTICS,
    PATH_EXP_AE_FIGURES,
)
from neptune.data import (
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_OCEAN_BIO,
    DATASET_VARIABLES_OCEAN_PHY,
    DATASET_VARIABLES_SURFACE,
    X,
    Y,
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
#          LOCAL CONSTANTS
# ================================
#
# visualize_error_vertical
_FIGSIZE_A        = (14, 16)
_XLIM_STD_SURFACE = 0.08
_XLIM_STD_VOLUME  = 0.50

# visualize_spectra
_FIGSIZE_B           = (14, 16)
_GT_COLOR            = "#808080"                     # Ground truth line color
_DX_KM               = 2.8                             # Simulation resolution [km/pixel] (0.025°)
_YLIM_SURFACE        = (1e-4, 5e4)                     # Y-axis limits for surface variables
_YLIM_OCEAN          = (1e-4, 1e4)                     # Y-axis limits for ocean variables
_SPECTRA_DEPTHS      = [0, 25, 41]                     # Depths at which to show spectra
_SPECTRA_WAVELENGTHS = [300, 80, 20]                   # Wavelengths at which to show errorbars
_LINESTYLES          = ["dashed", "dotted", "dashdot"] # Spectrum line styles


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
    """
    ref_w, ref_h = FIG_PROPERTIES["figsize"]
    scale = ((figsize[0] * figsize[1]) / (ref_w * ref_h)) ** 0.5
    props = dict(FIG_PROPERTIES)
    for key in ("fs_title", "fs_x_label", "fs_y_label", "fs_x_tick", "fs_y_tick", "fs_sup_x_label", "fs_sup_y_label", "line_width"):
        props[key] = FIG_PROPERTIES[key] * scale
    return props


def _plot_surface(ax: plt.Axes, var: str, q25: float, q50: float, q75: float, props: dict) -> None:
    r"""Create a horizontal bar plot for surface variables."""
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
    r"""Create an error-vs-depth plot for global variables."""
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


def _plot_spectrum(
    ax: plt.Axes,
    var: str,
    wavelengths: np.ndarray,
    gt_list:      list[np.ndarray],
    rec_list:     list[np.ndarray],
    gt_q25_list:  list[np.ndarray],
    gt_q75_list:  list[np.ndarray],
    rec_q25_list: list[np.ndarray],
    rec_q75_list: list[np.ndarray],
    props: dict,
) -> None:
    r"""Plot ground truth and reconstruction spectra at one or more depth levels."""

    color = CMAPS_LINE[var]
    for i, (gt, rec, gt_lo, gt_hi, rec_lo, rec_hi) in enumerate(zip(gt_list, rec_list, gt_q25_list, gt_q75_list, rec_q25_list, rec_q75_list, strict=False)):
        ls = _LINESTYLES[i % len(_LINESTYLES)]
        ax.plot(wavelengths, gt,  color=_GT_COLOR, linewidth=props["line_width"], linestyle=ls)
        ax.plot(wavelengths, rec, color=color,     linewidth=props["line_width"], linestyle=ls)
        for w in _SPECTRA_WAVELENGTHS:
            idx = int(np.argmin(np.abs(wavelengths - w)))
            ax.errorbar(wavelengths[idx], gt[idx],  yerr=[[gt[idx] - gt_lo[idx]], [gt_hi[idx] - gt[idx]]],    fmt="none", color=_GT_COLOR, capsize=5, linewidth=2, elinewidth=2)
            ax.errorbar(wavelengths[idx], rec[idx], yerr=[[rec[idx] - rec_lo[idx]], [rec_hi[idx] - rec[idx]]],fmt="none", color=color,     capsize=5, linewidth=2, elinewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.75)
    ax.tick_params(axis="x", labelsize=props["fs_x_tick"])
    ax.tick_params(axis="y", labelsize=props["fs_y_tick"])


def visualize_error_vertical(checkpoint_name: str) -> None:
    r"""Generate error-vs-depth figures for all variables (standardized and physical).

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
    """

    props  = scale_fig_properties(_FIGSIZE_A)
    depths = np.array([float(DEPTHS[i]) for i in range(Z)])
    n_surf = len(DATASET_VARIABLES_SURFACE)

    # Load per-channel statistics for both unit systems
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


def visualize_spectra(checkpoint_name: str) -> None:
    r"""Generate power-spectrum figure for all variables.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
    """

    props  = scale_fig_properties(_FIGSIZE_B)
    n_surf = len(DATASET_VARIABLES_SURFACE)

    # Load most recent spectra file and compute quantiles over time
    spectra_dir = PATH_DIAGNOSTICS / checkpoint_name / "power_spectra"
    data        = torch.load(sorted(spectra_dir.glob("*.pt"))[-1], map_location="cpu", weights_only=True)
    gt_q50  = data["ground_truth"].quantile(0.50, dim=0)   # (C, K)
    gt_q25  = data["ground_truth"].quantile(0.25, dim=0)   # (C, K)
    gt_q75  = data["ground_truth"].quantile(0.75, dim=0)   # (C, K)
    rec_q50 = data["reconstruction"].quantile(0.50, dim=0) # (C, K)
    rec_q25 = data["reconstruction"].quantile(0.25, dim=0) # (C, K)
    rec_q75 = data["reconstruction"].quantile(0.75, dim=0) # (C, K)

    K           = gt_q50.shape[-1]
    wavelengths = min(X, Y) * _DX_KM / np.arange(1, K)

    # Output directory
    save_dir = PATH_EXP_AE_FIGURES / checkpoint_name
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=_FIGSIZE_B)
    fig.subplots_adjust(hspace=0.30, wspace=0.12)

    # First row: surface variables
    for col, var in enumerate(DATASET_VARIABLES_SURFACE):
        ax     = axes[0][col]
        c      = col
        gt     = gt_q50[c, 1:].numpy()
        rec    = rec_q50[c, 1:].numpy()
        gt_lo  = gt_q25[c, 1:].numpy()
        gt_hi  = gt_q75[c, 1:].numpy()
        rec_lo = rec_q25[c, 1:].numpy()
        rec_hi = rec_q75[c, 1:].numpy()
        _plot_spectrum(ax, var, wavelengths, [gt], [rec], [gt_lo], [gt_hi], [rec_lo], [rec_hi], props)
        ax.set_ylim(_YLIM_SURFACE)
        ax.set_title(TRANSLATIONS[var], fontweight="bold", fontsize=props["fs_title"])
        if col == 0:
            ax.set_ylabel("Power Spectral Density", fontsize=props["fs_y_label"])
        else:
            ax.tick_params(labelleft=False)

    # Rows 1-3: volume variables
    volume_cells = (
        [(1, col, var) for col, var in enumerate(DATASET_VARIABLES_OCEAN_PHY)] +
        [(2 + i // 4, i % 4, var) for i, var in enumerate(DATASET_VARIABLES_OCEAN_BIO)]
    )

    for row, col, var in volume_cells:
        ax           = axes[row][col]
        ocean_idx    = DATASET_VARIABLES_OCEAN.index(var)
        cs           = [n_surf + ocean_idx * Z + d for d in _SPECTRA_DEPTHS]
        gt_list      = [gt_q50 [c, 1:].numpy() for c in cs]
        rec_list     = [rec_q50[c, 1:].numpy() for c in cs]
        gt_q25_list  = [gt_q25 [c, 1:].numpy() for c in cs]
        gt_q75_list  = [gt_q75 [c, 1:].numpy() for c in cs]
        rec_q25_list = [rec_q25[c, 1:].numpy() for c in cs]
        rec_q75_list = [rec_q75[c, 1:].numpy() for c in cs]
        _plot_spectrum(ax, var, wavelengths, gt_list, rec_list, gt_q25_list, gt_q75_list, rec_q25_list, rec_q75_list, props)
        ax.set_ylim(_YLIM_OCEAN)
        ax.set_title(TRANSLATIONS[var], fontweight="bold", fontsize=props["fs_title"])
        if col == 0:
            ax.set_ylabel("Power Spectral Density", fontsize=props["fs_y_label"])
        else:
            ax.tick_params(labelleft=False)

    # Single centered x-label
    fig.supxlabel(r"Wavelength $\lambda$ [km]", fontsize=props["fs_sup_x_label"], y=0.06)

    # Hide unused subplots in last row
    axes[3][2].set_visible(False)
    axes[3][3].set_visible(False)

    # Save
    fig.savefig(save_dir / "power_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
