r"""Visualizations of autoencoders diagnostics."""

__all__ = [
    "scale_fig_properties",
    "visualize_error_vertical",
    "visualize_error_maps_z",
    "visualize_error_maps_x",
    "visualize_error_maps_y",
    "visualize_spectra",
    "visualize_reconstructions",
]

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import re
import torch

from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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
from neptune.data.weights import get_weights_stats
from neptune.diagnostics import (
    CMAPS_FIELD,
    CMAPS_LINE,
    CMAPS_METRICS,
    DEPTHS,
    FIG_PROPERTIES,
    TRANSLATIONS,
    UNITS,
)
from neptune.diagnostics.geography import draw_geography, load_grid

# fmt: off
#
# =======================
#    LOCAL CONSTANTS
# =======================
#
_UNIT_FILE        = {"standardized": "scaled", "physical": "unscaled"}
_VARS_PHY         = DATASET_VARIABLES_SURFACE + DATASET_VARIABLES_OCEAN_PHY
_VARS_PHY_SECTION = DATASET_VARIABLES_OCEAN_PHY
_VARS_BIO         = DATASET_VARIABLES_OCEAN_BIO
_LON_TICKS        = [28, 32, 36, 40]
_LAT_TICKS        = [42, 44, 46]

# visualize_error_vertical
_FIGSIZE_VERT     = (14, 16)
_XLIM_STD_SURFACE = 0.08
_XLIM_STD_VOLUME  = 0.50

# visualize_error_maps_z / _x / _y
_FIGSIZE_MAP_Z    = (13, 9)
_FIGSIZE_MAP_XY   = (13, 7)
_STAT_LABELS      = {"rmse": "Root Mean Squared Error", "std": "Standard Deviation"}

# visualize_reconstructions
_FIGSIZE_RECON    = (13, 7.5)
_RECON_DEPTHS     = [0, 16, 32]

# visualize_spectra
_FIGSIZE_SPEC          = (14, 16)
_GT_COLOR              = "#808080"
_DX_KM                 = 2.8
_YLIM_SURFACE          = (1e-4, 5e4)
_YLIM_OCEAN            = (1e-4, 1e4)
_SPECTRA_DEPTHS        = [0, 25, Z - 1]
_SPECTRA_WAVELENGTHS   = [300, 80, 20]
_LINESTYLES            = ["dashed", "dotted", "dashdot"]


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
    r"""Return properties with font sizes and linewidth scaled to figsize."""
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
    r"""Plot ground truth and reconstruction spectra."""

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
    r"""Generate error-vs-depth figures of variables.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
    """

    props  = scale_fig_properties(_FIGSIZE_VERT)
    depths = np.array([float(DEPTHS[i]) for i in range(Z)])
    n_surf = len(DATASET_VARIABLES_SURFACE)

    # Load per-channel statistics for both unit systems
    rmse_dir  = PATH_DIAGNOSTICS / checkpoint_name / "rmse"
    stats_std = torch.load(rmse_dir / "rmse_standardized.pt", map_location="cpu", weights_only=True)
    stats_raw = torch.load(rmse_dir / "rmse_physical.pt",     map_location="cpu", weights_only=True)

    # Output directory
    save_dir = PATH_EXP_AE_FIGURES / checkpoint_name / "global"
    save_dir.mkdir(parents=True, exist_ok=True)

    for stats, label, x_label in [
        (stats_std, "standardized", "Root Mean Square Error (standardized data)"),
        (stats_raw, "physical",     "Root Mean Square Error"),
    ]:
        show_units = label == "physical"

        q25 = stats["q25"].numpy()
        q50 = stats["q50"].numpy()
        q75 = stats["q75"].numpy()

        fig, axes = plt.subplots(4, 4, figsize=_FIGSIZE_VERT)
        fig.subplots_adjust(hspace=0.30, wspace=0.12)

        # First row: surface variables
        for col, var in enumerate(DATASET_VARIABLES_SURFACE):
            ax    = axes[0][col]
            title = f"{TRANSLATIONS[var]} ${UNITS[var]}$" if show_units else f"{TRANSLATIONS[var]} $[-]$"
            _plot_surface(ax, var, float(q25[col]), float(q50[col]), float(q75[col]), props)
            ax.set_title(title, fontweight="bold", fontsize=props["fs_title"])
            if not show_units:
                ax.set_xlim(0, _XLIM_STD_SURFACE)

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
        fig.savefig(save_dir / f"error_vertical_{label}.pdf", bbox_inches="tight")
        plt.close(fig)


def visualize_spectra(checkpoint_name: str) -> None:
    r"""Generate power-spectrum figure of variables.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
    """

    props  = scale_fig_properties(_FIGSIZE_SPEC)
    n_surf = len(DATASET_VARIABLES_SURFACE)

    # Load most recent spectra file and compute quantiles over time
    spectra_dir = PATH_DIAGNOSTICS / checkpoint_name / "power_spectra"
    data        = torch.load(sorted(spectra_dir.glob("*.pt"))[-1], map_location="cpu", weights_only=True)

    gt_q50  = data["ground_truth"].quantile(0.50, dim=0)
    gt_q25  = data["ground_truth"].quantile(0.25, dim=0)
    gt_q75  = data["ground_truth"].quantile(0.75, dim=0)
    rec_q50 = data["reconstruction"].quantile(0.50, dim=0)
    rec_q25 = data["reconstruction"].quantile(0.25, dim=0)
    rec_q75 = data["reconstruction"].quantile(0.75, dim=0)

    K           = gt_q50.shape[-1]
    wavelengths = min(X, Y) * _DX_KM / np.arange(1, K)

    # Output directory
    save_dir = PATH_EXP_AE_FIGURES / checkpoint_name / "spectra"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=_FIGSIZE_SPEC)
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
    fig.savefig(save_dir / "power_spectra.pdf", bbox_inches="tight")
    plt.close(fig)


def _channel_index(
        var: str,
        depth: int = 0,
) -> int:
    r"""Return the flat channel index for a given variable and depth level.

    Arguments:
        var   : Variable name (must be in DATASET_VARIABLES_SURFACE or DATASET_VARIABLES_OCEAN).
        depth : Depth level index (ignored for surface variables).

    Returns:
        c : Flat channel index into the (C, Y, X) map tensor.
    """
    if var in DATASET_VARIABLES_SURFACE:
        return DATASET_VARIABLES_SURFACE.index(var)
    return len(DATASET_VARIABLES_SURFACE) + DATASET_VARIABLES_OCEAN.index(var) * Z + depth


def visualize_error_maps_z(checkpoint_name: str) -> None:
    r"""Generate depth-section error maps (z).

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
    """

    props        = scale_fig_properties(_FIGSIZE_MAP_Z)
    lon2d, lat2d = load_grid()
    extent       = [float(lon2d.min()), float(lon2d.max()), float(lat2d.min()), float(lat2d.max())]

    # Load maps
    maps = torch.load(PATH_DIAGNOSTICS / checkpoint_name / "maps" / "maps.pt",
                      map_location="cpu", weights_only=True)

    for unit_label in ("standardized", "physical"):
        show_units = unit_label == "physical"

        for vgroup_tag, vgroup in [("PHY", _VARS_PHY), ("BIO", _VARS_BIO)]:
            for stat_label in ("rmse", "std"):

                field = maps[f"{stat_label}_{unit_label}"]  # (C, Y, X)
                cmap  = CMAPS_METRICS[stat_label]
                nrows = -(-len(vgroup) // 3)               # ceil(n / 3)
                fsize = (_FIGSIZE_MAP_Z[0], _FIGSIZE_MAP_Z[1] * nrows / 3)

                fig, axes = plt.subplots(nrows, 3, figsize=fsize,
                                         subplot_kw={"projection": ccrs.PlateCarree()})
                fig.subplots_adjust(hspace=0.25, wspace=0.25)

                for i, var in enumerate(vgroup):
                    ax = axes.flat[i]

                    if var in DATASET_VARIABLES_SURFACE:
                        data = field[_channel_index(var)].numpy()
                    else:
                        c0   = _channel_index(var, 0)
                        data = np.nanmedian(field[c0 : c0 + Z].numpy(), axis=0)

                    vmax = float(np.nanquantile(data, FIG_PROPERTIES["quantile_vmax"]))

                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    im = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap,
                                       vmin=0, vmax=vmax, rasterized=True,
                                       transform=ccrs.PlateCarree(), shading="auto")

                    draw_geography(ax)

                    cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                                      pad=0.15, shrink=1.01, aspect=25)
                    cb.ax.tick_params(labelsize=props["fs_x_tick"])
                    _apply_sci_fmt(cb.ax)

                    unit_str = f"${UNITS[var]}$" if show_units else "$[-]$"
                    ax.set_title(f"{TRANSLATIONS[var]} {unit_str}",
                                 fontweight="bold", fontsize=props["fs_title"])

                    ax.set_xticks(_LON_TICKS, crs=ccrs.PlateCarree())
                    ax.set_yticks(_LAT_TICKS, crs=ccrs.PlateCarree())
                    ax.xaxis.set_major_formatter(LongitudeFormatter())
                    ax.yaxis.set_major_formatter(LatitudeFormatter())
                    ax.tick_params(axis="both", labelsize=props["fs_x_tick"])

                for ax in axes.flat[len(vgroup):]:
                    ax.set_visible(False)

                fig.supxlabel(_STAT_LABELS[stat_label], fontsize=props["fs_sup_x_label"], y=0.045)

                save_dir = PATH_EXP_AE_FIGURES / checkpoint_name / "maps" / stat_label / "Z"
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / f"maps_{stat_label}_Z_{vgroup_tag}_{_UNIT_FILE[unit_label]}.pdf", bbox_inches="tight")
                plt.close(fig)


def _visualize_error_section(checkpoint_name: str, agg_axis: int, agg_tag: str) -> None:
    r"""Helper to generate depth-section error maps (x or y).

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
        agg_axis        : Axis to aggregate over.
        agg_tag         : Label used in the output filename ("X" or "Y").
    """

    props  = scale_fig_properties(_FIGSIZE_MAP_XY)
    depths = np.array([float(DEPTHS[i]) for i in range(Z)])

    lon2d, lat2d  = load_grid()
    horiz_coords  = lat2d[:, 0] if agg_axis == 2 else lon2d[0, :]
    horiz_ticks   = _LAT_TICKS  if agg_axis == 2 else _LON_TICKS
    horiz_tlabels = [f"{v}°N" for v in _LAT_TICKS] if agg_axis == 2 else [f"{v}°E" for v in _LON_TICKS]

    maps = torch.load(PATH_DIAGNOSTICS / checkpoint_name / "maps" / "maps.pt", map_location="cpu", weights_only=True)

    for unit_label in ("standardized", "physical"):
        show_units = unit_label == "physical"

        for vgroup_tag, vgroup in [("PHY", _VARS_PHY_SECTION), ("BIO", _VARS_BIO)]:
            for stat_label in ("rmse", "std"):

                field = maps[f"{stat_label}_{unit_label}"]   # (C, Y, X)
                cmap  = CMAPS_METRICS[stat_label].copy()
                cmap.set_bad("lightgrey")
                nrows = -(-len(vgroup) // 3)
                fsize = (_FIGSIZE_MAP_XY[0], _FIGSIZE_MAP_XY[1] * nrows / 2)

                fig, axes = plt.subplots(nrows, 3, figsize=fsize)
                fig.subplots_adjust(hspace=0.25, wspace=0.25)

                for i, var in enumerate(vgroup):
                    ax   = axes.flat[i]
                    c0   = _channel_index(var, 0)
                    data = np.nanmedian(field[c0 : c0 + Z].numpy(), axis=agg_axis)  # (Z, horiz)
                    vmax = float(np.nanquantile(data, FIG_PROPERTIES["quantile_vmax"]))

                    im = ax.pcolormesh(horiz_coords, depths, np.ma.array(data, mask=np.isnan(data)), cmap=cmap, vmin=0, vmax=vmax, rasterized=True, shading="auto")
                    ax.contour(horiz_coords, depths, (~np.isnan(data)).astype(float), levels=[0.5], colors="k", linewidths=0.5)
                    ax.invert_yaxis()

                    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.15, shrink=1.01, aspect=25)
                    cb.ax.tick_params(labelsize=props["fs_x_tick"])
                    _apply_sci_fmt(cb.ax)

                    unit_str = f"${UNITS[var]}$" if show_units else "$[-]$"
                    ax.set_title(f"{TRANSLATIONS[var]} {unit_str}", fontweight="bold", fontsize=props["fs_title"])
                    ax.set_xticks(horiz_ticks)
                    ax.set_xticklabels(horiz_tlabels)
                    ax.tick_params(axis="both", labelsize=props["fs_x_tick"])
                    if i % 3 == 0:
                        ax.set_ylabel("Depth [m]", fontsize=props["fs_y_label"])

                for ax in axes.flat[len(vgroup):]:
                    ax.set_visible(False)

                fig.supxlabel(_STAT_LABELS[stat_label], fontsize=props["fs_sup_x_label"], y=0.045)
                save_dir = PATH_EXP_AE_FIGURES / checkpoint_name / "maps" / stat_label / agg_tag
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / f"maps_{stat_label}_{agg_tag}_{vgroup_tag}_{_UNIT_FILE[unit_label]}.pdf", bbox_inches="tight")
                plt.close(fig)


def visualize_error_maps_x(checkpoint_name: str) -> None:
    r"""Generate depth-section error maps (x)."""
    _visualize_error_section(checkpoint_name, agg_axis=2, agg_tag="X")


def visualize_error_maps_y(checkpoint_name: str) -> None:
    r"""Generate depth-section error maps (y)."""
    _visualize_error_section(checkpoint_name, agg_axis=1, agg_tag="Y")


def visualize_reconstructions(checkpoint_name: str) -> None:
    r"""Generate reconstruction map figures.

    Arguments:
        checkpoint_name : Name of the model checkpoint directory under PATH_MODELS.
    """

    props        = scale_fig_properties(_FIGSIZE_RECON)
    lon2d, lat2d = load_grid()
    extent       = [float(lon2d.min()), float(lon2d.max()), float(lat2d.min()), float(lat2d.max())]

    # Load reconstructions (standardized) and convert to physical units
    recon_dir = PATH_DIAGNOSTICS / checkpoint_name / "reconstructions"
    data      = torch.load(sorted(recon_dir.glob("*.pt"))[-1], map_location="cpu", weights_only=False)
    mean, std = get_weights_stats(dim=1)                                              # (C, 1, 1)
    gts       = data["ground_truths"]   * std.unsqueeze(0) + mean.unsqueeze(0)       # (N, C, Y, X)
    recs      = data["reconstructions"] * std.unsqueeze(0) + mean.unsqueeze(0)       # (N, C, Y, X)
    dates     = data["dates"]

    for date_idx, date in enumerate(dates):
        for depth in _RECON_DEPTHS:
            for vgroup_tag, vgroup in [("PHY", _VARS_PHY), ("BIO", _VARS_BIO)]:
                n_vars  = len(vgroup)
                n_pairs = -(-n_vars // 3)
                nrows   = 2 * n_pairs
                fsize   = (_FIGSIZE_RECON[0], _FIGSIZE_RECON[1] * nrows / 3)

                # GridSpec: tight spacing within GT/Rec pairs, larger gap between pairs
                fig      = plt.figure(figsize=fsize)
                outer_gs = GridSpec(n_pairs, 1, figure=fig, hspace=0.12, bottom=0.04, top=0.96)
                axes_rows = []
                for p in range(n_pairs):
                    inner_gs = GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_gs[p], hspace=0.0, wspace=0.25)
                    for r in range(2):
                        axes_rows.append([fig.add_subplot(inner_gs[r, c], projection=ccrs.PlateCarree()) for c in range(3)])
                axes = np.array(axes_rows)  # (nrows, 3)

                for i, var in enumerate(vgroup):
                    col      = i % 3
                    pair_idx = i // 3
                    gt_row   = pair_idx * 2
                    rec_row  = pair_idx * 2 + 1

                    c        = _channel_index(var, depth)
                    gt_data  = gts [date_idx, c].numpy()
                    rec_data = recs[date_idx, c].numpy()

                    vmin = float(np.nanquantile(gt_data, FIG_PROPERTIES["quantile_vmin"]))
                    vmax = float(np.nanquantile(gt_data, FIG_PROPERTIES["quantile_vmax"]))
                    cmap = CMAPS_FIELD[var]

                    for ax, field in [(axes[gt_row, col], gt_data), (axes[rec_row, col], rec_data)]:
                        ax.set_extent(extent, crs=ccrs.PlateCarree())
                        im = ax.pcolormesh(lon2d, lat2d, field, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True, transform=ccrs.PlateCarree(), shading="auto")
                        draw_geography(ax)
                        ax.set_xticks(_LON_TICKS, crs=ccrs.PlateCarree())
                        ax.set_yticks(_LAT_TICKS, crs=ccrs.PlateCarree())
                        ax.xaxis.set_major_formatter(LongitudeFormatter())
                        ax.yaxis.set_major_formatter(LatitudeFormatter())
                        ax.tick_params(axis="both", labelsize=props["fs_x_tick"])

                    # Colorbar below rec row; invisible dummy below gt row to keep equal axis sizes
                    axes[gt_row, col].set_title(f"{TRANSLATIONS[var]} ${UNITS[var]}$", fontweight="bold", fontsize=props["fs_title"])
                    for row, visible in [(gt_row, False), (rec_row, True)]:
                        cb = fig.colorbar(im, ax=axes[row, col], orientation="horizontal",
                                           pad=0.12, shrink=1.01, aspect=25)
                        cb.ax.tick_params(labelsize=props["fs_x_tick"])
                        if not visible:
                            cb.ax.set_visible(False)
                        else:
                            _apply_sci_fmt(cb.ax)

                # Row labels on last valid column of each pair
                for pair_idx in range(n_pairs):
                    n_in_pair = min(3, n_vars - pair_idx * 3)
                    last_col  = n_in_pair - 1
                    for row, label in [(pair_idx * 2, "Ground Truths"), (pair_idx * 2 + 1, "Reconstructions")]:
                        axes[row, last_col].text(
                            1.03, 0.5, label, transform=axes[row, last_col].transAxes,
                            va="center", ha="left", fontsize=props["fs_title"],
                            rotation=-90, fontweight="bold",
                        )

                # Hide empty subplots in last pair (when n_vars % 3 != 0)
                for i in range(n_vars, n_pairs * 3):
                    axes[i // 3 * 2,     i % 3].set_visible(False)
                    axes[i // 3 * 2 + 1, i % 3].set_visible(False)

                fig.supxlabel(f"Ground Truths VS Reconstructions on {date}  (depth = {DEPTHS[depth]} [m])", fontsize=props["fs_sup_x_label"])
                save_dir = PATH_EXP_AE_FIGURES / checkpoint_name / "reconstructions" / date / DEPTHS[depth]
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / f"reconstructions_{date}_{DEPTHS[depth]}_{vgroup_tag}_unscaled.pdf", bbox_inches="tight")
                plt.close(fig)
