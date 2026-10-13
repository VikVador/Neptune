r"""Figures and animations of the diagnostics of forecasts."""

__all__ = [
    "animate_forecast",
    "plot_animations",
    "plot_score_maps",
    "plot_score_local",
    "plot_score_global",
    "log_figures",
]

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, Colormap, Normalize
from matplotlib.lines import Line2D
from pathlib import Path
from PIL import Image
from torch import Tensor
from typing import Any

from neptune.config import PATH_ANALYSIS, PATH_DIAGNOSTICS
from neptune.data import (
    DATASET_VARIABLES_OCEAN,
    DATASET_VARIABLES_OCEAN_BIO,
    DATASET_VARIABLES_OCEAN_PHY,
    DATASET_VARIABLES_SURFACE,
)
from neptune.diagnostics import (
    CMAPS_LINE_MODELS,
    CMAPS_METRICS,
    CMAPS_SPATIAL_VARS,
    FIG_PROPERTIES,
    LAYERS,
    TRANSLATIONS_METRICS,
    TRANSLATIONS_MODELS,
    TRANSLATIONS_VARS,
    UNITS,
)
from neptune.diagnostics.geography import draw_geography, load_depth, load_grid

# fmt: off
#
_FS           = FIG_PROPERTIES
_LON_TICKS    = [28, 32, 36, 40]
_LAT_TICKS    = [42, 44, 46]
_DEPTH_TICKS  = [1, 10, 100, 1000]
_RATIO_BOUNDS = [0, *np.linspace(0.05, 1.95, 20), 2]
_RATIO_TICKS  = [0, 0.5, 1, 1.5, 2]
_PANEL_SIZE   = (2.4, 0.95)
_PANEL_SPACE  = (0.7, 0.85)
_PANEL_BREAK  = 0.08
_LINE         = dict(linewidth=1.5, marker="o", markersize=3)
_GRID         = dict(color="black", linewidth=0.3, alpha=0.3)
_GROUPS       = {"physique": (DATASET_VARIABLES_SURFACE, DATASET_VARIABLES_OCEAN_PHY), "biogeochimie": ([], DATASET_VARIABLES_OCEAN_BIO)}


def _unit(unit: str) -> str:
    r"""Unit between brackets, upright and regular whatever the weight of its text.

    Arguments:
        unit : Unit, e.g. 'm/s' or '-'.

    Returns:
        unit : Unit in mathtext.
    """
    return f"$[\\mathrm{{{unit}}}]$"


def _title(variable: str, metric: str = "skill") -> str:
    r"""Title of a variable, with the units of a metric (none for the spread-skill ratio).

    Arguments:
        variable : Name of the variable.
        metric   : Name of the metric, a key of TRANSLATIONS_METRICS.

    Returns:
        title : Title of the variable.
    """
    return f"{TRANSLATIONS_VARS[variable]} {_unit('-' if metric == 'ratio' else UNITS[variable])}"


def _metric(metric: str) -> str:
    r"""Name of a metric, with a bold arrow between regular parentheses for the metrics to minimize.

    Arguments:
        metric : Name of the metric, a key of TRANSLATIONS_METRICS.

    Returns:
        name : Name of the metric.
    """
    return f"{TRANSLATIONS_METRICS[metric]} $($↓$)$" if metric in ("skill", "spread") else TRANSLATIONS_METRICS[metric]


def _set_depth(ax: plt.Axes, depth: np.ndarray) -> None:
    r"""Set a logarithmic depth on the y axis, from the surface on top, with a horizontal grid on its ticks.

    Arguments:
        ax    : Axes to format.
        depth : Depth of each level (Z,) [m].
    """

    ax.set_yscale("log")
    ax.set_ylim(depth[-1], depth[0])
    ax.set_yticks(_DEPTH_TICKS, [f"{d} [m]" for d in _DEPTH_TICKS])
    ax.set_yticks([], minor=True)
    ax.grid(axis="y", which="major", **_GRID)


def _set_map(ax: plt.Axes, latitudes: bool = True, longitudes: bool = True) -> None:
    r"""Draw the coastline and rivers of the Black Sea on a map, with its latitudes and longitudes.

    Arguments:
        ax         : Axes to format, in the PlateCarree projection.
        latitudes  : Whether to show the latitudes.
        longitudes : Whether to show the longitudes.
    """

    draw_geography(ax)
    if latitudes:
        ax.set_yticks(_LAT_TICKS, crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
    if longitudes:
        ax.set_xticks(_LON_TICKS, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.tick_params(axis="both", labelsize=_FS["fs_tick"])


def _set_section(ax: plt.Axes, plane: str, depth: np.ndarray, depths: bool = True, coordinates: bool = True) -> None:
    r"""Set the logarithmic depth and the latitudes (y-z) or longitudes (z-x) of a section.

    Arguments:
        ax          : Axes to format.
        plane       : Plane of the section, 'y-z' or 'z-x'.
        depth       : Depth of each level (Z,) [m].
        depths      : Whether to show the depths.
        coordinates : Whether to show the latitudes or longitudes.
    """

    ticks, suffix = (_LAT_TICKS, "°N") if plane == "y-z" else (_LON_TICKS, "°E")
    _set_depth(ax, depth)
    ax.set_xticks(ticks, [f"{t}{suffix}" for t in ticks])
    ax.tick_params(left=depths, labelleft=depths, bottom=coordinates, labelbottom=coordinates, labelsize=_FS["fs_tick"])


def _colors(field: np.ndarray, metric: str, cmap: Colormap) -> dict:
    r"""Colors of a field of a metric, up to a quantile of the field, or binned for the spread-skill ratio.

    Arguments:
        field  : Field of the metric, NaN on land.
        metric : Name of the metric, a key of TRANSLATIONS_METRICS.
        cmap   : Colormap of the metric.

    Returns:
        colors : Colormap and normalization of the field.
    """

    if metric == "ratio":
        bins = len(_RATIO_BOUNDS) - 1
        return {"cmap": cmap.resampled(bins), "norm": BoundaryNorm(_RATIO_BOUNDS, bins)}

    return {"cmap": cmap, "norm": Normalize(0.0, float(np.nanquantile(field, _FS["quantile_vmax"])))}


def _set_colorbar(colorbar: Colorbar, metric: str, ticks: list[float] = _RATIO_TICKS, size: float = _FS["fs_tick"]) -> None:
    r"""Size the ticks of the colorbar of a metric, the spread-skill ratio having fixed ones.

    Arguments:
        colorbar : Colorbar to format.
        metric   : Name of the metric, a key of TRANSLATIONS_METRICS.
        ticks    : Ticks of the spread-skill ratio.
        size     : Font size of the ticks.
    """

    colorbar.ax.tick_params(labelsize=size)
    if metric == "ratio":
        colorbar.set_ticks(ticks)
        colorbar.minorticks_off()


def _save(fig: plt.Figure, path: Path, **kwargs: Any) -> None:
    r"""Save a figure as a PDF and as a PNG, the image logged to Weights & Biases, then close it.

    Arguments:
        fig    : Figure to save.
        path   : Path to the output .pdf file.
        kwargs : Keyword arguments of Figure.savefig.
    """

    fig.savefig(path, **kwargs)
    fig.savefig(path.with_suffix(".png"), dpi=150, **kwargs)
    plt.close(fig)


def _to_gif_palette(images: list[Image.Image]) -> list[Image.Image]:
    r"""Convert the frames of a GIF to a single palette of 256 colors, keeping exact colors.

    Arguments:
        images : Frames in RGB.

    Returns:
        images : Frames in palette mode, sharing the same palette.
    """

    exact = [(255, 0, 0), (255, 255, 255), (0, 0, 0), (211, 211, 211)]
    adaptive = 256 - len(exact)
    colors = [c for color in exact for c in color] + images[0].quantize(colors=adaptive, dither=Image.Dither.NONE).getpalette()[: 3 * adaptive]
    palette = Image.new("P", (1, 1))
    palette.putpalette(colors)

    frames = []
    for image in images:
        indices = np.array(image.quantize(palette=palette, dither=Image.Dither.NONE))
        for i, color in enumerate(exact):
            indices[(np.asarray(image) == color).all(axis=-1)] = i

        frame = Image.fromarray(indices, mode="P")
        frame.putpalette(colors)
        frames.append(frame)

    return frames


def animate_forecast(
    frames: dict[str, Tensor],
    variable: str,
    date: str,
    plane: str,
    position: str,
    path: Path,
    duration: float = 0.5,
) -> None:
    r"""Animate an ensemble forecast of a slice of a field against the ground truth, one frame per day.

    Arguments:
        frames   : Fields (K, A, B), members (K, M, A, B) and skills (K,) of each day, from compute_animations.
        variable : Name of the variable, e.g. 'votemper'.
        date     : Date of the last input state 'YYYY-MM-DD', the forecast starting the day after.
        plane    : Plane of the slice, 'x-y', 'y-z' or 'z-x'.
        position : Position of the slice in the title, e.g. 'Surface' or '43.00 [°N]'.
        path     : Path to the output .gif file.
        duration : Time between two frames (days) [s].
    """

    frames = {key: value.numpy() for key, value in frames.items()}
    days, members = frames["members"].shape[:2]
    vmin, vmax = np.nanquantile(frames["truth"], [0.05, 0.95])
    colors = dict(cmap=CMAPS_SPATIAL_VARS[variable], vmin=vmin, vmax=vmax)
    lon, lat = load_grid()
    depth = load_depth()

    # Sections | Horizontal coordinate of the slice (latitude or longitude), with the aspect of the maps
    coords = lat[:, 0] if plane == "y-z" else lon[0]
    aspect = (lat.max() - lat.min()) / (lon.max() - lon.min())

    # Figure | Ground truth, ensemble mean and median on top, members below (4 per row), shared colorbar,
    #          with heights in inches : rows of maps (geographic aspect) separated by gaps, the last one
    #          leaving room for the longitudes, then the colorbar
    rows = 1 + -(-members // 4)
    heights = [1.25, 0.25] * (rows - 1) + [1.25, 0.45, 0.12]
    height = sum(heights) + 1.05
    fig = plt.figure(figsize=(16, height), facecolor="white")
    grid = fig.add_gridspec(len(heights), 4, height_ratios=heights, hspace=0, wspace=0.1, top=1 - 0.75 / height, bottom=0.3 / height)
    panels = [("Ground Truth", 0, 0), ("Ensemble Mean", 0, 1), ("Ensemble Median", 0, 2)]
    panels += [(None, 1 + m // 4, m % 4) for m in range(members)]

    axes, meshes, values, counters = [], [], [], []
    for i, (title, row, col) in enumerate(panels):
        cell = grid[2 * row, col]

        # Coordinates | Latitudes or depths on the first column, longitudes or latitudes on the last row
        if plane == "x-y":
            ax = fig.add_subplot(cell, projection=ccrs.PlateCarree())
            meshes.append(ax.pcolormesh(lon, lat, frames["truth"][0], transform=ccrs.PlateCarree(), **colors))
            _set_map(ax, latitudes=col == 0, longitudes=row == rows - 1)
        else:
            ax = fig.add_subplot(cell)
            meshes.append(ax.pcolormesh(coords, depth, frames["truth"][0], shading="auto", **colors))
            ax.set_box_aspect(aspect)
            _set_section(ax, plane, depth, depths=col == 0, coordinates=row == rows - 1)

        ax.set_facecolor("lightgray")
        if title is not None:
            ax.set_title(title, fontsize=_FS["fs_title"], fontweight="bold", pad=4)
        axes.append(ax)

        # Lead time in the upper left corner, and skill of the ensemble mean and median in the upper right corner
        style = dict(va="top", color="white", fontsize=_FS["fs_text"], fontweight="bold", bbox=dict(facecolor="black", edgecolor="none", pad=1), zorder=4)
        corner = dict(xycoords="axes fraction", textcoords="offset points", **style)
        counters.append(ax.annotate("", xy=(0, 1), xytext=(2, -2), ha="left", **corner))
        if i in (1, 2):
            values.append(ax.annotate("", xy=(1, 1), xytext=(-2, -2), ha="right", **corner))
            ax.annotate("Skill = ", xy=(0, 1), xycoords=values[-1], ha="right", **style)

    colorbar = fig.colorbar(meshes[0], cax=fig.add_subplot(grid[-1, :]), orientation="horizontal")
    colorbar.ax.tick_params(labelsize=_FS["fs_tick"])

    # Members | Labelled on the right of the last column, once the maps have their geographic aspect
    fig.canvas.draw()
    top, bottom = axes[6].get_position(), axes[-1].get_position()
    fig.text(top.x1 + 0.01, (bottom.y0 + top.y1) / 2, "Members", rotation=-90, ha="left", va="center", fontsize=_FS["fs_title"], fontweight="bold")

    fig.suptitle(f"{_title(variable)} — {position} — {date}", fontsize=_FS["fs_suptitle"], fontweight="bold")

    # Frames | One per forecasted day, in RGB since GIF transparency would erase the black pixels
    images = []
    for k in range(days):
        fields = [frames["truth"][k], frames["mean"][k], frames["median"][k], *frames["members"][k]]

        for mesh, field in zip(meshes, fields, strict=True):
            mesh.set_array(np.ma.masked_invalid(field))

        skills = (frames["skill"][k], frames["skill_median"][k])
        for text, value in zip(values, skills, strict=True):
            text.set_text(f"{value:.4f}")
            text.set_color("red" if value > min(skills) else "white")

        for counter in counters:
            counter.set_text(f"+{k + 1}")

        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))

    plt.close(fig)

    images = _to_gif_palette(images)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 * duration),
        loop=0,
    )


def plot_animations(
    checkpoint_name: str,
    split: str,
    date: str,
    variable: str,
    duration: float = 0.5,
) -> None:
    r"""Animate the forecast of a variable started on a date, along each saved slice.

    Arguments:
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
        date            : Date of the last input state 'YYYY-MM-DD', from compute_animations.
        variable        : Name of the variable, e.g. 'votemper'.
        duration        : Time between two frames (days) [s].
    """

    depth = load_depth()
    lon, lat = load_grid()
    slices = torch.load(PATH_DIAGNOSTICS / checkpoint_name / split / "animations" / date / f"{variable}.pt", weights_only=True)

    # Paths and titles | Surface (surface variables and first level), depth, longitude or latitude of each slice
    for (plane, p), frames in slices.items():
        if plane == "x-y":
            surface = p in (None, 0)
            position = "Surface" if surface else f"{depth[p]:.1f} {_unit('m')}"
            name = "surface" if surface else f"{depth[p]:.2f}"
        elif plane == "y-z":
            position, name = f"{lon[0, p]:.2f} {_unit('°E')}", f"{lon[0, p]:.2f}E"
        else:
            position, name = f"{lat[p, 0]:.2f} {_unit('°N')}", f"{lat[p, 0]:.2f}N"

        folder = PATH_ANALYSIS / split / checkpoint_name / "animations" / date / plane
        path = folder / TRANSLATIONS_VARS[variable].lower().replace(" ", "-") / f"{variable}-{name}.gif"
        path.parent.mkdir(parents=True, exist_ok=True)
        animate_forecast(frames, variable, date, plane, position, path, duration)


def _plot_score_map(fields: dict[str, np.ndarray], metric: str, cmap: Colormap, plane: str, path: Path) -> None:
    r"""Plot a metric of each variable on a map (x-y) or a section (y-z or z-x), 3 per row.

    Arguments:
        fields : Field of the metric of each variable, NaN on land, (Y, X), (Z, Y) or (Z, X).
        metric : Name of the metric, a key of TRANSLATIONS_METRICS.
        cmap   : Colormap of the metric.
        plane  : Plane of the fields, 'x-y', 'y-z' or 'z-x'.
        path   : Path to the output .pdf file.
    """

    depth = load_depth()
    lon, lat = load_grid()
    rows = -(-len(fields) // 3)
    fig = plt.figure(figsize=(13, (3.0 if plane == "x-y" else 3.5) * rows))

    for i, (var, field) in enumerate(fields.items()):
        colors = dict(**_colors(field, metric, cmap), rasterized=True, shading="auto")
        if plane == "x-y":
            ax = fig.add_subplot(rows, 3, i + 1, projection=ccrs.PlateCarree())
            ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())], crs=ccrs.PlateCarree())
            image = ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), transform=ccrs.PlateCarree(), **colors)
            _set_map(ax)
        else:
            ax = fig.add_subplot(rows, 3, i + 1)
            coords = lat[:, 0] if plane == "y-z" else lon[0]
            image = ax.pcolormesh(coords, depth, np.ma.masked_invalid(field), **colors)
            ax.contour(coords, depth, np.isfinite(field).astype(float), levels=[0.5], colors="black", linewidths=0.5)
            _set_section(ax, plane, depth)

        ax.set_facecolor("lightgray")
        ax.set_title(_title(var, metric), fontsize=_FS["fs_title"], fontweight="bold")
        _set_colorbar(fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.15, shrink=1.01, aspect=25, spacing="proportional"), metric)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig.supxlabel(_metric(metric), fontsize=_FS["fs_suptitle"], fontweight="bold", y=0.045)
    _save(fig, path, bbox_inches="tight")


def plot_score_maps(
    maps: dict[str, dict[str, Tensor]],
    checkpoint_name: str,
    split: str,
    day: str,
) -> None:
    r"""Plot the maps of each metric of a forecasted day over a split, one folder per metric.

    Arguments:
        maps            : Maps of each metric, at the surface and along each axis, from aggregate_maps.
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
        day             : Forecasted day, 'first' or 'last'.
    """

    for metric, fields in maps.items():
        folder = PATH_ANALYSIS / split / checkpoint_name / "score-maps" / day / metric
        folder.mkdir(parents=True, exist_ok=True)

        for group, (surface, ocean) in _GROUPS.items():
            cmap = CMAPS_METRICS[group][metric]
            if surface:
                _plot_score_map({var: fields["surface"][DATASET_VARIABLES_SURFACE.index(var)].numpy() for var in surface}, metric, cmap, "x-y", folder / "surface.pdf")

            for plane, name in (("x-y", "Z"), ("y-z", "X"), ("z-x", "Y")):
                _plot_score_map({var: fields[name][DATASET_VARIABLES_OCEAN.index(var)].numpy() for var in ocean}, metric, cmap, plane, folder / f"ocean_{group}_{name}.pdf")


def _plot_score_card(
    fig: plt.Figure,
    ax: plt.Axes,
    values: np.ndarray,
    metric: str,
    cmap: Colormap,
    variable: str,
    first: bool,
    last: bool,
) -> plt.Text | None:
    r"""Plot a metric against the lead time and the depth, or along a single row at the surface.

    Arguments:
        fig      : Figure (or subfigure) of the axes, holding the colorbar.
        ax       : Axes to draw on.
        values   : Metric of each forecasted day, at the surface (K,) or per level (K, Z).
        metric   : Name of the metric, a key of TRANSLATIONS_METRICS.
        cmap     : Colormap of the metric.
        variable : Name of the variable.
        first    : Whether the axes belong to the first column.
        last     : Whether the axes belong to the last column.

    Returns:
        text : Name of the variable on the first column, to keep when saving the figure, else None.
    """

    days = np.arange(1, len(values) + 1)
    ax.set_facecolor("lightgray")
    colors = dict(**_colors(values, metric, cmap), shading="nearest", rasterized=True)

    if values.ndim == 1:
        image = ax.pcolormesh(np.arange(len(values) + 1) + 0.5, [0, 1], values[None], **{**colors, "shading": "flat"})
        ax.set_yticks([])
        for day, value in zip(days, values, strict=True):
            if np.isfinite(value):
                ax.text(day, 0.5, f"{value:.2g}", ha="center", va="center", fontsize=_FS["fs_text"], fontweight="light", color="black")
    else:
        depth = load_depth()
        image = ax.pcolormesh(days, depth, values.T, **colors)
        _set_depth(ax, depth)
        ax.tick_params(axis="y", left=False, labelleft=False, right=last, labelright=last)

    ax.set_xticks(days)
    ax.tick_params(axis="both", labelsize=_FS["fs_tick"])

    colorbar = fig.colorbar(image, ax=ax, location="left", pad=0.02, aspect=10 if values.ndim == 1 else 20, spacing="proportional")
    _set_colorbar(colorbar, metric, [0, 1, 2] if values.ndim == 1 else _RATIO_TICKS, _FS["fs_text"])

    # Variable | Outside of the layout, to keep the same width for every column
    if not first:
        return None

    text = colorbar.ax.annotate(
        f"{TRANSLATIONS_VARS[variable]}\n{_unit(UNITS[variable])}",
        xy=(0, 0.5),
        xycoords="axes fraction",
        xytext=(-36, 0) if values.ndim == 1 else (-46, 0),
        textcoords="offset points",
        rotation=0 if values.ndim == 1 else 90,
        rotation_mode="anchor",
        ha="right" if values.ndim == 1 else "center",
        multialignment="right" if values.ndim == 1 else "center",
        va="center",
        fontsize=_FS["fs_label"],
        fontweight="bold",
    )
    text.set_in_layout(False)

    return text


def plot_score_local(
    metrics: dict[tuple[str, str], tuple[Tensor, Tensor, Tensor]],
    checkpoint_name: str,
    split: str,
) -> None:
    r"""Plot each metric of the forecast against the lead time and the depth.

    Arguments:
        metrics         : Metrics of the forecast and of the baselines, from aggregate_metrics.
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
    """

    folder = PATH_ANALYSIS / split / checkpoint_name / "score-local"
    folder.mkdir(parents=True, exist_ok=True)

    for group, (surface, ocean) in _GROUPS.items():
        heights = [0.35] * len(surface) + [1.05] * len(ocean)
        fig = plt.figure(figsize=(13.5, sum(heights) + 0.25 * len(ocean) + 1.2), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0, wspace=0)

        names = []
        # Columns | The last one is wider by the width of the depths, such that every panel has the same width
        columns = fig.subfigures(1, 3, wspace=0.03, width_ratios=[1, 1, 1.16])
        for c, (column, metric) in enumerate(zip(columns, TRANSLATIONS_METRICS, strict=True)):
            scores, skills = ([t.numpy() for t in metrics[("forecast", name)]] for name in (metric, "skill"))
            axes = column.subplots(len(heights), 1, sharex=True, gridspec_kw={"height_ratios": heights})

            for ax, var in zip(axes, surface + ocean, strict=True):
                g, variables = (0, DATASET_VARIABLES_SURFACE) if var in DATASET_VARIABLES_SURFACE else (1, DATASET_VARIABLES_OCEAN)
                values, skill = scores[g][:, variables.index(var)], skills[g][:, variables.index(var)]

                # Constant levels (e.g. anoxic oxygen) have a null skill and an undefined ratio, shown in gray
                if metric == "ratio":
                    values = np.where(skill > 0, values, np.nan)

                names.append(_plot_score_card(column, ax, values, metric, CMAPS_METRICS[group][metric], var, first=c == 0, last=c == 2))

            # Lead times on the last row only
            for ax in axes[:-1]:
                ax.tick_params(axis="x", bottom=False)
            axes[-1].set_xlabel("Lead Time [days]", fontsize=_FS["fs_title"], labelpad=8)

            # Metric | Column title, an arrow for the metrics to minimize and a short empty line keeping it away from the first variable
            suffix = f" {_unit('-')}" if metric == "ratio" else ""
            column.suptitle(f"{_metric(metric)}{suffix}\n", fontsize=_FS["fs_suptitle"], fontweight="bold", linespacing=0.6)

        # Saving | The names of the variables, outside of the layout, added to the default artists of the figure
        extra = fig.get_default_bbox_extra_artists() + [name for name in names if name is not None]
        _save(fig, folder / f"sl-{group}.pdf", bbox_inches="tight", bbox_extra_artists=extra)


def _break(top: plt.Axes, bottom: plt.Axes) -> None:
    r"""Join two axes stacked vertically by a break of their y axis, drawn as // on both sides.

    Arguments:
        top    : Upper axes.
        bottom : Lower axes.
    """

    top.spines.bottom.set_visible(False)
    bottom.spines.top.set_visible(False)
    top.tick_params(axis="x", bottom=False, labelbottom=False)

    marks = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8, linestyle="none", color="black", markeredgewidth=1, clip_on=False)
    top.plot([0, 1], [0, 0], transform=top.transAxes, **marks)
    bottom.plot([0, 1], [1, 1], transform=bottom.transAxes, **marks)


def plot_score_global(
    metrics: dict[tuple[str, str], tuple[Tensor, Tensor, Tensor]],
    checkpoint_name: str,
    split: str,
) -> None:
    r"""Plot the skill of the forecast and of the baselines against the lead time, one panel per variable.

    Arguments:
        metrics         : Metrics of the forecast and of the baselines, from aggregate_metrics.
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
    """

    folder = PATH_ANALYSIS / split / checkpoint_name / "score-global"
    folder.mkdir(parents=True, exist_ok=True)

    tops = [0, *LAYERS]
    layers = [f"{top}–{bottom} {_unit('m')}" for top, bottom in zip(tops, LAYERS, strict=False)] + [f"> {tops[-1]} {_unit('m')}"]
    baselines = ["forecast", *CMAPS_LINE_MODELS]

    # Layout | In inches, 4 panels per row, a surface panel being as high as a layer of an ocean one
    width, height = _PANEL_SIZE
    space_x, space_y = _PANEL_SPACE
    for group, (surface, ocean) in _GROUPS.items():
        rows = ([surface] if surface else []) + [ocean[i : i + 4] for i in range(0, len(ocean), 4)]
        counts = [1 if row is surface else len(layers) for row in rows]
        heights = [n * height + (n - 1) * _PANEL_BREAK for n in counts]
        size = (4 * width + 3 * space_x, sum(heights) + len(rows) * space_y)
        fig = plt.figure(figsize=size)
        colors = {"forecast": CMAPS_METRICS[group]["skill"](0.8), **CMAPS_LINE_MODELS}

        top = size[1]
        for r, row in enumerate(rows):
            top -= space_y / 2
            for c, var in enumerate(row):

                # Panel | A single axes at the surface, else one per layer from the top, joined by breaks of the y axis
                if row is surface:
                    i = DATASET_VARIABLES_SURFACE.index(var)
                    names, curves = ["Surface"], [{b: metrics[(b, "skill")][0][:, i].numpy() for b in baselines}]
                else:
                    i = DATASET_VARIABLES_OCEAN.index(var)
                    names, curves = layers, [{b: metrics[(b, "skill")][2][:, i, l].numpy() for b in baselines} for l in range(len(layers))]

                left = c * (width + space_x)
                axes = [fig.add_axes([left / size[0], (top - (l + 1) * height - l * _PANEL_BREAK) / size[1], width / size[0], height / size[1]]) for l in range(counts[r])]
                for upper, lower in zip(axes[:-1], axes[1:], strict=True):
                    lower.sharex(upper)
                    _break(upper, lower)

                # Lines | The forecast on top of the baselines
                for ax, name, skills in zip(axes, names, curves, strict=True):
                    days = np.arange(1, len(skills["forecast"]) + 1)
                    for baseline, skill in reversed(skills.items()):
                        ax.plot(days, skill, color=colors[baseline], **_LINE)

                    ax.set_xticks(days)
                    ax.set_ylim(bottom=0)
                    ax.tick_params(axis="both", labelsize=_FS["fs_tick"])
                    if c == 0:
                        ax.set_ylabel(name, fontsize=_FS["fs_label"], fontweight="bold")

                axes[0].set_title(_title(var), fontsize=_FS["fs_title"], fontweight="bold")
                if r == len(rows) - 1:
                    axes[-1].set_xlabel("Lead Time [days]", fontsize=_FS["fs_title"])

            top -= heights[r] + space_y / 2

        # Legend | On a single row above the panels
        handles = [Line2D([], [], color=colors[b], **_LINE) for b in baselines]
        fig.legend(handles, [TRANSLATIONS_MODELS[b] for b in baselines], loc="lower center", bbox_to_anchor=(0.5, 1), ncols=len(handles), fontsize=_FS["fs_title"], frameon=False)

        # Metric | Below the lead times of the last row
        fig.text(0.5, -0.2 / size[1], _metric("skill"), ha="center", va="top", fontsize=_FS["fs_suptitle"], fontweight="bold")
        _save(fig, folder / f"sg-{group}.pdf", bbox_inches="tight")


def log_figures(checkpoint_name: str, split: str, config_wandb: dict) -> None:
    r"""Log the figures, as images, and the animations of the diagnostics of a split to Weights & Biases.

    Arguments:
        checkpoint_name : Name of the checkpoint in PATH_MODELS.
        split           : Split of the dataset, 'validation' or 'test'.
        config_wandb    : Weights & Biases entity, project and mode.
    """

    folder = PATH_ANALYSIS / split / checkpoint_name
    wandb.init(**config_wandb, name=f"{checkpoint_name}-{split}", job_type="diagnostics")

    # Media | Keyed by their path in the folder of the figures
    media = {path.relative_to(folder).with_suffix("").as_posix(): wandb.Image(str(path)) for path in sorted(folder.rglob("*.png"))}
    media |= {path.relative_to(folder).with_suffix("").as_posix(): wandb.Video(str(path), format="gif") for path in sorted(folder.rglob("*.gif"))}
    wandb.log(media)

    wandb.finish()
