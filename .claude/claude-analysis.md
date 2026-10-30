```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       Analysis: how to write reports and draw figures
     ┌─┴───┴─┐       for my projects.
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# ANALYSIS

Read this file **before writing a report or drawing a figure**. Projects and libraries to know:
`claude-ressources.md`. How to write the code: `claude-python.md`.

**Last checked: 2026-10-30** (every link below was opened and verified on that day).

**Contents**

0. In short
1. Reports
2. Choosing the plot
3. Encoding and colours
4. Scales
5. Guides, captions and uncertainty
6. Maps
7. Anti-patterns
8. Code
9. Before delivering
10. Sources

---

## 0. IN SHORT

1. **I like figures**: when a result can be shown, show it. The text explains the figure, not the other way around.
2. **Start from the message**: a comparison, a distribution, a correlation or an evolution.
3. **Geographic data**: one map at the start with every zone named in the text (§ 6).
4. **Every map**: longitude and latitude axes, units in `[]` with LaTeX, a colorbar of the size of the axis,
   the 2% and 98% quantiles as limits (except for an extreme event), thin black contours (§ 6).
5. **Do not let the reader misread**: position before colour, no rainbow, no cut bars, no 3D, uncertainty shown.

---

## 1. REPORTS

| Rule                    | In practice                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| Figures first           | every result comes with a figure; prefer a figure to a paragraph of numbers                 |
| One figure, one message | the title states the finding ("Shelf oxygen drops in summer"), not the content              |
| Explanatory, not a dump | as many figures as there are messages; an exploratory sweep stays out of the report         |
| Rawest data first       | show fields and samples before their summaries: the same statistics can hide anything       |
| All the data            | no subset of runs, dates or zones without saying so: the pattern may be in what is cut      |
| Same choices everywhere | one variable, model or zone keeps the same colour, name and order in the whole report       |
| Table or figure         | a table for exact numbers, a heatmap or lines for a pattern, never bars with printed values |

Metrics come from `albus` (`claude-ressources.md`) rather than from a new implementation.

---

## 2. CHOOSING THE PLOT

| Message      | Plot                                                            | Example in my projects            |
| ------------ | --------------------------------------------------------------- | --------------------------------- |
| Comparison   | bars starting at zero, or dots                                  | skill of several models           |
| Distribution | all the points along an axis, or a histogram / box plot if many | sea surface temperature at a site |
| Correlation  | a scatter plot, a matrix of scatter plots above two variables   | predicted against true oxygen     |
| Evolution    | a line against time or another ordered variable                 | CRPS against the lead time        |

- **Subsets**: tell them apart within one panel, or use small multiples **sharing their scales**. Keeping the other
  points in grey gives each panel its reference.
- **Overplotting**: use transparency, jitter or binning to give the density back.
- **Coordinates**: Cartesian by default. Polar only for cyclic data (seasons), as it distorts lengths.

---

## 3. ENCODING AND COLOURS

Channels from the most to the least accurately read:

**position on a common scale > position on unaligned scales > length > angle > area > depth > luminance, saturation >
curvature, volume.**

So a map, read through colour, shows **where**. To compare **how much**, add a line, a transect or a profile.

| Data                         | Colormap                                                                 |
| ---------------------------- | ------------------------------------------------------------------------ |
| Ordered                      | sequential and perceptually uniform: `viridis`, `cividis`                |
| Ordered, meaningful midpoint | diverging, centred on the midpoint: `RdBu_r`, `cmo.balance`              |
| Unordered groups             | few colourblind-safe colours: `#0072B2`, `#E69F00`, `#009E73`, `#D55E00` |
| Temperature, salinity        | `cmo.thermal`, `cmo.haline`                                              |
| Chlorophyll, current speed   | `cmo.algae`, `cmo.speed`                                                 |

- **Never `jet` or a rainbow**: it invents edges and bands that are not in the data.
- A signed quantity (anomaly, bias) is diverging. A positive one (RMSE, CRPS, spread) is sequential.
- An ordered quantity (lead time, noise level) gets a sequential colormap, not categorical colours.
- **Colour never carries alone** what the reader must not miss (1 in 12 men has a colour vision deficiency):
  repeat it with position, shape or a label.
- Colour can draw attention: everything in grey, the one thing to notice in colour.

---

## 4. SCALES

| Scale       | When                                 | Beware                                                         |
| ----------- | ------------------------------------ | -------------------------------------------------------------- |
| Linear      | by default                           | large values can dwarf the small ones                          |
| Logarithmic | several orders of magnitude, spectra | hides absolute differences, no zero nor negative value, say it |
| Quantile    | skewed data, equal-count classes     | says who is ahead, not by how much (not the limits of § 6)     |

- **Binning is a choice**: too few bins hide a group, too many show noise as structure. Try several.
- **Bars start at zero**, as they encode a length. Dots and lines encode a position and can show a narrow range.
- No misleading aspect ratio.

---

## 5. GUIDES, CAPTIONS AND UNCERTAINTY

- **Label in place** rather than with a legend, and **annotate** the one thing the reader should notice.
- A legend, if needed, never covers data. A colorbar or legend shared by panels is given once.
- **Axes**: quantity and unit in `[]`, with maths in LaTeX, on every figure:

| Quantity           | Label                                     |
| ------------------ | ----------------------------------------- |
| Temperature        | `r"Temperature [$^\circ\mathrm{C}$]"`     |
| Current            | `r"Zonal current [$\mathrm{m\,s^{-1}}$]"` |
| Wind stress        | `r"Wind stress [$\mathrm{N\,m^{-2}}$]"`   |
| Oxygen, nutrients  | `r"Oxygen [$\mathrm{mmol\,m^{-3}}$]"`     |
| Sea surface height | `r"Sea surface height [$\mathrm{m}$]"`    |

These are examples: take the true unit from the metadata of the dataset.

- **Caption**: what is shown, period, zone, depth, number of samples or members, what the colour limits are.
- **Show the uncertainty**: a mean alone hides the data. Show the points with the mean and its interval.
- **Always say what the error bars show**: the standard deviation is the spread of the data, the standard error the
  uncertainty on the mean. Print which one.
- **Ensembles**: show the spread. Many lines are fine when the spread is the message (spaghetti plot).
- Remove what no one reads (background, heavy grid, top and right frames), but keep light grid lines when values must
  be read off a scale.

---

## 6. MAPS

These are my own rules: they apply to **every figure showing a geographic zone**.

### The map of the zones

When a report works with geographic data, **one single figure** shows **all the zones named in the text**, before they
are used. Each zone is outlined and labelled in place, with the name used in the text. A new zone goes on this figure,
not on a second one.

### Each map

| Element  | Rule                                                                                       |
| -------- | ------------------------------------------------------------------------------------------ |
| Axes     | `Longitude [°E]`, `Latitude [°N]`                                                          |
| Units    | in `[]`, with maths in LaTeX (§ 5)                                                         |
| Colorbar | **exactly the size of the axis** it sits on: the height on the side, the width below       |
| Limits   | `vmin` and `vmax` are the **2% and 98% quantiles**, to keep contrast                       |
| Extreme  | **except** when showing an extreme event: then the full range, or the limits would hide it |
| Contours | thin black lines for the coasts, countries and lakes, to locate the plot                   |

And to stay honest with these limits:

- Compute them **ignoring the land** (`nanquantile`) and say them in the caption.
- The arrows at the ends of the colorbar show that values are clipped.
- Panels to compare **share their limits**, computed on all the panels, and a single colorbar spanning them.
- A signed field stays centred on zero: `vmax = max(|q2|, |q98|)`, `vmin = -vmax`.

---

## 7. ANTI-PATTERNS

| Do not                          | Because                                              | Instead                      |
| ------------------------------- | ---------------------------------------------------- | ---------------------------- |
| Pie chart                       | angles are hard to compare                           | bars, or two or three slices |
| 3D                              | it distorts perception, and adds nothing             | position, then colour        |
| Cut a bar from zero             | a bar encodes a length: the cut exaggerates          | bars from zero, or dots      |
| Channels out of scale with data | the drawn difference must be the true one            | channel proportional to data |
| Only summary statistics         | identical statistics can hide any shape (Datasaurus) | plot the rawest data         |
| Plot only a subset              | Challenger: the pattern was in the flights left out  | plot everything              |
| Publish an exploratory sweep    | 72 panels and no message                             | one question, one plot       |

---

## 8. CODE

`plot_map` and `plot_zones` follow the rules of § 6, for Heimdall and Neptune alike.

```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter


def color_limits(field: np.ndarray, extreme: bool = False, diverging: bool = False) -> tuple[float, float]:
    r"""Compute the limits of a colormap from the 2% and 98% quantiles of a field.

    Arguments:
        field     : Field or stack of panels sharing their limits, NaN on land.
        extreme   : Whether the field shows an extreme event (full range instead of the quantiles).
        diverging : Whether the field is signed (limits centred on zero).

    Returns:
        vmin : Lower limit of the colormap.
        vmax : Upper limit of the colormap.
    """
    vmin, vmax = np.nanquantile(field, [0, 1] if extreme else [0.02, 0.98])
    if diverging:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    return vmin, vmax


def add_geography(ax: plt.Axes, scale: str = "50m") -> None:
    r"""Add the contours and the longitude and latitude axes to a map.

    Arguments:
        ax    : Map axis, with the projection PlateCarree.
        scale : Resolution of Natural Earth ("10m", "50m" or "110m").
    """
    for feature in (cfeature.COASTLINE, cfeature.BORDERS, cfeature.LAKES):
        ax.add_feature(feature.with_scale(scale), facecolor="none", edgecolor="black", linewidth=0.4)

    gridlines = ax.gridlines(draw_labels=True, linewidth=0)
    gridlines.top_labels = gridlines.right_labels = False
    gridlines.xformatter = gridlines.yformatter = FormatStrFormatter("%g")

    fraction = dict(xycoords="axes fraction", textcoords="offset points", va="center")
    ax.annotate(r"Longitude [$^\circ$E]", (0.5, 0), (0, -30), ha="center", **fraction)
    ax.annotate(r"Latitude [$^\circ$N]", (0, 0.5), (-35, 0), ha="right", rotation=90, **fraction)


def plot_map(
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    label: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
) -> None:
    r"""Plot a field on a map, with a colorbar of the height of the axis.

    Arguments:
        ax    : Map axis, with the projection PlateCarree.
        lon   : Longitudes (X) [°E].
        lat   : Latitudes (Y) [°N].
        field : Field to plot, NaN on land (Y, X).
        label : Label of the colorbar, with the unit in brackets.
        vmin  : Lower limit of the colormap.
        vmax  : Upper limit of the colormap.
        cmap  : Name of the colormap.
    """
    image = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    add_geography(ax)

    # Bounds relative to the axis: same height for any projection or aspect ratio, arrows included.
    clipped = (np.nanmin(field) < vmin, np.nanmax(field) > vmax)
    extend = {(0, 0): "neither", (1, 0): "min", (0, 1): "max", (1, 1): "both"}[tuple(map(int, clipped))]
    ax.figure.colorbar(image, cax=ax.inset_axes([1.02, 0, 0.03, 1]), label=label, extend=extend)


def plot_zones(ax: plt.Axes, extent: list[float], zones: dict[str, list[float]]) -> None:
    r"""Plot the map of all the zones named in a report.

    Arguments:
        ax     : Map axis, with the projection PlateCarree.
        extent : Limits of the map [°E, °E, °N, °N].
        zones  : Name and limits of each zone [°E, °E, °N, °N].
    """
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    add_geography(ax)

    crs = ccrs.PlateCarree()
    for name, (west, east, south, north) in zones.items():
        ax.add_patch(Rectangle((west, south), east - west, north - south, fill=False, linewidth=1.5, transform=crs))
        ax.text((west + east) / 2, north, name, ha="center", va="bottom", weight="bold", transform=crs)
```

- Always create the figure with `constrained_layout=True`, and the axis with `projection=ccrs.PlateCarree()`.
- `cartopy` (and `cmocean`) go into `pyproject.toml` if they are missing from the environment.
- `cartopy` downloads the contours of Natural Earth on first use: if a compute node has no internet access, run it once
  on a login node.

---

## 9. BEFORE DELIVERING

- [ ] Can the message of each figure be said in one sentence? Is it its title?
- [ ] Is there a map of all the zones named in the text?
- [ ] Do all the axes and colorbars have a quantity and a unit in `[]`?
- [ ] Is each colorbar exactly the size of its axis?
- [ ] Are the limits the 2% and 98% quantiles, or the full range for an extreme event? Does the caption say so?
- [ ] Do the panels to compare share their scales?
- [ ] Is the colormap perceptually uniform, and centred if the field is signed?
- [ ] Would the figure still read without colour vision?
- [ ] Is the uncertainty shown, and is it said what the error bars are?
- [ ] Is anything left out of the data? Is it said?

---

## 10. SOURCES

| Source                                  | Link                                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------------- |
| Gilles Louppe, Lecture 3: Visualization | [dats0001](https://glouppe.github.io/dats0001-foundations-of-data-science/?p=lecture3.md#1) |
| Choosing colormaps in Matplotlib        | [matplotlib.org](https://matplotlib.org/stable/users/explain/colors/colormaps.html)         |
| Crameri et al. (2020), misuse of colour | [doi.org](https://doi.org/10.1038/s41467-020-19160-7)                                       |
| cmocean, colormaps for oceanography     | [matplotlib.org/cmocean](https://matplotlib.org/cmocean/)                                   |
| Cartopy                                 | [scitools.org.uk/cartopy](https://scitools.org.uk/cartopy/docs/latest/)                     |
