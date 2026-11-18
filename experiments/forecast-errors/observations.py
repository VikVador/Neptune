r"""Observations of the forecast errors, prepared once on the grid of the model: units, subsets and operators."""

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from errors import PATH_OBSERVATIONS, load_structure, reanalysis_paths

# fmt: off
#
# ----- Days of the comparison: from J-1 of the first run to J+9 of the last one
#
ORIGIN = "2025-01-01"
DAYS   = pd.date_range(ORIGIN, "2026-09-24")

# Gridded products: file, variable, model variable, offset to model units, whether finer than the model
PRODUCTS_GRIDDED = {
    "l4_sst_0.0625" : ("satellite/sst_l4_nrt_0.0625deg.nc",      "analysed_sst",                     "votemper", -273.15, False),
    "l4_sst_1km"    : ("satellite/sst_l4_nrt_0.01deg.nc",        "analysed_sst",                     "votemper", -273.15, True),
    "l4_chl_1km"    : ("satellite/chl_l4_my_gapfree_1km.nc",     "CHL",                              "CHL",       0.0,    True),
    "l4_sla_0.0625" : ("satellite/ssh_l4_nrt_0.0625deg.nc",      "adt",                              "ssh",       0.0,    False),
    "l4_sss_0.125"  : ("satellite/sss_l4_nrt_0.125deg.nc",       "sos",                              "vosaline",  0.0,    False),
    "l3_sst_0.0625" : ("satellite_l3/sst_l3s_nrt_0.0625deg.nc",  "adjusted_sea_surface_temperature", "votemper", -273.15, False),
    "l3_sst_1km"    : ("satellite_l3/sst_l3s_nrt_0.01deg.nc",    "adjusted_sea_surface_temperature", "votemper", -273.15, True),
    "l3_chl_1km"    : ("satellite_l3/chl_l3_my_1km.nc",          "CHL",                              "CHL",       0.0,    True),
    "l3_sss_smos_asc" : ("satellite_l3/sss_l3_smos_asc_0.2deg.nc", "Sea_Surface_Salinity",           "vosaline",  0.0,    False),
    "l3_sss_smos_des" : ("satellite_l3/sss_l3_smos_des_0.2deg.nc", "Sea_Surface_Salinity",           "vosaline",  0.0,    False),
}

# In situ variables: names in the files, per model variable, and their conversion to the model units
INSITU = {
    "votemper" : {"TEMP": 1.0},
    "vosaline" : {"PSAL": 1.0},
    "DOX"      : {"DOX2_ADJUSTED": "density", "DOXY": 1.0, "DOX1": 44.66},
    "CHL"      : {"CPHL_ADJUSTED": 1.0},
}

# Layers of the quality documents [m], and the largest distance of a coastal station to the sea of the model [°]
LAYERS  = [0, 10, 20, 30, 50, 75, 100, 200, 500, 1000]
COASTAL = 0.025
# fmt: on


def subsets_surface(cells: np.ndarray, regions: bool) -> tuple[list[str], np.ndarray]:
    r"""Subsets of units lying on model cells at the surface: basin, shelf, open sea and sub-regions.

    Arguments:
        cells   : Flat index (Y * X) of the model cell of each unit (N,).
        regions : Whether to add the 11 sub-regions of the chlorophyll quality metrics.

    Returns:
        names   : Names of the subsets.
        members : Membership of each unit (N, S).
    """

    structure = load_structure()
    shelf = structure["shelf"].ravel()[cells]
    names, members = ["all", "shelf", "open_sea"], [np.ones_like(shelf), shelf, ~shelf]
    if regions:
        region = structure["region"].ravel()[cells]
        names += [f"region_{r:02d}" for r in range(1, 12)]
        members += [region == r for r in range(1, 12)]

    return names, np.stack(members, axis=1)


def save(
    product: str, variable: str, anomaly: bool, names: list[str], units: dict, op: dict
) -> None:
    r"""Save a prepared product, its units sorted by day and its operator sorted by unit.

    Arguments:
        product  : Name of the product.
        variable : Model variable compared to the product.
        anomaly  : Whether the mean of each pass is removed (sea level).
        names    : Names of the subsets.
        units    : Arrays 'day', 'value', 'area', 'pass', 'subsets' of the units.
        op       : Arrays 'unit', 'cell', 'weight' of the operator, from the flat field of the model to the units.
    """

    order = np.argsort(units["day"], kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    unit = rank[op["unit"]]
    sort = np.argsort(unit, kind="stable")

    path = PATH_OBSERVATIONS / "prepared"
    path.mkdir(parents=True, exist_ok=True)
    np.savez(
        path / f"{product}.npz",
        variable     = variable,
        origin       = ORIGIN,
        anomaly      = anomaly,
        subsets      = np.array(names),
        unit_day     = units["day"][order].astype(np.int32),
        unit_value   = units["value"][order].astype(np.float64),
        unit_area    = units["area"][order].astype(np.float64),
        unit_pass    = units["pass"][order].astype(np.int64),
        unit_subsets = units["subsets"][order].astype(bool),
        op_unit      = unit[sort].astype(np.int64),
        op_cell      = op["cell"][sort].astype(np.int64),
        op_weight    = op["weight"][sort].astype(np.float64),
    )  # fmt: skip


def prepare_gridded(product: str) -> None:
    r"""Prepare a gridded satellite product on the model grid, comparing on the coarser of the two grids.

    Arguments:
        product : Name of the product, a key of PRODUCTS_GRIDDED.
    """

    file, name, variable, offset, fine = PRODUCTS_GRIDDED[product]
    structure = load_structure()
    lon, lat, sea, area = (
        structure["lon"],
        structure["lat"],
        structure["mask_t"][0],
        structure["area_t"],
    )
    step = lon[1] - lon[0]

    ds = xr.open_dataset(PATH_OBSERVATIONS / file)
    olon, olat = ds["longitude"].values, ds["latitude"].values
    ds = ds.sel(time=slice(DAYS[0], DAYS[-1] + pd.Timedelta(hours=23)))
    days = (pd.DatetimeIndex(ds["time"].values).normalize() - pd.Timestamp(ORIGIN)).days

    if fine:
        # Each observation pixel falls into one model cell, whose valid pixels are averaged
        i = np.rint((olon - lon[0]) / step).astype(int)
        j = np.rint((olat - lat[0]) / step).astype(int)
        jj, ii = np.meshgrid(j, i, indexing="ij")
        inside = (ii >= 0) & (ii < lon.size) & (jj >= 0) & (jj < lat.size)
        cell = np.where(
            inside, jj.clip(0, lat.size - 1) * lon.size + ii.clip(0, lon.size - 1), -1
        ).ravel()
        cell[cell >= 0] = np.where(sea.ravel()[cell[cell >= 0]], cell[cell >= 0], -1)
    else:
        # Each sea cell of the model falls into one observation pixel, whose cells are averaged by area
        ostep_x, ostep_y = olon[1] - olon[0], olat[1] - olat[0]
        jm, im = np.nonzero(sea)
        pi = np.rint((lon[im] - olon[0]) / ostep_x).astype(int)
        pj = np.rint((lat[jm] - olat[0]) / ostep_y).astype(int)
        keep = (pi >= 0) & (pi < olon.size) & (pj >= 0) & (pj < olat.size)
        cells, pixel = (jm * lon.size + im)[keep], (pj * olon.size + pi)[keep]
        counts = np.bincount(pixel, minlength=olon.size * olat.size)
        covered = counts >= 0.5 * (ostep_x / step) * (ostep_y / step)
        weight = area.ravel()[cells]
        pixel_area = np.bincount(pixel, weights=weight, minlength=counts.size)
        centre = np.zeros(counts.size, dtype=int)
        centre[pixel] = cells

    units = {k: [] for k in ("day", "value", "area", "pass", "subsets")}
    op = {k: [] for k in ("unit", "cell", "weight")}
    n = 0
    for t, day in enumerate(days):
        x = np.asarray(ds[name].isel(time=t).values, dtype=np.float64).squeeze().ravel() + offset
        if fine:
            ok = (cell >= 0) & np.isfinite(x)
            counts = np.bincount(cell[ok], minlength=sea.size)
            sums = np.bincount(cell[ok], weights=x[ok], minlength=sea.size)
            cells = np.flatnonzero(counts)
            values, unit_area, centres = sums[cells] / counts[cells], area.ravel()[cells], cells
            op_unit, op_cell, op_weight = n + np.arange(cells.size), cells, np.ones(cells.size)
        else:
            pixels = np.flatnonzero(covered & np.isfinite(x))
            values, unit_area, centres = x[pixels], pixel_area[pixels], centre[pixels]
            rank = np.full(counts.size, -1)
            rank[pixels] = np.arange(pixels.size)
            used = rank[pixel] >= 0
            op_unit = n + rank[pixel][used]
            op_cell, op_weight = cells[used], weight[used] / pixel_area[pixel][used]

        names, members = subsets_surface(centres, regions=variable == "CHL")
        units["day"].append(np.full(values.size, day)), units["value"].append(values)
        units["area"].append(unit_area), units["pass"].append(np.zeros(values.size, int))
        units["subsets"].append(members)
        op["unit"].append(op_unit), op["cell"].append(op_cell), op["weight"].append(op_weight)
        n += values.size

    save(
        product,
        variable,
        variable == "ssh",
        names,
        *({k: np.concatenate(v) for k, v in d.items()} for d in (units, op)),
    )


def prepare_alongtrack() -> None:
    r"""Prepare the along-track sea level of all satellites: super-observations per track and model cell."""

    structure = load_structure()
    lon, lat, sea, area = (
        structure["lon"],
        structure["lat"],
        structure["mask_t"][0],
        structure["area_t"],
    )
    step = lon[1] - lon[0]

    tables = []
    for path in sorted((PATH_OBSERVATIONS / "satellite_l3" / "sla_alongtrack").glob("*.csv")):
        df = pd.read_csv(
            path, usecols=["variable", "platform_id", "time", "longitude", "latitude", "value"]
        )
        df = df.pivot_table(
            index=["platform_id", "time", "longitude", "latitude"],
            columns="variable",
            values="value",
        ).reset_index()
        tables.append(df)
    df = pd.concat(tables).dropna(subset=["sla_filtered", "mdt"])

    # Absolute dynamic topography, compared to the model sea level once the mean of each track is removed
    df["value"] = df["sla_filtered"] + df["mdt"]
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df["day"] = (df["time"].dt.normalize() - pd.Timestamp(ORIGIN)).dt.days
    df["i"] = np.rint((df["longitude"] - lon[0]) / step).astype(int)
    df["j"] = np.rint((df["latitude"] - lat[0]) / step).astype(int)
    df = df[
        (df["day"] >= 0)
        & (df["day"] < len(DAYS))
        & df["i"].between(0, lon.size - 1)
        & df["j"].between(0, lat.size - 1)
    ]
    df["cell"] = df["j"] * lon.size + df["i"]
    df = df[sea.ravel()[df["cell"].to_numpy()]]

    # A track is a run of points of one satellite without a gap of more than 10 minutes
    df = df.sort_values(["platform_id", "time"])
    gap = df.groupby("platform_id")["time"].diff() > pd.Timedelta(minutes=10)
    df["track"] = gap.cumsum()
    units = df.groupby(["day", "track", "cell"], as_index=False)["value"].mean()
    units["pass"] = units.groupby("day")["track"].rank(method="dense").astype(int) - 1

    cells = units["cell"].to_numpy()
    names, members = subsets_surface(cells, regions=False)
    save(
        "l3_sla_alongtrack", "ssh", True, names,
        {"day": units["day"].to_numpy(), "value": units["value"].to_numpy(), "area": area.ravel()[cells],
         "pass": units["pass"].to_numpy(), "subsets": members},
        {"unit": np.arange(cells.size), "cell": cells, "weight": np.ones(cells.size)},
    )  # fmt: skip


def read_insitu() -> pd.DataFrame:
    r"""Read the good in situ measurements of the floats and moorings, in model units except oxygen of floats.

    Returns:
        records : One row per measurement: platform, kind, time, position, depth, variable, value, conversion.
    """

    rows = []
    for path in sorted((PATH_OBSERVATIONS / "insitu").rglob("*.nc")):
        kind = {"PF": "profiler", "MO": "mooring"}.get(path.parent.parent.name)
        if kind is None:
            continue
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(True)
            time = pd.to_datetime(
                np.ma.filled(ds["TIME"][:], np.nan), unit="D", origin="1950-01-01"
            )
            nt = time.size
            good_time = qc_good(ds, "TIME_QC", (nt,))
            good_pos = qc_good(ds, "POSITION_QC", (nt,))
            lon = np.broadcast_to(np.ma.filled(ds["LONGITUDE"][:], np.nan), (nt,))
            lat = np.broadcast_to(np.ma.filled(ds["LATITUDE"][:], np.nan), (nt,))
            if "DEPH" in ds.variables:
                depth = np.ma.filled(ds["DEPH"][:].astype(np.float64), np.nan)
                good_depth = qc_good(ds, "DEPH_QC", depth.shape)
            else:
                pressure = np.ma.filled(ds["PRES"][:].astype(np.float64), np.nan)
                depth = pressure_to_depth(pressure, lat[:, None])
                good_depth = qc_good(ds, "PRES_QC", pressure.shape)
            depth = np.broadcast_to(depth, (nt, ds.dimensions["DEPTH"].size))
            good_depth = np.broadcast_to(good_depth, depth.shape)

            for variable, sources in INSITU.items():
                for source, factor in sources.items():
                    if source not in ds.variables:
                        continue
                    value = np.ma.filled(ds[source][:].astype(np.float64), np.nan).reshape(
                        depth.shape
                    )
                    good = (
                        qc_good(ds, f"{source}_QC", value.shape)
                        & good_depth
                        & (good_time & good_pos)[:, None]
                    )
                    good &= np.isfinite(value) & np.isfinite(depth)
                    t, k = np.nonzero(good)
                    rows.append(pd.DataFrame({
                        "platform" : path.stem.rsplit("_", 1)[0],
                        "kind"     : kind,
                        "time"     : time[t],
                        "lon"      : lon[t],
                        "lat"      : lat[t],
                        "depth"    : depth[t, k],
                        "variable" : variable,
                        "value"    : value[t, k] * (1.0 if factor == "density" else factor),
                        "density"  : factor == "density",
                    }))  # fmt: skip
                    break

    return pd.concat(rows).drop_duplicates(["platform", "time", "depth", "variable"])


def qc_good(ds: netCDF4.Dataset, name: str, shape: tuple[int, ...]) -> np.ndarray:
    r"""Whether the quality flags of a variable are good (1) or probably good (2).

    Arguments:
        ds    : Opened in situ file.
        name  : Name of the flag variable.
        shape : Shape of the values it flags.

    Returns:
        good : Boolean array of that shape, true where the flag is missing from the file.
    """

    if name not in ds.variables:
        return np.ones(shape, dtype=bool)
    flags = np.ma.filled(ds[name][:], 0).astype(int)

    return np.broadcast_to(
        np.isin(flags, (1, 2)).reshape(flags.shape + (1,) * (len(shape) - flags.ndim)), shape
    )


def pressure_to_depth(pressure: np.ndarray, lat: np.ndarray) -> np.ndarray:
    r"""Convert sea pressure to depth with the formula of Saunders (1981).

    Arguments:
        pressure : Sea pressure [dbar].
        lat      : Latitude [°N].

    Returns:
        depth : Depth [m].
    """

    c1 = (5.92 + 5.25 * np.sin(np.radians(lat)) ** 2) * 1e-3

    return (1 - c1) * pressure - 2.21e-6 * pressure**2


def prepare_insitu() -> None:
    r"""Prepare the in situ products: super-observations per day, platform, model cell and level, with their operator."""

    structure = load_structure()
    lon, lat, depth, sea = (
        structure["lon"],
        structure["lat"],
        structure["depth"],
        structure["mask_t"],
    )
    step = lon[1] - lon[0]
    nz, ny, nx = sea.shape

    df = read_insitu()
    df["day"] = (df["time"].dt.normalize() - pd.Timestamp(ORIGIN)).dt.days
    df["i"] = np.rint((df["lon"] - lon[0]) / step).astype(int)
    df["j"] = np.rint((df["lat"] - lat[0]) / step).astype(int)
    df["k"] = np.abs(df["depth"].to_numpy()[:, None] - depth[None]).argmin(axis=1)
    df = df[
        (df["day"] >= 0)
        & (df["day"] < len(DAYS))
        & df["i"].between(0, nx - 1)
        & df["j"].between(0, ny - 1)
    ]

    # Super-observations: the mean of the measurements of a platform in a model cell and level, on one day
    keys = ["variable", "platform", "kind", "density", "day", "j", "i", "k"]
    units = df.groupby(keys, as_index=False)[["value", "lon", "lat", "depth"]].mean()

    # Bilinear weights among the sea cells around each unit, then linear in depth between two levels
    x, y = (units["lon"].to_numpy() - lon[0]) / step, (units["lat"].to_numpy() - lat[0]) / step
    x0, y0 = np.floor(x).astype(int).clip(0, nx - 2), np.floor(y).astype(int).clip(0, ny - 2)
    fx, fy = x - x0, y - y0
    d = units["depth"].to_numpy().clip(depth[0], None)
    k0 = (np.searchsorted(depth, d) - 1).clip(0, nz - 2)
    fz = ((d - depth[k0]) / (depth[k0 + 1] - depth[k0])).clip(0, 1)

    corners = [
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ]
    op_unit, op_cell, op_weight = [], [], []
    valid, coastal = np.ones(len(units), bool), np.zeros(len(units), bool)
    for k, wz in ((k0, 1 - fz), (k0 + 1, fz)):
        w = np.stack([wc * sea[k, y0 + dj, x0 + di] for di, dj, wc in corners], axis=1)
        total = w.sum(axis=1)
        used = wz > 0
        coastal |= (k == k0) & (total < 1 - 1e-9)
        for c, (di, dj, _) in enumerate(corners):
            keep = used & (w[:, c] > 0)
            op_unit.append(np.flatnonzero(keep))
            op_cell.append(((k * ny + y0 + dj) * nx + x0 + di)[keep])
            op_weight.append((wz * w[:, c] / np.where(total > 0, total, 1))[keep])

        # Without a sea corner, a coastal unit takes the nearest sea cell within one model cell, else is rejected
        todo = used & (total == 0)
        jn, in_, found = nearest_sea(units["j"].to_numpy(), units["i"].to_numpy(), k, sea, todo)
        valid &= ~todo | found
        keep = todo & found
        op_unit.append(np.flatnonzero(keep))
        op_cell.append(((k * ny + jn) * nx + in_)[keep])
        op_weight.append(wz[keep])

    op_unit, op_cell, op_weight = (np.concatenate(a) for a in (op_unit, op_cell, op_weight))
    units["coastal"] = coastal

    # Oxygen of the floats, per kilogram, is converted with the density of the reanalysis on the same day
    density, days = units["density"].to_numpy() & valid, units["day"].to_numpy()
    for day in np.unique(days[density]):
        sel = density & (days == day)
        member = sel[op_unit]
        rho = read_density(str((pd.Timestamp(ORIGIN) + pd.Timedelta(days=int(day))).date()))
        rho = np.bincount(
            op_unit[member],
            weights=op_weight[member] * rho.ravel()[op_cell[member]],
            minlength=len(units),
        )
        units.loc[sel, "value"] *= rho[sel] / 1000.0

    for variable in INSITU:
        sel = np.flatnonzero((units["variable"].to_numpy() == variable) & valid)
        table = units.iloc[sel]
        rank = np.full(len(units), -1)
        rank[sel] = np.arange(sel.size)
        member = rank[op_unit] >= 0

        cells = table["j"].to_numpy() * nx + table["i"].to_numpy()
        names, surface = subsets_surface(cells, regions=False)
        depths = table["depth"].to_numpy()
        layers = [
            (depths >= a) & (depths < b) for a, b in zip(LAYERS[:-1], LAYERS[1:], strict=False)
        ]
        names += [f"layer_{a}-{b}m" for a, b in zip(LAYERS[:-1], LAYERS[1:], strict=False)] + [
            "profiler",
            "mooring",
            "coastal",
        ]
        members = np.concatenate([
            surface,
            np.stack(layers, axis=1),
            np.stack([table["kind"].to_numpy() == "profiler", table["kind"].to_numpy() == "mooring", table["coastal"].to_numpy()], axis=1),
        ], axis=1)  # fmt: skip

        save(
            f"insitu_{variable}", variable, False, names,
            {"day": table["day"].to_numpy(), "value": table["value"].to_numpy(), "area": np.ones(sel.size),
             "pass": np.zeros(sel.size, int), "subsets": members},
            {"unit": rank[op_unit[member]], "cell": op_cell[member], "weight": op_weight[member]},
        )  # fmt: skip


def nearest_sea(
    j: np.ndarray, i: np.ndarray, k: np.ndarray, sea: np.ndarray, todo: np.ndarray
) -> tuple[np.ndarray, ...]:
    r"""Find the nearest sea cell within one cell of a model point, at a given level.

    Arguments:
        j, i : Nearest model row and column of each unit (N,).
        k    : Level of each unit (N,).
        sea  : Sea mask of the model (Z, Y, X).
        todo : Units to search for (N,).

    Returns:
        j, i  : Row and column of the nearest sea cell (N,).
        found : Whether a sea cell was found (N,).
    """

    nz, ny, nx = sea.shape
    jn, in_, found = j.copy(), i.copy(), np.zeros(j.size, bool)
    for dj, di in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        jj, ii = (j + dj).clip(0, ny - 1), (i + di).clip(0, nx - 1)
        hit = todo & ~found & (k < nz) & sea[k.clip(0, nz - 1), jj, ii]
        jn[hit], in_[hit], found[hit] = jj[hit], ii[hit], True

    return jn, in_, found


def read_density(date: str) -> np.ndarray:
    r"""Read the potential density of the interim reanalysis of a date.

    Arguments:
        date : Date string 'YYYY-MM-DD'.

    Returns:
        rho : Potential density, NaN on land (Z, Y, X) [kg/m3].
    """

    with netCDF4.Dataset(reanalysis_paths()[date]["grid_T"]) as ds:
        rho = np.ma.filled(ds["rho"][0].astype(np.float64), np.nan)

    return np.where(load_structure()["mask_t"], rho, np.nan)
