r"""Script to fix missing/empty NetCDF samples by copying adjacent day data."""

import argparse
import numpy as np
import os
import re
import xarray as xr
import yaml

from datetime import datetime, timedelta
from dawgz import job, schedule
from pathlib import Path

from neptune.config import SIMULATION_DATA

_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--config", "-c", type=str, required=True)
_early_args, _ = _early.parse_known_args()

_cfg = yaml.safe_load((Path(__file__).parent / _early_args.config).read_text())
_JOB_CFG = _cfg["fix_missing_sample"]

FILES_ISSUE = [
    [
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/1999/BS_1d_19990217_19990217_{datatype}_19990217-19990217.nc",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2000/BS_1d_20000121_20000121_{datatype}_20000121-20000121.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2000/BS_1d_20000602_20000602_{datatype}_20000602-20000602.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2001/BS_1d_20010802_20010802_{datatype}_20010802-20010802.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2003/BS_1d_20030616_20030616_{datatype}_20030616-20030616.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2004/BS_1d_20041124_20041124_{datatype}_20041124-20041124.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2006/BS_1d_20061116_20061116_{datatype}_20061116-20061116.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2007/BS_1d_20070407_20070407_{datatype}_20070407-20070407.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2007/BS_1d_20071019_20071019_{datatype}_20071019-20071019.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2009/BS_1d_20090418_20090418_{datatype}_20090418-20090418.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2010/BS_1d_20100112_20100112_{datatype}_20100112-20100112.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2014/BS_1d_20141004_20141004_{datatype}_20141004-20141004.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2015/BS_1d_20150405_20150405_{datatype}_20150405-20150405.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2016/BS_1d_20161225_20161225_{datatype}_20161225-20161225.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2017/BS_1d_20170223_20170223_{datatype}_20170223-20170223.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2018/BS_1d_20180915_20180915_{datatype}_20180915-20180915.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2020/BS_1d_20200112_20200112_{datatype}_20200112-20200112.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2021/BS_1d_20210714_20210714_{datatype}_20210714-20210714.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2021/BS_1d_20210727_20210727_{datatype}_20210727-20210727.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2021/BS_1d_20210829_20210829_{datatype}_20210829-20210829.nc4",
        f"/gpfs/projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/2021/BS_1d_20210906_20210906_{datatype}_20210906-20210906.nc4",
    ]
    for datatype in ["ptrc_T"]
]

ALL_FILES = [f for group in FILES_ISSUE for f in group]


def parse_date(file_path: str) -> datetime:
    r"""Extract the date from a NetCDF filename.

    Arguments:
        file_path: path to the NetCDF file.

    Returns:
        date: parsed date from the filename.
    """
    fname = os.path.basename(file_path)
    match = re.match(r"BS_1d_(\d{8})_", fname)
    if match is None:
        raise ValueError(f"Cannot parse date from filename: {fname}")
    return datetime.strptime(match.group(1), "%Y%m%d")


def adjacent_path(file_path: str, offset: int) -> str:
    r"""Return the path of the file for the day adjacent to the given file.

    Arguments:
        file_path: path to the missing/empty NetCDF file.
        offset: day offset, either -1 (previous day) or +1 (next day).

    Returns:
        path: path to the adjacent day's file.
    """
    fname = os.path.basename(file_path)
    match = re.match(r"BS_1d_(\d{8})_\d{8}_(.+)_\d{8}-\d{8}\.(nc4?)", fname)
    if match is None:
        raise ValueError(f"Unexpected filename format: {fname}")

    date_str, datatype, _ = match.groups()
    adj_date = datetime.strptime(date_str, "%Y%m%d") + timedelta(days=offset)
    adj_str = adj_date.strftime("%Y%m%d")
    stem = f"BS_1d_{adj_str}_{adj_str}_{datatype}_{adj_str}-{adj_str}"

    for ext in ("nc4", "nc"):
        path = os.path.join(SIMULATION_DATA, str(adj_date.year), f"{stem}.{ext}")
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"No adjacent file found for {fname} with offset={offset:+d}")


@job(**_JOB_CFG)
def fix_missing_sample(idx: int) -> None:
    r"""Fix a missing NetCDF sample by copying data from an adjacent day.

    Arguments:
        idx: index into ALL_FILES.
    """
    file_path = ALL_FILES[idx]

    try:
        src_path = adjacent_path(file_path, -1)
    except FileNotFoundError:
        src_path = adjacent_path(file_path, +1)

    with xr.open_dataset(src_path) as raw:
        dataset = raw.load()

    # Shift all temporal coordinates and bounds to match the target date
    delta = np.timedelta64(parse_date(file_path) - parse_date(src_path))
    time_coords = [
        c for c in ("time_instant", "time_counter", "time_centered") if c in dataset.coords
    ]
    dataset = dataset.assign_coords({c: dataset[c] + delta for c in time_coords})
    for v in ("time_instant_bounds", "time_counter_bounds", "time_centered_bounds"):
        if v in dataset:
            dataset[v] = dataset[v] + delta

    # Update global attributes
    src_str = parse_date(src_path).strftime("%Y%m%d")
    tgt_str = parse_date(file_path).strftime("%Y%m%d")
    if "name" in dataset.attrs:
        dataset.attrs["name"] = dataset.attrs["name"].replace(src_str, tgt_str)
    dataset.attrs["Fixed by"] = "Victor Mangeleer"

    fname = os.path.basename(file_path)
    output_fname = re.sub(r"\.nc4?$", "_fixed.nc", fname)
    output_path = os.path.join(os.path.dirname(file_path), output_fname)

    dataset.to_netcdf(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix missing NetCDF samples using adjacent day data."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML config file (relative to scripts/).",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster and 'async' for local execution.",
    )

    args = parser.parse_args()

    schedule(
        *[fix_missing_sample(i) for i in range(len(ALL_FILES))],
        name="NEPTUNE",
        backend=args.backend,
    )
