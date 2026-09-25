r"""Global paths and configuration helpers."""

__all__ = [
    "SIMULATION",
    "SIMULATION_DATA",
    "PATH_MAIN_SCRATCH",
    "PATH_MAIN_PROJECT",
    "PATH_NEP_PROJECT",
    "PATH_NEP_SCRATCH",
    "PATH_DATASETS",
    "PATH_MODELS",
    "PATH_PATHS",
    "PATH_GRID_U",
    "PATH_GRID_V",
    "PATH_GRID_W",
    "PATH_GRID_T",
    "PATH_PTRC",
    "PATH_BTRC",
    "PATH_STATS",
    "PATH_STATS_INCREMENTS",
    "PATH_MASK",
]

from pathlib import Path

# fmt: off
#
SIMULATION      = Path("/gpfs/projects/acad/bsmfc/nemo4.2.0/")
SIMULATION_DATA = SIMULATION / "BSFS_BIO" / "output_HR001"

# Scratch (wiping)
PATH_MAIN_SCRATCH = Path("/gpfs/scratch/acad/bsmfc/vmangele/")

# Project (non-wiping)
PATH_MAIN_PROJECT = Path("/gpfs/projects/acad/bsmfc/Obs/mastdb/vmangele/")

# ----- Main Folders
#
PATH_NEP_PROJECT = PATH_MAIN_PROJECT / "neptune"
PATH_NEP_SCRATCH = PATH_MAIN_SCRATCH / "neptune"

# ----- Subfolders
PATH_DATASETS = PATH_NEP_PROJECT / "datasets"
PATH_MODELS   = PATH_NEP_PROJECT / "models"
PATH_PATHS    = PATH_NEP_PROJECT / "paths"


# ----- Others
#
PATH_GRID_U = PATH_PATHS / "grid_U.txt"
PATH_GRID_V = PATH_PATHS / "grid_V.txt"
PATH_GRID_W = PATH_PATHS / "grid_W.txt"
PATH_GRID_T = PATH_PATHS / "grid_T.txt"
PATH_PTRC   = PATH_PATHS / "ptrc_T.txt"
PATH_BTRC   = PATH_PATHS / "btrc_T.txt"

PATH_STATS            = PATH_DATASETS / "statistics" / "black_sea_phys_bio_hr001_statistics_states.zarr"
PATH_STATS_INCREMENTS = PATH_DATASETS / "statistics" / "black_sea_phys_bio_hr001_statistics_increments.zarr"
PATH_MASK             = PATH_DATASETS / "structure"  / "black_sea_phys_bio_hr001_mask.zarr"
