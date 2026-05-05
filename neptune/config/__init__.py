r"""Global paths and configuration helpers."""

from pathlib import Path

# fmt: off
#
SIMULATION = Path("/gpfs/projects/acad/bsmfc/nemo4.2.0/")

SIMULATION_DATA = SIMULATION / "BSFS_BIO" / "output_HR001"
SIMULATION_MASK = SIMULATION / "BSFS"     / "mesh_mask.nc_new59_CMCC_noAzov"

# Personnal
PATH_MAIN_LOCAL = Path("/gpfs/home/acad/ulg-mast/vmangele/")

# Scratch (wiping)
PATH_MAIN_SCRATCH = Path("/gpfs/scratch/acad/bsmfc/vmangele/")

# Project (non-wiping)
PATH_MAIN_PROJECT = Path("/gpfs/projects/acad/bsmfc/Obs/mastdb/vmangele/")

# ======================================
#             N E P T U N E
# ======================================
#
# ----- Main Folders
#
PATH_NEP_LOCAL   = PATH_MAIN_LOCAL   / "neptune"
PATH_NEP_PROJECT = PATH_MAIN_PROJECT / "neptune"
PATH_NEP_SCRATCH = PATH_MAIN_SCRATCH / "neptune"

# ----- Subfolders
PATH_DATASETS   = PATH_NEP_PROJECT / "datasets"
PATH_PATHS      = PATH_NEP_PROJECT / "paths"

# ----- Others
#
PATH_GRID_U  = PATH_PATHS / "grid_U.txt"
PATH_GRID_V  = PATH_PATHS / "grid_V.txt"
PATH_GRID_W  = PATH_PATHS / "grid_W.txt"
PATH_GRID_T  = PATH_PATHS / "grid_T.txt"
PATH_PTRC    = PATH_PATHS / "ptrc_T.txt"
PATH_BTRC    = PATH_PATHS / "btrc_T.txt"

PATH_STATS = PATH_DATASETS / "statistics" / "black_sea_phys_bio_hr001_statistics.zarr"
PATH_MASK  = PATH_DATASETS / "structure"  / "black_sea_phys_bio_hr001_mask.zarr"
