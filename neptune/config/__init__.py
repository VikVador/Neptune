r"""Global paths and configuration helpers."""

from pathlib import Path

# fmt: off
#
# ----- Simulation
#
SIMULATION      = Path("/gpfs/projects/acad/bsmfc/nemo4.2.0/")
SIMULATION_DATA = SIMULATION / "BSFS_BIO" / "output_HR001"
SIMULATION_MASK = SIMULATION / "BSFS"     / "mesh_mask.nc_new59_CMCC_noAzov"

# ----- Main Folders
#
# Personnal
PATH_MAIN_LOCAL = Path("/gpfs/home/acad/ulg-mast/vmangele/")

# MAST-DB (non-wiping)
PATH_MAIN_PROJECT = Path("/gpfs/projects/acad/bsmfc/Obs/mastdb/vmangele/")

# Scratch (wiping)
PATH_MAIN_SCRATCH = Path("/gpfs/scratch/acad/bsmfc/vmangele/")

# ======================================
#           N E P T U N E
# ======================================
#
# ----- Main Folders
#
PATH_NEP_LOCAL   = PATH_MAIN_LOCAL   / "neptune"
PATH_NEP_PROJECT = PATH_MAIN_PROJECT / "neptune"
PATH_NEP_SCRATCH = PATH_MAIN_SCRATCH / "neptune"

# ----- Others
#
PATH_MODEL   = PATH_NEP_SCRATCH / "models"
PATH_GRID_U  = PATH_NEP_PROJECT / "paths"    / "grid_U.txt"
PATH_GRID_V  = PATH_NEP_PROJECT / "paths"    / "grid_V.txt"
PATH_GRID_W  = PATH_NEP_PROJECT / "paths"    / "grid_W.txt"
PATH_GRID_T  = PATH_NEP_PROJECT / "paths"    / "grid_T.txt"
PATH_PTRC    = PATH_NEP_PROJECT / "paths"    / "ptrc_T.txt"
PATH_BTRC    = PATH_NEP_PROJECT / "paths"    / "btrc_T.txt"
PATH_MESH    = PATH_NEP_PROJECT / "datasets" / "structure"  / "mesh_black_sea.zarr"
PATH_MASK_B  = PATH_NEP_PROJECT / "datasets" / "structure"  / "mask_black_sea.zarr"
PATH_MASK_V  = PATH_NEP_PROJECT / "datasets" / "structure"  / "mask_variables.zarr"
