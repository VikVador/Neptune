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
    "PATH_DIAGNOSTICS",
    "PATH_ANALYSIS",
    "PATH_GRID_U",
    "PATH_GRID_V",
    "PATH_GRID_W",
    "PATH_GRID_T",
    "PATH_PTRC",
    "PATH_BTRC",
    "PATH_STATS",
    "PATH_STATS_INCREMENTS",
    "PATH_STATS_CONDITIONING",
    "PATH_MASK",
    "PATH_MASK_NEMO",
    "PATH_CONDITIONING",
    "PATH_CLIMATOLOGY",
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
PATH_DATASETS    = PATH_NEP_PROJECT / "datasets"
PATH_MODELS      = PATH_NEP_PROJECT / "models"
PATH_PATHS       = PATH_NEP_PROJECT / "paths"
PATH_DIAGNOSTICS = PATH_NEP_SCRATCH / "diagnostics"
PATH_ANALYSIS    = Path(__file__).parents[2] / "experiments" / "analysis"

# ----- Others
#
PATH_GRID_U = PATH_PATHS / "grid_U.txt"
PATH_GRID_V = PATH_PATHS / "grid_V.txt"
PATH_GRID_W = PATH_PATHS / "grid_W.txt"
PATH_GRID_T = PATH_PATHS / "grid_T.txt"
PATH_PTRC   = PATH_PATHS / "ptrc_T.txt"
PATH_BTRC   = PATH_PATHS / "btrc_T.txt"

# Land/sea mask of NEMO, and the one used by Neptune, where the voxels with impossible values are land
PATH_MASK_NEMO = PATH_DATASETS / "structure" / "black_sea_phys_bio_hr001_mask.zarr"
PATH_MASK      = PATH_DATASETS / "structure" / "black_sea_phys_bio_hr001_mask_cleaned.zarr"

# Daily means of the ERA5 atmospheric forcing of NEMO, interpolated on its grid
PATH_CONDITIONING = PATH_DATASETS / "conditioning" / "black_sea_era5_hr001_conditioning.zarr"

# Daily climatology of the states over the training period
PATH_CLIMATOLOGY = PATH_DATASETS / "climatology" / "black_sea_phys_bio_hr001_climatology.zarr"

# Statistics of the cleaned states and of their daily increments, over the training period
PATH_STATS            = PATH_DATASETS / "statistics" / "black_sea_phys_bio_hr001_statistics_states_cleaned.zarr"
PATH_STATS_INCREMENTS = PATH_DATASETS / "statistics" / "black_sea_phys_bio_hr001_statistics_increments_cleaned.zarr"

# Statistics of the conditioning over the whole period (global), or averaged over those of each month or day
PATH_STATS_CONDITIONING = {
    "global"  : PATH_DATASETS / "statistics" / "black_sea_era5_hr001_statistics_conditioning.zarr",
    "monthly" : PATH_DATASETS / "statistics" / "black_sea_era5_hr001_statistics_conditioning_monthly.zarr",
    "daily"   : PATH_DATASETS / "statistics" / "black_sea_era5_hr001_statistics_conditioning_daily.zarr",
}
