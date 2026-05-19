r"""Information about our dataset"""

__all__ = [
    "VARIABLES_CLIPPING",
    "DATASET_DATES_TRAINING",
    "DATASET_DATES_VALIDATION",
    "DATASET_DATES_TEST",
    "DATASET_REGION",
    "DATASET_VARIABLES_SURFACE",
    "DATASET_VARIABLES_OCEAN",
    "DATASET_VARIABLES",
    "Z",
    "C",
    "X",
    "Y",
    "C_IN",
    "C_OUT",
]

# fmt: off
#
# ----- Preprocessing
#
VARIABLES_CLIPPING = {
    "windsp":   (0, None),
    "vosaline": (0, None),
    "CHL":      (0, None),
    "DOX":      (0, None),
    "PHO":      (0, None),
    "SIO":      (0, None),
    "NOS":      (0, None),
}

# ----- Black Sea
#
DATASET_DATES_TRAINING   = ("1998-01-01", "2017-12-31")
DATASET_DATES_VALIDATION = ("2018-01-01", "2020-12-31")
DATASET_DATES_TEST       = ("2021-01-01", "2023-12-31")

DATASET_REGION = {
    "x":      slice(2, 578),
    "y":      slice(2, 258),
    "depthu": slice(0, 48),
    "depthv": slice(0, 48),
    "deptht": slice(0, 48),
}

DATASET_VARIABLES_SURFACE = [
    "windsp",
    "tauuo",
    "tauvo",
    "ssh",
]

DATASET_VARIABLES_OCEAN = [
    "uo",
    "vo",
    "votemper",
    "vosaline",
    "CHL",
    "DOX",
    "PAR",
    "PHO",
    "SIO",
    "NOS",
]

DATASET_VARIABLES = \
    DATASET_VARIABLES_SURFACE + DATASET_VARIABLES_OCEAN

# ----- Dimensions
#
# Global dimensions
Z = DATASET_REGION["deptht"].stop - DATASET_REGION["deptht"].start     # Depth levels
C = len(DATASET_VARIABLES_SURFACE) + len(DATASET_VARIABLES_OCEAN) * Z  # Aggregated levels
X = DATASET_REGION["x"].stop - DATASET_REGION["x"].start               # Longitudes
Y = DATASET_REGION["y"].stop - DATASET_REGION["y"].start               # Latitudes

# Autoencoder state
C_IN  = C + Z         # Variables + Mask
C_OUT = C             # Variables
