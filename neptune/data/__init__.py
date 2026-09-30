r"""Information about our dataset"""

__all__ = [
    "VARIABLES_CLIPPING",
    "VARIABLES_STANDARDIZATION",
    "VARIABLES_CONDITIONING_STANDARDIZATION",
    "DATASET_DATES_TRAINING",
    "DATASET_DATES_VALIDATION",
    "DATASET_DATES_TEST",
    "DATASET_REGION",
    "DATASET_VARIABLES_SURFACE",
    "DATASET_VARIABLES_OCEAN",
    "DATASET_VARIABLES",
    "DATASET_CONDITIONING",
    "Z",
    "X",
    "Y",
]

# fmt: off
#
# ----- Preprocessing
#
VARIABLES_CLIPPING = {
    "vosaline": (0, None),
    "CHL":      (0, None),
    "DOX":      (0, None),
    "PAR":      (0, None),
    "PHO":      (0, None),
    "SIO":      (0, None),
    "NOS":      (0, None),
}

VARIABLES_STANDARDIZATION = {
    "tauuo"    : "daily",
    "tauvo"    : "daily",
    "ssh"      : "daily",
    "uo"       : "global",
    "vo"       : "global",
    "votemper" : "daily",
    "vosaline" : "global",
    "CHL"      : "global",
    "DOX"      : "daily",
    "PAR"      : "daily",
    "PHO"      : "global",
    "SIO"      : "global",
    "NOS"      : "global",
    "default"  : "daily",
}

VARIABLES_CONDITIONING_STANDARDIZATION = {
    "t2m"      : "daily",
    "msdwswrf" : "daily",
    "msdwlwrf" : "daily",
    "u10"      : "daily",
    "v10"      : "daily",
    "si10"     : "daily",
    "default"  : "daily",
}

# ----- Black Sea
#
DATASET_DATES_TRAINING   = ("1998-01-01", "2017-12-31")
DATASET_DATES_VALIDATION = ("2018-01-01", "2020-12-31")
DATASET_DATES_TEST       = ("2021-01-01", "2023-12-31")

DATASET_REGION = {
    "x": slice(2, 578),
    "y": slice(2, 258),
    "z": slice(0, 48),
}

DATASET_CONDITIONING = [
    "t2m",
    "msdwswrf",
    "msdwlwrf",
    "u10",
    "v10",
    "si10",
]

DATASET_VARIABLES_SURFACE = [
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

DATASET_VARIABLES = DATASET_VARIABLES_SURFACE + DATASET_VARIABLES_OCEAN

# ----- Dimensions
#
X = DATASET_REGION["x"].stop - DATASET_REGION["x"].start  # Longitudes
Y = DATASET_REGION["y"].stop - DATASET_REGION["y"].start  # Latitudes
Z = DATASET_REGION["z"].stop - DATASET_REGION["z"].start  # Depth levels
