r"""Information about our dataset"""

__all__ = [
    "VARIABLES_CLIPPING",
    "VARIABLES_BOUNDS",
    "VARIABLES_LOG10",
    "VARIABLES_CONDITIONING_STANDARDIZATION",
    "DATASET_DATES_TRAINING",
    "DATASET_DATES_VALIDATION",
    "DATASET_DATES_TEST",
    "DATASET_DATES_EXCLUDED",
    "DATASET_REGION",
    "DATASET_VARIABLES_SURFACE",
    "DATASET_VARIABLES_OCEAN",
    "DATASET_VARIABLES_OCEAN_PHY",
    "DATASET_VARIABLES_OCEAN_BIO",
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

# Physical bounds, outside of which a value is impossible and becomes NaN (applied after clipping)
VARIABLES_BOUNDS = {
    "tauuo"    : (-3,   3),
    "tauvo"    : (-3,   3),
    "ssh"      : (-2,   2),
    "uo"       : (-3,   3),
    "vo"       : (-3,   3),
    "votemper" : (-2,   35),
    "vosaline" : (0,    39.9),
    "CHL"      : (0,    100),
    "DOX"      : (0,    600),
    "PAR"      : (0,    500),
    "PHO"      : (0,    10),
    "SIO"      : (0,    400),
    "NOS"      : (0,    400),
}

# Log-normal variables, given as log10(x + epsilon) with epsilon below their smallest meaningful values
VARIABLES_LOG10 = {
    "CHL" : 1e-5,
    "NOS" : 1e-7,
    "PHO" : 1e-4,
    "SIO" : 1e-3,
    "PAR" : 1e-6,
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
DATASET_DATES_TEST       = ("2021-01-01", "2022-12-31")

# Empty outputs of NEMO filled with a copy of a neighbouring day (the 2023 outputs are set aside, being recomputed)
DATASET_DATES_EXCLUDED = [
    "1999-02-17", "2000-01-21", "2000-06-02", "2001-08-02", "2003-06-16", "2004-11-24", "2006-11-16",
    "2007-04-07", "2007-10-19", "2009-04-18", "2010-01-12", "2014-10-04", "2015-04-05", "2016-12-25",
    "2017-02-23", "2018-09-15", "2020-01-12", "2021-07-14", "2021-07-27", "2021-08-29", "2021-09-06",
]

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

DATASET_VARIABLES_OCEAN_PHY = DATASET_VARIABLES_OCEAN[:4]
DATASET_VARIABLES_OCEAN_BIO = DATASET_VARIABLES_OCEAN[4:]
DATASET_VARIABLES           = DATASET_VARIABLES_SURFACE + DATASET_VARIABLES_OCEAN

# ----- Dimensions
#
X = DATASET_REGION["x"].stop - DATASET_REGION["x"].start  # Longitudes
Y = DATASET_REGION["y"].stop - DATASET_REGION["y"].start  # Latitudes
Z = DATASET_REGION["z"].stop - DATASET_REGION["z"].start  # Depth levels
