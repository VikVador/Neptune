r"""Information about our dataset"""

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
    "x":      slice(0, 578),
    "y":      slice(0, 258),
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
    "PHO",
    "SIO",
    "NOS",
]

DATASET_VARIABLES = \
    DATASET_VARIABLES_SURFACE + DATASET_VARIABLES_OCEAN
