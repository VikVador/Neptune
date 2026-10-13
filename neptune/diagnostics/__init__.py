r"""Global constants for diagnostics."""

__all__ = [
    "TRANSLATIONS_VARS",
    "TRANSLATIONS_METRICS",
    "TRANSLATIONS_MODELS",
    "UNITS",
    "CMAPS_LINE_MODELS",
    "CMAPS_SPATIAL_VARS",
    "CMAPS_METRICS",
    "LAYERS",
    "SPLITS",
    "FIG_PROPERTIES",
]

import cmocean.cm as cmo_cm
import matplotlib.cm as plt_cm

from neptune.data import DATASET_DATES_TEST, DATASET_DATES_VALIDATION

# fmt: off
#
SPLITS = {
    "validation" : DATASET_DATES_VALIDATION,
    "test"       : DATASET_DATES_TEST,
}

# Depths [m] separating the layers of the ocean, each level belonging to the layer of its depth
LAYERS = [200]

# Font sizes of the figure titles, panel titles (and axis labels, legends), row labels, ticks and small texts,
# and quantile of a field giving the upper limit of the colors of the skill and spread
FIG_PROPERTIES = {
    "fs_suptitle"   : 13,
    "fs_title"      : 11,
    "fs_label"      : 10,
    "fs_tick"       : 9,
    "fs_text"       : 8,
    "quantile_vmax" : 0.97,
}

TRANSLATIONS_VARS = {
    "tauuo"    : "Downward X-Stress",
    "tauvo"    : "Downward Y-Stress",
    "ssh"      : "Sea Surface Height",
    "uo"       : "X-Velocity",
    "vo"       : "Y-Velocity",
    "votemper" : "Temperature",
    "vosaline" : "Salinity",
    "CHL"      : "Chlorophyll",
    "DOX"      : "Oxygen",
    "PAR"      : "Radiation",
    "PHO"      : "Phosphate",
    "SIO"      : "Silicic Acid",
    "NOS"      : "Nitrate",
}

TRANSLATIONS_METRICS = {
    "skill"  : "Skill",
    "spread" : "Spread",
    "ratio"  : "Spread-Skill Ratio",
}

TRANSLATIONS_MODELS = {
    "forecast"    : "Neptune",
    "persistence" : "Persistence",
    "climatology" : "Climatology",
}

UNITS = {
    "tauuo"    : "N/m^2",
    "tauvo"    : "N/m^2",
    "ssh"      : "m",
    "uo"       : "m/s",
    "vo"       : "m/s",
    "votemper" : "°C",
    "vosaline" : "PSU",
    "CHL"      : "mg/m^3",
    "DOX"      : "mmol/m^3",
    "PAR"      : "W/m^2",
    "PHO"      : "mmol/m^3",
    "SIO"      : "mmol/m^3",
    "NOS"      : "mmol/m^3",
}

CMAPS_LINE_MODELS = {
    "persistence" : "#808080",
    "climatology" : "#000000",
}

# Light for a null skill or spread and for a ratio of 1, darker away from them, per group of variables
CMAPS_METRICS = {
    "physique"     : {"skill": plt_cm.Reds,    "spread": plt_cm.Oranges, "ratio": plt_cm.RdBu},
    "biogeochimie" : {"skill": plt_cm.BuPu,    "spread": plt_cm.Purples, "ratio": plt_cm.RdBu},
}

CMAPS_SPATIAL_VARS = {
    "tauuo"    : plt_cm.BrBG_r,
    "tauvo"    : plt_cm.BrBG_r,
    "ssh"      : cmo_cm.diff,
    "uo"       : plt_cm.coolwarm,
    "vo"       : plt_cm.coolwarm,
    "votemper" : cmo_cm.thermal,
    "vosaline" : cmo_cm.amp,
    "CHL"      : cmo_cm.algae,
    "DOX"      : cmo_cm.haline,
    "PAR"      : cmo_cm.solar,
    "PHO"      : cmo_cm.rain,
    "SIO"      : cmo_cm.rain,
    "NOS"      : cmo_cm.rain,
}
