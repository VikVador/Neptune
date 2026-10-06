r"""Global constants for diagnostics."""

__all__ = [
    "TRANSLATIONS_VARS",
    "TRANSLATIONS_METRICS",
    "UNITS",
    "CMAPS_LINE_MODELS",
    "CMAPS_LINE_VARS",
    "CMAPS_SPATIAL_METRICS",
    "CMAPS_SPATIAL_VARS",
    "SEASONS",
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

SEASONS = {
    "winter" : (1,   2, 3),
    "spring" : (4,   5, 6),
    "summer" : (7,   8, 9),
    "autumn" : (10, 11, 12),
}

FIG_PROPERTIES = {
    "figsize"                   : (14, 18),
    "fs_title"                  : 12,
    "fs_x_label"                : 12,
    "fs_y_label"                : 12,
    "fs_x_tick"                 : 11,
    "fs_y_tick"                 : 11,
    "fs_sup_x_label"            : 14,
    "fs_sup_y_label"            : 14,
    "line_width"                : 2,
    "line_opacity"              : 1,
    "line_opacity_fill_between" : 0.25,
    "quantile_vmin"             : 0.02,
    "quantile_vmax"             : 0.98,
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

UNITS = {
    "tauuo"    : "[N/m^2]",
    "tauvo"    : "[N/m^2]",
    "ssh"      : "[m]",
    "uo"       : "[m/s]",
    "vo"       : "[m/s]",
    "votemper" : "[°C]",
    "vosaline" : "[PSU]",
    "CHL"      : "[mg/m^3]",
    "DOX"      : "[mmol/m^3]",
    "PAR"      : "[W/m^2]",
    "PHO"      : "[mmol/m^3]",
    "SIO"      : "[mmol/m^3]",
    "NOS"      : "[mmol/m^3]",
}

CMAPS_LINE_MODELS = {
    "forecast"    : "#4d68d7",
    "persistence" : "#808080",
    "climatology" : "#E67E22",
}

CMAPS_LINE_VARS = {
    "tauuo"    : "#cfa155",
    "tauvo"    : "#8fd3c8",
    "ssh"      : "#9C7AFF",
    "uo"       : "#4d68d7",
    "vo"       : "#b60d29",
    "votemper" : "#E67E22",
    "vosaline" : "#BF3F64",
    "CHL"      : "#40C88C",
    "DOX"      : "#0077B6",
    "PAR"      : "#d4b325",
    "PHO"      : "#88a47a",
    "SIO"      : "#06736d",
    "NOS"      : "#262e47",
}

CMAPS_SPATIAL_METRICS = {
    "skill"  : plt_cm.magma,
    "spread" : plt_cm.cividis,
    "ratio"  : plt_cm.RdBu_r,
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
