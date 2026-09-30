r"""Global constants for diagnostics."""

__all__ = [
    "TRANSLATIONS",
    "UNITS",
    "CMAPS_LINE",
    "CMAPS_FIELD",
    "CMAPS_METRICS",
]

import cmocean.cm as cmo_cm
import matplotlib.cm as plt_cm

# fmt: off
#
TRANSLATIONS = {
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

CMAPS_METRICS = {
    "rmse" : plt_cm.magma,
    "std"  : plt_cm.cividis,
}

CMAPS_LINE = {
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

CMAPS_FIELD = {
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
