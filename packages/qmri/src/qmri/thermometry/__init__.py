"""MR thermometry models.

This module provides functions for:

- Proton Resonance Frequency (PRF) thermometry
- Multi-echo dual-resonance thermometry (ethylene glycol phantoms)
"""

from qmri.thermometry.multiecho import (
    R_SQUARED_THRESHOLD,
    RANDOM_SEED,
    DfInitMethod,
    MultiEchoResult,
    RegionAnalysisMethod,
    RegionThermometryResult,
    calculate_df_from_temperature,
    calculate_temperature_from_df,
    calculate_temperature_uncertainty,
    fit_multiecho_thermometry,
    fit_multiecho_thermometry_image,
    lsq_fit_thermometry_signal_model,
    thermometry_signal_model,
)
from qmri.thermometry.prf import (
    PRF_THERMAL_COEFFICIENT,
    PRFResult,
    calculate_temperature,
    signal_phase_shift,
)

__all__ = [
    # PRF thermometry
    "PRF_THERMAL_COEFFICIENT",
    "PRFResult",
    "calculate_temperature",
    "signal_phase_shift",
    # Multi-echo thermometry
    "R_SQUARED_THRESHOLD",
    "RANDOM_SEED",
    "DfInitMethod",
    "MultiEchoResult",
    "RegionAnalysisMethod",
    "RegionThermometryResult",
    "calculate_df_from_temperature",
    "calculate_temperature_from_df",
    "calculate_temperature_uncertainty",
    "fit_multiecho_thermometry",
    "fit_multiecho_thermometry_image",
    "lsq_fit_thermometry_signal_model",
    "thermometry_signal_model",
]
