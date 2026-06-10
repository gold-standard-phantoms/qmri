"""Physical constants for MRI.

This module provides:

- Gyromagnetic ratios
- Tissue relaxation times at various field strengths
- Blood-brain partition coefficients
- Water diffusivity
"""

GAMMA_H: float = 42.577_478_92e6
"""Gyromagnetic ratio for ¹H (proton) in Hz/T."""

GAMMA_C13: float = 10.7084e6
"""Gyromagnetic ratio for ¹³C in Hz/T."""

GAMMA_F19: float = 40.052e6
"""Gyromagnetic ratio for ¹⁹F in Hz/T."""

GAMMA_P31: float = 17.235e6
"""Gyromagnetic ratio for ³¹P in Hz/T."""

GAMMA_NA23: float = 11.262e6
"""Gyromagnetic ratio for ²³Na in Hz/T."""

LAMBDA_BRAIN: float = 0.9
"""Blood-brain partition coefficient in ml/g."""

D_WATER_37C: float = 3.0e-3
"""Water diffusivity at 37°C in mm²/s."""

__all__ = [
    "GAMMA_H",
    "GAMMA_C13",
    "GAMMA_F19",
    "GAMMA_P31",
    "GAMMA_NA23",
    "LAMBDA_BRAIN",
    "D_WATER_37C",
]
