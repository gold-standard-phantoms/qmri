"""Diffusion-weighted imaging (DWI) signal models and fitting.

This module provides functions for:

- ADC (Apparent Diffusion Coefficient) fitting
- DWI signal generation
- B-value calibration

Example:
    ```python
    import numpy as np
    from qmri.diffusion import adc

    b_values = np.array([0, 500, 1000, 2000])
    signal = np.array([1000, 606, 368, 135])
    result = adc.fit(signal, b_values)
    print(f"ADC: {result.adc:.2e} mm²/s")
    ```
"""

from qmri.diffusion import adc
from qmri.diffusion.adc import ADCMapResult, ADCResult

__all__ = ["adc", "ADCResult", "ADCMapResult"]
