r"""Proton Resonance Frequency (PRF) thermometry.

This module provides functions for MR thermometry using the PRF shift method,
which exploits the temperature dependence of the water proton resonance
frequency.

Theory:
    The PRF shift method is based on the temperature-dependent chemical shift
    of water protons. The resonance frequency of water changes with temperature
    due to the breaking and reforming of hydrogen bonds, which affects the
    electron screening of water protons.

Signal Models:
    **Phase shift**:

    The temperature-induced phase shift between a baseline and heated state is:

    $$\Delta\phi = -\gamma \alpha B_0 \Delta T \cdot TE$$

    where:

    - $\gamma$ is the gyromagnetic ratio (Hz/T)
    - $\alpha$ is the PRF thermal coefficient (approximately -0.01 ppm/°C)
    - $B_0$ is the main magnetic field strength (T)
    - $\Delta T$ is the temperature change (°C)
    - $TE$ is the echo time (s)

    **Temperature from phase difference**:

    The temperature change is calculated from the phase difference:

    $$\Delta T = \frac{\Delta\phi}{\gamma \alpha B_0 \cdot TE}$$

    where $\Delta\phi = \phi_{heated} - \phi_{baseline}$ is the phase
    difference in radians.

References:
    .. [1] Ishihara, Y., et al. "A precise and fast temperature mapping using
           water proton chemical shift." MRM 34(6):814-823, 1995.
    .. [2] Rieke, V. and Butts Pauly, K. "MR thermometry." J Magn Reson Imaging
           27(2):376-390, 2008.
    .. [3] De Poorter, J., et al. "Noninvasive MRI thermometry with the proton
           resonance frequency method: study of susceptibility effects."
           MRM 34(3):359-367, 1995.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qmri.constants import GAMMA_H

__all__ = [
    "PRF_THERMAL_COEFFICIENT",
    "PRFResult",
    "signal_phase_shift",
    "calculate_temperature",
]

# PRF thermal coefficient (ppm/°C) - approximately -0.01 ppm/°C for water
# The negative sign indicates that increasing temperature decreases the
# resonance frequency (less hydrogen bonding, less electron screening)
PRF_THERMAL_COEFFICIENT: float = -0.01e-6  # converted to per °C (dimensionless)


@dataclass(frozen=True)
class PRFResult:
    """Result of PRF thermometry calculation.

    Attributes:
        temperature_change: Temperature change in degrees Celsius.
        phase_difference: Phase difference between heated and baseline
            images in radians.
    """

    temperature_change: NDArray[np.floating]
    phase_difference: NDArray[np.floating]


def signal_phase_shift(
    temperature_change: NDArray[np.floating] | float,
    echo_time: NDArray[np.floating] | float,
    magnetic_field: float,
    prf_coefficient: float = PRF_THERMAL_COEFFICIENT,
) -> NDArray[np.floating]:
    r"""Calculate temperature-induced phase shift.

    Implements the PRF phase shift equation:

    $$\Delta\phi = -\gamma \alpha B_0 \Delta T \cdot TE$$

    Note that the negative sign arises from the negative PRF thermal
    coefficient, which means increasing temperature causes a negative
    phase shift (frequency decrease).

    Args:
        temperature_change: Temperature change in degrees Celsius.
        echo_time: Echo time (TE) in seconds.
        magnetic_field: Main magnetic field strength (B0) in Tesla.
        prf_coefficient: PRF thermal coefficient in per degree Celsius
            (default -0.01e-6). The standard value for water is
            approximately -0.01 ppm/°C.

    Returns:
        Phase shift in radians.

    Example:
        ```python
        import numpy as np
        from qmri.thermometry import prf

        # Calculate phase shift for 10°C temperature increase at 3T
        phase_shift = prf.signal_phase_shift(
            temperature_change=10.0,
            echo_time=0.020,  # 20 ms
            magnetic_field=3.0,
        )
        print(f"Phase shift: {np.degrees(phase_shift):.2f} degrees")
        ```

    The phase shift is proportional to:

    - Temperature change (linear)
    - Echo time (longer TE gives larger phase shift but lower SNR)
    - Field strength (higher field gives larger phase shift)

    Typical phase shifts at 3T with TE=20ms are approximately 5-10 degrees
    per degree Celsius temperature change.
    """
    delta_t = np.asarray(temperature_change)
    te = np.asarray(echo_time)
    b0 = magnetic_field
    alpha = prf_coefficient

    # Convert gyromagnetic ratio to rad/s/T for phase calculation
    # gamma in Hz/T * 2*pi gives rad/s/T
    gamma_rad = 2 * np.pi * GAMMA_H

    # Phase shift: delta_phi = gamma * alpha * B0 * delta_T * TE
    # Note: alpha is already negative for water, so increasing T gives
    # negative phase shift
    result: NDArray[np.floating] = gamma_rad * alpha * b0 * delta_t * te
    return result


def calculate_temperature(
    phase_difference: NDArray[np.floating] | float,
    echo_time: NDArray[np.floating] | float,
    magnetic_field: float,
    baseline_temperature: NDArray[np.floating] | float = 37.0,
    prf_coefficient: float = PRF_THERMAL_COEFFICIENT,
) -> PRFResult:
    r"""Calculate temperature change from phase difference.

    Implements the inverse PRF equation:

    $$\Delta T = \frac{\Delta\phi}{\gamma \alpha B_0 \cdot TE}$$

    where the phase difference is $\phi_{heated} - \phi_{baseline}$.

    Args:
        phase_difference: Phase difference between heated and baseline
            images in radians. Calculated as phase_heated - phase_baseline.
        echo_time: Echo time (TE) in seconds.
        magnetic_field: Main magnetic field strength (B0) in Tesla.
        baseline_temperature: Baseline temperature in degrees Celsius
            (default 37.0). This is the reference temperature at which
            the baseline phase image was acquired. The absolute temperature
            is calculated as baseline_temperature + temperature_change.
        prf_coefficient: PRF thermal coefficient in per degree Celsius
            (default -0.01e-6). The standard value for water is
            approximately -0.01 ppm/°C.

    Returns:
        Result containing temperature change and phase difference.

    Example:
        ```python
        import numpy as np
        from qmri.thermometry import prf

        # Calculate temperature from measured phase difference
        result = prf.calculate_temperature(
            phase_difference=-0.16,  # radians (negative = heating)
            echo_time=0.020,  # 20 ms
            magnetic_field=3.0,
        )
        print(f"Temperature change: {result.temperature_change:.1f} °C")
        ```

    Important considerations for PRF thermometry:

    1. **Phase wrapping**: Phase values are typically wrapped to [-pi, pi].
       For large temperature changes, phase unwrapping may be required.

    2. **Reference tissue**: PRF thermometry provides relative temperature
       changes. A reference tissue (e.g., subcutaneous fat) with known
       temperature can be used for drift correction.

    3. **Fat signal**: Fat does not exhibit PRF shift with temperature.
       Fat suppression or water-fat separation is often used.

    4. **Motion**: Motion between baseline and heated acquisitions causes
       artefacts. Multi-baseline or referenceless methods can help.

    5. **Field drift**: B0 drift over time causes apparent temperature
       changes. Drift correction using reference regions is recommended.

    See Also:
        signal_phase_shift: Calculate phase shift from temperature change.
    """
    delta_phi = np.asarray(phase_difference)
    te = np.asarray(echo_time)
    b0 = magnetic_field
    alpha = prf_coefficient

    # Convert gyromagnetic ratio to rad/s/T
    gamma_rad = 2 * np.pi * GAMMA_H

    # Temperature change: delta_T = delta_phi / (gamma * alpha * B0 * TE)
    # Division by alpha (which is negative) will give correct sign
    denominator = gamma_rad * alpha * b0 * te

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        temperature_change: NDArray[np.floating] = np.where(
            np.abs(denominator) > 1e-20,
            delta_phi / denominator,
            np.nan,
        )

    return PRFResult(
        temperature_change=temperature_change,
        phase_difference=delta_phi,
    )
