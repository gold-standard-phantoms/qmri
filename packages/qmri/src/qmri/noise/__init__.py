"""Noise models for MRI simulation.

This module provides:

- Gaussian noise generation
- Rician noise generation
- SNR-based noise scaling

Example:
    ```python
    import numpy as np
    from qmri.noise import gaussian, rician

    signal = np.array([1000.0, 800.0, 600.0, 400.0])

    # Add Gaussian noise (appropriate for complex data)
    noisy_gaussian = gaussian.add_noise(signal, snr=50.0)

    # Add Rician noise (appropriate for magnitude data)
    noisy_rician = rician.add_noise(signal, snr=50.0)
    ```

Notes:
    For complex MRI data, use Gaussian noise. For magnitude-reconstructed
    images, use Rician noise which correctly models the non-Gaussian
    noise statistics.
"""

from qmri.noise import gaussian, rician
from qmri.noise.gaussian import calculate_sigma_from_snr

__all__: list[str] = [
    "gaussian",
    "rician",
    "calculate_sigma_from_snr",
]
