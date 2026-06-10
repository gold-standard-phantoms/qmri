"""Rician noise generation for magnitude MRI simulation.

This module provides functions for adding Rician noise to magnitude MRI
signals, which is the appropriate noise model for magnitude-reconstructed
MRI data.

Example:
    ```python
    import numpy as np
    from qmri.noise import rician

    signal = np.array([1000.0, 800.0, 600.0, 400.0])
    noisy = rician.add_noise(signal, snr=20.0)
    ```

When complex MRI data (with Gaussian noise in real and imaginary channels)
is magnitude-reconstructed, the resulting noise follows a Rician distribution.
This is particularly important at low SNR, where the Rician distribution
causes a positive bias in the signal.

References:
    .. [1] Gudbjartsson, H., & Patz, S. (1995). "The Rician distribution of
           noisy MRI data." Magnetic Resonance in Medicine, 34(6), 910-914.
    .. [2] Henkelman, R.M. (1985). "Measurement of signal intensities in the
           presence of noise in MR images." Medical Physics, 12(2), 232-233.
"""

from typing import overload

import numpy as np
from numpy.typing import NDArray
from qmri.noise.gaussian import calculate_sigma_from_snr

__all__ = [
    "add_noise",
]


@overload
def add_noise(
    signal: float,
    *,
    snr: float | None = None,
    sigma: float | None = None,
    rng: np.random.Generator | None = None,
) -> float: ...


@overload
def add_noise(
    signal: NDArray[np.floating],
    *,
    snr: float | None = None,
    sigma: float | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]: ...


def add_noise(
    signal: NDArray[np.floating] | float,
    *,
    snr: float | None = None,
    sigma: float | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating] | float:
    r"""Add Rician noise to a magnitude signal.

    Simulates the noise in magnitude MRI images by adding Gaussian noise
    to both real and imaginary channels of complex data, then taking the
    magnitude.

    Args:
        signal: Input magnitude signal to add noise to. Can be a scalar or array.
            Must be non-negative.
        snr: Target signal-to-noise ratio. The noise sigma is calculated as
            mean(signal) / snr. Mutually exclusive with `sigma`.
        sigma: Noise standard deviation for each channel (real and imaginary).
            Mutually exclusive with `snr`.
        rng: Random number generator. If None, uses numpy's default RNG.

    Returns:
        Signal with added Rician noise. Same shape as input.

    Raises:
        ValueError: If neither or both of `snr` and `sigma` are provided.
            If `snr` is not positive.
            If `sigma` is negative.

    Example:
        Add noise with a specific SNR:

        ```python
        import numpy as np
        from qmri.noise import rician

        rng = np.random.default_rng(42)
        signal = np.array([1000.0, 800.0, 600.0, 400.0])
        noisy = rician.add_noise(signal, snr=50.0, rng=rng)

        # Add noise with a specific sigma:
        noisy = rician.add_noise(signal, sigma=20.0, rng=rng)
        ```

    The Rician distribution arises from taking the magnitude of complex
    data with Gaussian noise:

    $$S_{noisy} = \sqrt{(S + \epsilon_r)^2 + \epsilon_i^2}$$

    where $\epsilon_r, \epsilon_i \sim \mathcal{N}(0, \sigma^2)$
    are independent Gaussian noise in the real and imaginary channels.

    At high SNR, the Rician distribution approximates a Gaussian. At low
    SNR, the distribution becomes increasingly skewed with a positive bias.

    The probability density function is:

    $$p(x | \nu, \sigma) = \frac{x}{\sigma^2}
    \exp\left(-\frac{x^2 + \nu^2}{2\sigma^2}\right)
    I_0\left(\frac{x\nu}{\sigma^2}\right)$$

    where $\nu$ is the true signal, $\sigma$ is the noise
    standard deviation, and $I_0$ is the modified Bessel function
    of the first kind.
    """
    # Validate parameters
    if snr is None and sigma is None:
        msg = "Either 'snr' or 'sigma' must be provided"
        raise ValueError(msg)
    if snr is not None and sigma is not None:
        msg = "Only one of 'snr' or 'sigma' can be provided, not both"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    signal_arr = np.asarray(signal)
    is_scalar = signal_arr.ndim == 0

    # Calculate sigma from SNR if needed
    if snr is not None:
        noise_sigma = calculate_sigma_from_snr(signal_arr, snr)
    else:
        if sigma is not None and sigma < 0:
            msg = f"sigma must be non-negative, got {sigma}"
            raise ValueError(msg)
        noise_sigma = sigma if sigma is not None else 0.0

    # Handle zero noise case
    if noise_sigma == 0.0:
        return signal

    # Generate Rician noise by adding Gaussian noise to real and imaginary
    # channels of complex data, then taking magnitude
    # Assume input signal is purely real (magnitude image)
    noise_real = rng.normal(0.0, noise_sigma, signal_arr.shape)
    noise_imag = rng.normal(0.0, noise_sigma, signal_arr.shape)

    # Magnitude of complex signal with noise
    noisy_signal: NDArray[np.floating] = np.sqrt(
        (signal_arr + noise_real) ** 2 + noise_imag**2
    )

    if is_scalar:
        return float(noisy_signal)
    return noisy_signal
