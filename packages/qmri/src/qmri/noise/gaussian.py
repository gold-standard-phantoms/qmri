"""Gaussian noise generation for MRI simulation.

This module provides functions for adding Gaussian noise to MRI signals,
which is the fundamental noise model for complex MRI data.

Example:
    ```python
    import numpy as np
    from qmri.noise import gaussian

    signal = np.array([1000.0, 800.0, 600.0, 400.0])
    noisy = gaussian.add_noise(signal, snr=20.0)
    ```

In MRI, thermal noise in the receiver coil follows a Gaussian distribution
in both the real and imaginary channels of complex data. For magnitude
images, see the Rician noise model in :mod:`qmri.noise.rician`.

References:
    .. [1] Gudbjartsson, H., & Patz, S. (1995). "The Rician distribution of
           noisy MRI data." Magnetic Resonance in Medicine, 34(6), 910-914.
"""

from typing import overload

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "add_noise",
    "calculate_sigma_from_snr",
]


def calculate_sigma_from_snr(
    signal: NDArray[np.floating] | float,
    snr: float,
) -> float:
    """Calculate noise standard deviation (sigma) from SNR.

    The SNR is defined as the mean signal divided by the noise
    standard deviation.

    Args:
        signal: Signal intensity or array of signal intensities. For arrays,
            the mean of non-zero values is used.
        snr: Target signal-to-noise ratio. Must be positive.

    Returns:
        Noise standard deviation (sigma).

    Raises:
        ValueError: If SNR is not positive.

    Example:
        ```python
        import numpy as np
        from qmri.noise.gaussian import calculate_sigma_from_snr

        sigma = calculate_sigma_from_snr(1000.0, snr=50.0)
        print(f"sigma = {sigma:.1f}")
        # sigma = 20.0
        ```
    """
    if snr <= 0:
        msg = f"SNR must be positive, got {snr}"
        raise ValueError(msg)

    signal_arr = np.asarray(signal)

    # Calculate mean signal from non-zero values
    if signal_arr.ndim == 0:
        mean_signal = float(signal_arr)
    else:
        nonzero_mask = signal_arr != 0
        if not np.any(nonzero_mask):
            return 0.0
        mean_signal = float(np.mean(np.abs(signal_arr[nonzero_mask])))

    return mean_signal / snr


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
    r"""Add Gaussian noise to a signal.

    Adds normally distributed random noise with mean 0 and standard
    deviation sigma. Either `snr` or `sigma` must be provided.

    Args:
        signal: Input signal to add noise to. Can be a scalar or array.
        snr: Target signal-to-noise ratio. The noise sigma is calculated as
            mean(signal) / snr. Mutually exclusive with `sigma`.
        sigma: Noise standard deviation. Mutually exclusive with `snr`.
        rng: Random number generator. If None, uses numpy's default RNG.

    Returns:
        Signal with added Gaussian noise. Same shape as input.

    Raises:
        ValueError: If neither or both of `snr` and `sigma` are provided.
            If `snr` is not positive.
            If `sigma` is negative.

    Example:
        Add noise with a specific SNR:

        ```python
        import numpy as np
        from qmri.noise import gaussian

        rng = np.random.default_rng(42)
        signal = np.array([1000.0, 800.0, 600.0, 400.0])
        noisy = gaussian.add_noise(signal, snr=50.0, rng=rng)

        # Add noise with a specific sigma:
        noisy = gaussian.add_noise(signal, sigma=20.0, rng=rng)
        ```

    The Gaussian noise model assumes:

    $$S_{noisy} = S + \epsilon$$

    where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

    For SNR-based noise, sigma is calculated as:

    $$\sigma = \frac{\bar{S}}{\text{SNR}}$$

    where $\bar{S}$ is the mean of non-zero signal values.
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

    # Generate and add noise
    noise = rng.normal(0.0, noise_sigma, signal_arr.shape)
    noisy_signal: NDArray[np.floating] = signal_arr + noise

    if is_scalar:
        return float(noisy_signal)
    return noisy_signal
