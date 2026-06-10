"""DWI phantom generation for ADC validation.

This module provides functions for generating synthetic diffusion-weighted
imaging (DWI) data with known ground truth ADC values.

Example:
    ```python
    from qmri.dro import dwi
    from qmri.diffusion import adc

    # Generate a single-voxel phantom
    phantom = dwi.generate(adc=1e-3, s0=1000, snr=50, seed=42)

    # Fit and compare to ground truth
    result = adc.fit(phantom.signal, phantom.b_values)
    print(f"True ADC: {phantom.ground_truth['adc'].value:.2e}")
    print(f"Fitted ADC: {result.adc:.2e}")
    ```
"""

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from qmri.diffusion import adc as adc_module
from qmri.dro._types import DWIPhantom, GroundTruth
from qmri.noise import gaussian, rician

__all__ = [
    "generate",
    "generate_calibration_phantom",
]


@overload
def generate(
    adc: float,
    s0: float = ...,
    b_values: Sequence[float] = ...,
    *,
    snr: float | None = ...,
    noise_model: Literal["rician", "gaussian"] = ...,
    seed: int | None = ...,
) -> DWIPhantom: ...


@overload
def generate(
    adc: NDArray[np.floating],
    s0: NDArray[np.floating] | float = ...,
    b_values: Sequence[float] = ...,
    *,
    snr: float | None = ...,
    noise_model: Literal["rician", "gaussian"] = ...,
    seed: int | None = ...,
) -> DWIPhantom: ...


def generate(
    adc: NDArray[np.floating] | float,
    s0: NDArray[np.floating] | float = 1000.0,
    b_values: Sequence[float] = (0, 500, 1000, 2000),
    *,
    snr: float | None = None,
    noise_model: Literal["rician", "gaussian"] = "rician",
    seed: int | None = None,
) -> DWIPhantom:
    r"""Generate a DWI phantom with known ADC.

    Creates synthetic DWI signal using the mono-exponential diffusion model
    with optional noise for validation and testing purposes.

    Args:
        adc: Apparent Diffusion Coefficient in mm²/s. Can be a scalar for
            single-voxel phantom or an array for multi-voxel phantom.
        s0: Baseline signal intensity (at b=0). Can be scalar or array
            matching the shape of `adc`. Default is 1000.0.
        b_values: Diffusion weighting values in s/mm². Default is
            (0, 500, 1000, 2000).
        snr: Signal-to-noise ratio. If None, no noise is added.
        noise_model: Type of noise to add ("rician" or "gaussian").
            Default is "rician", which is more realistic for magnitude MRI.
        seed: Random seed for reproducibility. If None, uses random state.

    Returns:
        DWIPhantom containing the synthetic signal, b-values, and ground truth.

    Raises:
        ValueError: If noise_model is not "rician" or "gaussian".

    Example:
        Single-voxel phantom:

        ```python
        from qmri.dro import dwi

        phantom = dwi.generate(adc=1e-3, snr=50, seed=42)
        print(phantom.signal)
        # array([1000.  , 606.5..., 367.8..., 135.3...])
        ```

        Multi-voxel phantom:

        ```python
        import numpy as np
        from qmri.dro import dwi

        adc_map = np.array([[0.5e-3, 1.0e-3],
                           [1.5e-3, 2.0e-3]])
        phantom = dwi.generate(adc=adc_map, snr=100, seed=42)
        print(phantom.signal.shape)
        # (2, 2, 4)
        ```

    Notes:
        The signal is generated using the Stejskal-Tanner equation:

        $$S(b) = S_0 \\exp(-b \\cdot \\text{ADC})$$

        For magnitude MRI, Rician noise is the appropriate model and causes
        a positive bias at low SNR. Gaussian noise is suitable for complex
        data or high SNR scenarios.
    """
    if noise_model not in ("rician", "gaussian"):
        msg = f"noise_model must be 'rician' or 'gaussian', got '{noise_model}'"
        raise ValueError(msg)

    b_arr = np.asarray(b_values, dtype=np.float64)
    adc_arr = np.asarray(adc, dtype=np.float64)
    s0_arr = np.asarray(s0, dtype=np.float64)

    # Generate clean signal
    # For multi-voxel, we need to handle broadcasting carefully
    if adc_arr.ndim == 0:
        # Single voxel case
        signal = adc_module.signal_model(s0_arr, adc_arr, b_arr)
    else:
        # Multi-voxel case: output shape is (*adc.shape, n_bvalues)
        # Broadcast s0 to match adc shape
        s0_bc = np.broadcast_to(s0_arr, adc_arr.shape)

        # Generate signal for each b-value
        signal = np.zeros((*adc_arr.shape, len(b_arr)), dtype=np.float64)
        for i, b in enumerate(b_arr):
            signal[..., i] = s0_bc * np.exp(-b * adc_arr)

    # Add noise if requested
    if snr is not None:
        rng = np.random.default_rng(seed)
        if noise_model == "rician":
            signal = rician.add_noise(signal, snr=snr, rng=rng)
        else:
            signal = gaussian.add_noise(signal, snr=snr, rng=rng)

    # Build ground truth dictionary
    ground_truth: dict[str, GroundTruth[float | NDArray[np.floating]]] = {
        "adc": GroundTruth(
            value=float(adc) if adc_arr.ndim == 0 else adc_arr,
            units="mm²/s",
            description="Apparent Diffusion Coefficient",
        ),
        "s0": GroundTruth(
            value=float(s0) if np.asarray(s0).ndim == 0 else s0_arr,
            units="a.u.",
            description="Baseline signal intensity (b=0)",
        ),
    }

    return DWIPhantom(
        signal=signal,
        b_values=b_arr,
        ground_truth=ground_truth,
        snr=snr,
        seed=seed,
    )


def generate_calibration_phantom(
    adc_values: Sequence[float] = (0.3e-3, 0.7e-3, 1.0e-3, 1.5e-3, 2.0e-3, 3.0e-3),
    b_values: Sequence[float] = (0, 50, 100, 200, 400, 600, 800, 1000),
    *,
    s0: float = 1000.0,
    snr: float | None = 50.0,
    noise_model: Literal["rician", "gaussian"] = "rician",
    seed: int | None = None,
) -> DWIPhantom:
    """Generate a calibration phantom with multiple ADC values.

    Creates a phantom with multiple voxels at different ADC values,
    suitable for method validation and benchmarking.

    Args:
        adc_values: Sequence of ADC values in mm²/s. Default covers the
            range from restricted diffusion to free water.
        b_values: Diffusion weighting values in s/mm². Default provides
            good sampling for ADC range.
        s0: Baseline signal intensity for all voxels. Default is 1000.0.
        snr: Signal-to-noise ratio. Default is 50.0.
        noise_model: Type of noise to add. Default is "rician".
        seed: Random seed for reproducibility.

    Returns:
        DWIPhantom with shape (n_adc_values, n_bvalues).

    Example:
        ```python
        from qmri.dro import dwi
        from qmri.diffusion import adc

        # Generate calibration phantom
        phantom = dwi.generate_calibration_phantom(seed=42)
        print(f"Shape: {phantom.signal.shape}")
        # Shape: (6, 8)

        # Fit each "voxel"
        for i, true_adc in enumerate(phantom.ground_truth['adc'].value):
            result = adc.fit(phantom.signal[i], phantom.b_values)
            print(f"True: {true_adc:.2e}, Fitted: {result.adc:.2e}")
        ```

    Notes:
        Default ADC values represent:

        - 0.3e-3: Highly restricted (e.g., tumour)
        - 0.7e-3: White matter
        - 1.0e-3: Grey matter
        - 1.5e-3: Less restricted tissue
        - 2.0e-3: CSF-like
        - 3.0e-3: Free water
    """
    adc_arr = np.asarray(adc_values, dtype=np.float64)

    return generate(
        adc=adc_arr,
        s0=s0,
        b_values=b_values,
        snr=snr,
        noise_model=noise_model,
        seed=seed,
    )
