"""Apparent Diffusion Coefficient (ADC) fitting.

This module provides functions for fitting ADC from diffusion-weighted MRI data
using various least squares methods.

Example:
    ```python
    import numpy as np
    from qmri.diffusion import adc

    b_values = np.array([0, 500, 1000, 2000])
    signal = np.array([1000, 606, 368, 135])
    result = adc.fit(signal, b_values, method="iwlls")
    print(f"ADC: {result.adc:.2e} mm²/s")
    # ADC: 1.00e-03 mm²/s
    ```

References:
    .. [1] Veraart, J., et al. (2013). "Weighted linear least squares estimation
           of diffusion MRI parameters: Strengths, limitations, and pitfalls."
           NeuroImage 81:335-346.
    .. [2] Basser, P.J., Mattiello, J., Le Bihan, D. (1994). "Estimation of the
           effective self-diffusion tensor from the NMR spin echo."
           J Magn Reson B 103(3):247-254.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ADCResult:
    """Result of ADC fitting for a single voxel or signal.

    Attributes:
        adc: Apparent Diffusion Coefficient in mm²/s.
        s0: Baseline signal intensity (b=0).
        r_squared: Coefficient of determination (0 to 1). Higher values indicate
            better fit quality.
        iterations: Number of iterations performed (only for IWLLS method).
    """

    adc: float
    s0: float
    r_squared: float
    iterations: int | None = None


@dataclass(frozen=True)
class ADCMapResult:
    """Result of ADC fitting for an entire volume.

    Attributes:
        adc: ADC map in mm²/s. Same spatial dimensions as input.
        s0: Baseline signal (b=0) map.
        r_squared: R² quality map. Values close to 1 indicate good fits.
    """

    adc: NDArray[np.floating]
    s0: NDArray[np.floating]
    r_squared: NDArray[np.floating]


FittingMethod = Literal["lls", "wlls", "iwlls"]


def _calculate_r_squared(
    observed: NDArray[np.floating], predicted: NDArray[np.floating]
) -> float:
    r"""Calculate coefficient of determination (R²).

    Args:
        observed: Measured signal values.
        predicted: Model-predicted signal values.

    Returns:
        R² value between 0 and 1.

    Notes:
        $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

        where $SS_{res} = \sum_i (y_i - \hat{y}_i)^2$ and
        $SS_{tot} = \sum_i (y_i - \bar{y})^2$.
    """
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def fit_lls(signal: NDArray[np.floating], b_values: NDArray[np.floating]) -> ADCResult:
    r"""Fit ADC using Linear Least Squares (LLS).

    Args:
        signal: Signal intensities at each b-value. Must be positive.
        b_values: Diffusion weighting values in s/mm².

    Returns:
        Fitted ADC, S0, and R² quality metric.

    Notes:
        Implements the standard LLS estimator [1]_:

        $$\hat{\boldsymbol{\beta}}_{LLS}
        = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

        where $\mathbf{y} = \ln(\mathbf{S})$ and design matrix
        $\mathbf{X} = [\mathbf{1}, -\mathbf{b}]$.

        The method assumes SNR > 2 for unbiased estimation [2]_.

    References:
        .. [1] Basser, P.J., et al. (1994). J Magn Reson B 103(3):247-254.
        .. [2] Salvador, R., et al. (2005). Hum Brain Mapp 24(2):144-155.
    """
    # Filter valid signals (must be positive for log transform)
    valid_mask = signal > 0
    valid_signal = signal[valid_mask]
    valid_b = b_values[valid_mask]

    if len(valid_signal) < 2:
        return ADCResult(adc=0.0, s0=0.0, r_squared=0.0)

    # Linearise: ln(S) = ln(S0) - b*ADC
    y = np.log(valid_signal)

    # Construct design matrix: [1, -b]
    x_mat = np.column_stack([np.ones_like(valid_b), -valid_b])

    # Solve normal equations: beta = (X'X)^-1 X'y
    xtx = x_mat.T @ x_mat
    xty = x_mat.T @ y
    beta = np.linalg.solve(xtx, xty)

    # Extract parameters
    ln_s0 = float(beta[0])
    adc_val = float(beta[1])
    s0_val = np.exp(ln_s0)

    # Calculate R²
    predicted = s0_val * np.exp(-valid_b * adc_val)
    r_squared = _calculate_r_squared(valid_signal, predicted)

    return ADCResult(adc=adc_val, s0=s0_val, r_squared=r_squared)


def fit_wlls(signal: NDArray[np.floating], b_values: NDArray[np.floating]) -> ADCResult:
    r"""Fit ADC using Weighted Linear Least Squares (WLLS).

    Args:
        signal: Signal intensities at each b-value. Must be positive.
        b_values: Diffusion weighting values in s/mm².

    Returns:
        Fitted ADC, S0, and R² quality metric.

    Notes:
        Implements the WLLS2 estimator from Veraart et al. (2013) [1]_:

        $$\hat{\boldsymbol{\beta}}_{WLLS}
        = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{y}$$

        where $\mathbf{W} = \text{diag}(\exp(2\mathbf{X}\hat{\beta}_{LLS}))$.

        This approach uses predicted signals from an initial LLS fit for weights,
        which provides better accuracy than using noisy measured signals.

    References:
        .. [1] Veraart, J., et al. (2013). NeuroImage 81:335-346.
    """
    # Get initial estimate from LLS
    lls_result = fit_lls(signal, b_values)

    # Filter valid signals
    valid_mask = signal > 0
    valid_signal = signal[valid_mask]
    valid_b = b_values[valid_mask]

    if len(valid_signal) < 2:
        return ADCResult(adc=0.0, s0=0.0, r_squared=0.0)

    # Predict signals from LLS fit
    predicted_signal = lls_result.s0 * np.exp(-valid_b * lls_result.adc)

    # Construct weight matrix from predicted signals
    weights = predicted_signal**2
    w_mat = np.diag(weights)

    # Weighted least squares solution
    y = np.log(valid_signal)
    x_mat = np.column_stack([np.ones_like(valid_b), -valid_b])

    # Solve: beta = (X'WX)^-1 X'Wy
    xtw = x_mat.T @ w_mat
    xtwx = xtw @ x_mat
    xtwy = xtw @ y
    beta = np.linalg.solve(xtwx, xtwy)

    # Extract parameters
    ln_s0 = float(beta[0])
    adc_val = float(beta[1])
    s0_val = np.exp(ln_s0)

    # Calculate R²
    final_predicted = s0_val * np.exp(-valid_b * adc_val)
    r_squared = _calculate_r_squared(valid_signal, final_predicted)

    return ADCResult(adc=adc_val, s0=s0_val, r_squared=r_squared)


def fit_iwlls(
    signal: NDArray[np.floating],
    b_values: NDArray[np.floating],
    *,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> ADCResult:
    r"""Fit ADC using Iterative Weighted Linear Least Squares (IWLLS).

    Args:
        signal: Signal intensities at each b-value. Must be positive.
        b_values: Diffusion weighting values in s/mm².
        max_iterations: Maximum number of iterations. Default is 10.
        tolerance: Convergence tolerance for ADC change. Default is 1e-6.

    Returns:
        Fitted ADC, S0, R² quality metric, and iteration count.

    Notes:
        Implements the iterative WLLS from Veraart et al. (2013) [1]_:

        $$\tilde{\mathbf{W}}_n = \text{diag}(\exp(2\mathbf{X}\hat{\beta}_{n-1}))$$

        The algorithm iteratively refines weights until convergence.
        Typically converges in 2-3 iterations.

    References:
        .. [1] Veraart, J., et al. (2013). NeuroImage 81:335-346.
    """
    # Initial estimate from LLS
    current_result = fit_lls(signal, b_values)

    # Filter valid signals
    valid_mask = signal > 0
    valid_signal = signal[valid_mask]
    valid_b = b_values[valid_mask]

    if len(valid_signal) < 2:
        return ADCResult(adc=0.0, s0=0.0, r_squared=0.0, iterations=0)

    # Prepare matrices
    y = np.log(valid_signal)
    x_mat = np.column_stack([np.ones_like(valid_b), -valid_b])

    iterations_done = 0
    for iteration in range(max_iterations):
        iterations_done = iteration + 1

        # Predict signals from current estimate
        predicted = current_result.s0 * np.exp(-valid_b * current_result.adc)

        # Update weights
        weights = predicted**2
        w_mat = np.diag(weights)

        # Weighted least squares step
        xtw = x_mat.T @ w_mat
        xtwx = xtw @ x_mat
        xtwy = xtw @ y
        beta = np.linalg.solve(xtwx, xtwy)

        # Extract new parameters
        new_s0 = float(np.exp(beta[0]))
        new_adc = float(beta[1])

        # Check convergence
        if abs(new_adc - current_result.adc) < tolerance:
            current_result = ADCResult(
                adc=new_adc,
                s0=new_s0,
                r_squared=current_result.r_squared,
            )
            break

        # Update current result
        current_result = ADCResult(
            adc=new_adc,
            s0=new_s0,
            r_squared=current_result.r_squared,
        )

    # Final R² calculation
    final_predicted = current_result.s0 * np.exp(-valid_b * current_result.adc)
    r_squared = _calculate_r_squared(valid_signal, final_predicted)

    return ADCResult(
        adc=current_result.adc,
        s0=current_result.s0,
        r_squared=r_squared,
        iterations=iterations_done,
    )


def fit(
    signal: NDArray[np.floating],
    b_values: NDArray[np.floating],
    *,
    method: FittingMethod = "iwlls",
    mask: NDArray[np.bool_] | None = None,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> ADCResult | ADCMapResult:
    """Fit ADC from diffusion-weighted signal data.

    This is the main entry point for ADC fitting. It automatically handles
    both single-voxel signals (1D) and multi-dimensional volumes.

    Args:
        signal: Signal intensities. For single voxel: 1D array of length n_bvalues.
            For volumes: ND array where the last dimension is n_bvalues.
        b_values: Diffusion weighting values in s/mm². Length must match last
            dimension of signal.
        method: Fitting method. Default is "iwlls" (recommended).
            - "lls": Linear Least Squares (fastest, least accurate)
            - "wlls": Weighted Linear Least Squares
            - "iwlls": Iterative Weighted Linear Least Squares (recommended)
        mask: Binary mask for volume processing. Only voxels where mask is True
            will be fitted. Shape must match signal shape excluding last dimension.
        max_iterations: Maximum iterations for IWLLS. Default is 10.
        tolerance: Convergence tolerance for IWLLS. Default is 1e-6.

    Returns:
        For 1D input: ADCResult with scalar values.
        For ND input: ADCMapResult with arrays matching input spatial dims.

    Example:
        Single voxel fitting:

        ```python
        import numpy as np
        from qmri.diffusion import adc

        b_values = np.array([0, 500, 1000, 2000])
        signal = np.array([1000, 606, 368, 135])
        result = adc.fit(signal, b_values)
        print(f"ADC: {result.adc:.2e} mm²/s, R²: {result.r_squared:.3f}")
        ```

        Volume fitting:

        ```python
        dwi_4d = np.random.rand(64, 64, 30, 4) * 1000  # (x, y, z, b)
        result = adc.fit(dwi_4d, b_values, method="iwlls")
        print(f"ADC map shape: {result.adc.shape}")
        ```
    """
    signal = np.asarray(signal)
    b_values = np.asarray(b_values)

    # Single voxel case (1D signal)
    if signal.ndim == 1:
        if method == "lls":
            return fit_lls(signal, b_values)
        elif method == "wlls":
            return fit_wlls(signal, b_values)
        elif method == "iwlls":
            return fit_iwlls(
                signal, b_values, max_iterations=max_iterations, tolerance=tolerance
            )
        else:
            msg = f"Unknown method: {method}. Use 'lls', 'wlls', or 'iwlls'."
            raise ValueError(msg)

    # Volume case (ND signal)
    spatial_shape = signal.shape[:-1]
    n_voxels = int(np.prod(spatial_shape))

    # Reshape to (n_voxels, n_bvalues)
    signal_2d = signal.reshape(n_voxels, -1)

    # Initialise output arrays
    adc_flat = np.zeros(n_voxels, dtype=np.float64)
    s0_flat = np.zeros(n_voxels, dtype=np.float64)
    r2_flat = np.zeros(n_voxels, dtype=np.float64)

    # Handle mask
    if mask is not None:
        mask_flat = mask.reshape(n_voxels)
    else:
        mask_flat = np.ones(n_voxels, dtype=bool)

    # Fit each voxel
    for i in range(n_voxels):
        if not mask_flat[i]:
            continue

        voxel_signal = signal_2d[i, :]

        if method == "lls":
            result = fit_lls(voxel_signal, b_values)
        elif method == "wlls":
            result = fit_wlls(voxel_signal, b_values)
        elif method == "iwlls":
            result = fit_iwlls(
                voxel_signal,
                b_values,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
        else:
            msg = f"Unknown method: {method}. Use 'lls', 'wlls', or 'iwlls'."
            raise ValueError(msg)

        adc_flat[i] = result.adc
        s0_flat[i] = result.s0
        r2_flat[i] = result.r_squared

    # Reshape back to original spatial dimensions
    return ADCMapResult(
        adc=adc_flat.reshape(spatial_shape),
        s0=s0_flat.reshape(spatial_shape),
        r_squared=r2_flat.reshape(spatial_shape),
    )


def signal_model(
    s0: NDArray[np.floating] | float,
    adc: NDArray[np.floating] | float,
    b_values: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Generate DWI signal using the mono-exponential diffusion model.

    Args:
        s0: Baseline signal intensity (at b=0).
        adc: Apparent Diffusion Coefficient in mm²/s.
        b_values: Diffusion weighting values in s/mm².

    Returns:
        Predicted signal at each b-value.

    Notes:
        Implements the Stejskal-Tanner equation:

        $$S(b) = S_0 \exp(-b \cdot \text{ADC})$$

    Example:
        ```python
        import numpy as np
        from qmri.diffusion import adc

        b_values = np.array([0, 500, 1000, 2000])
        signal = adc.signal_model(s0=1000, adc=1e-3, b_values=b_values)
        print(signal.round(0))
        # [1000.  607.  368.  135.]
        ```
    """
    b_values = np.asarray(b_values)
    result: NDArray[np.floating] = np.asarray(s0) * np.exp(-b_values * np.asarray(adc))
    return result
